"""TTS streaming de Telnyx y adaptación de audio para media bidireccional."""

import asyncio
import base64
import json
import logging
import os
import sys
from array import array
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from urllib.parse import urlencode

import websockets
from websockets.exceptions import WebSocketException

from call_session import CallSession
from voice_config import VoiceAgentSettings


logger = logging.getLogger(__name__)
TELNYX_TTS_WS_URL = "wss://api.telnyx.com/v2/text-to-speech/speech"
TELNYX_MEDIA_SAMPLE_RATE = 16_000
TELNYX_MEDIA_FRAME_MILLISECONDS = 20
TELNYX_MEDIA_FRAME_BYTES = (
    TELNYX_MEDIA_SAMPLE_RATE * TELNYX_MEDIA_FRAME_MILLISECONDS // 1000 * 2
)
MIN_TEXT_CHARS = 40
MAX_TEXT_CHARS = 180

SendMediaPayload = Callable[[bytes], Awaitable[None]]


class TTSConfigurationError(RuntimeError):
    """La configuración de síntesis no permite iniciar TTS."""


def pcm_s16le_to_l16(pcm_s16le: bytes) -> bytes:
    """Prepara PCM S16LE para el payload ``L16`` de Media Streaming Telnyx.

    Telnyx denomina ``L16`` a este codec, pero sus payloads WebSocket usan
    muestras signed 16-bit little-endian. No se deben intercambiar bytes.
    """
    if len(pcm_s16le) % 2:
        raise ValueError("El audio PCM S16LE debe contener muestras completas")
    return pcm_s16le


class PCMResampler:
    """Remuestreador lineal streaming para PCM S16LE mono.

    Conserva la última muestra entre chunks para evitar cortes y no depende de
    ``audioop``, eliminado de las versiones modernas de Python.
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Las frecuencias de audio deben ser positivas")
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._step = source_rate / target_rate
        self._position = 0.0
        self._samples: list[int] = []

    def feed(self, pcm_s16le: bytes) -> bytes:
        if len(pcm_s16le) % 2:
            raise ValueError("El audio PCM S16LE debe contener muestras completas")
        if not pcm_s16le:
            return b""
        if self.source_rate == self.target_rate:
            return pcm_s16le

        input_samples = array("h")
        input_samples.frombytes(pcm_s16le)
        if sys.byteorder != "little":
            input_samples.byteswap()
        self._samples.extend(input_samples)

        output = array("h")
        while self._position + 1 < len(self._samples):
            left_index = int(self._position)
            fraction = self._position - left_index
            left = self._samples[left_index]
            right = self._samples[left_index + 1]
            output.append(round(left + (right - left) * fraction))
            self._position += self._step

        consumed = int(self._position)
        if consumed:
            self._samples = self._samples[consumed:]
            self._position -= consumed

        if sys.byteorder != "little":
            output.byteswap()
        return output.tobytes()


class MediaFramePacker:
    """Agrupa PCM a 20 ms para enviarlo como payload L16 de Telnyx."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, pcm_s16le: bytes) -> Iterator[bytes]:
        self._buffer.extend(pcm_s16le)
        while len(self._buffer) >= TELNYX_MEDIA_FRAME_BYTES:
            pcm_frame = bytes(self._buffer[:TELNYX_MEDIA_FRAME_BYTES])
            del self._buffer[:TELNYX_MEDIA_FRAME_BYTES]
            yield pcm_s16le_to_l16(pcm_frame)

    def flush(self) -> Iterator[bytes]:
        if not self._buffer:
            return
        # El último paquete conserva el ritmo de 20 ms solicitado por Telnyx.
        self._buffer.extend(b"\x00" * (TELNYX_MEDIA_FRAME_BYTES - len(self._buffer)))
        yield pcm_s16le_to_l16(bytes(self._buffer))
        self._buffer.clear()


def split_text_for_streaming(buffer: str, *, final: bool) -> tuple[list[str], str]:
    """Extrae fragmentos pronunciables sin esperar toda la respuesta del LLM."""
    fragments: list[str] = []
    pending = buffer.strip()

    while len(pending) >= MIN_TEXT_CHARS:
        candidates = [
            pending.rfind(marker, 0, MAX_TEXT_CHARS + 1)
            for marker in (". ", "? ", "! ", ", ", "; ", ": ")
        ]
        split_at = max(candidates)
        if split_at < MIN_TEXT_CHARS - 1:
            if len(pending) <= MAX_TEXT_CHARS:
                break
            split_at = pending.rfind(" ", 0, MAX_TEXT_CHARS + 1)
            if split_at <= 0:
                split_at = MAX_TEXT_CHARS
        else:
            split_at += 1

        fragment = pending[:split_at].strip()
        if fragment:
            fragments.append(fragment)
        pending = pending[split_at:].strip()

    if final and pending:
        fragments.append(pending)
        pending = ""
    return fragments, pending


async def _agent_response_events(session: CallSession) -> AsyncIterator[str]:
    """Produce una respuesta del agente; el marcador final cierra el turno."""
    while True:
        event = await session.agent_text_queue.get()
        if event is None or event.is_final:
            return
        if event.text:
            yield event.text


async def _send_agent_text(
    websocket: websockets.ClientConnection,
    tokens: AsyncIterator[str],
) -> None:
    """Envía al TTS frases parciales y cierra su entrada al acabar el turno."""
    await websocket.send(json.dumps({"text": " "}))
    pending = ""
    async for token in tokens:
        pending += token
        fragments, pending = split_text_for_streaming(pending, final=False)
        for fragment in fragments:
            await websocket.send(json.dumps({"text": fragment}))

    fragments, _ = split_text_for_streaming(pending, final=True)
    for fragment in fragments:
        await websocket.send(json.dumps({"text": fragment}))
    # Telnyx finaliza la sesión de TTS tras un frame de texto vacío.
    await websocket.send(json.dumps({"text": ""}))


def _tts_websocket_url(settings: VoiceAgentSettings) -> str:
    if not settings.tts_voice:
        raise TTSConfigurationError("Falta TTS_VOICE")
    query = urlencode(
        {"voice": settings.tts_voice, "audio_format": settings.tts_audio_format}
    )
    return f"{TELNYX_TTS_WS_URL}?{query}"


async def stream_telnyx_tts(
    tokens: AsyncIterator[str],
    settings: VoiceAgentSettings,
    *,
    session: CallSession,
    interrupted: asyncio.Event,
) -> AsyncIterator[bytes]:
    """Recibe PCM S16LE incremental del WebSocket TTS de Telnyx."""
    api_key = os.getenv("TELNYX_API_KEY")
    if not api_key:
        raise TTSConfigurationError("Falta configurar TELNYX_API_KEY")
    if not settings.is_streaming_tts_configured:
        raise TTSConfigurationError(
            "Configura TTS_PROVIDER=telnyx, TTS_VOICE y TTS_AUDIO_FORMAT=pcm"
        )

    try:
        async with websockets.connect(
            _tts_websocket_url(settings),
            additional_headers={"Authorization": f"Bearer {api_key}"},
            open_timeout=10,
            ping_interval=20,
            max_size=2**20,
        ) as websocket:
            force_lock = asyncio.Lock()
            force_sent = False

            async def force_current_synthesis() -> None:
                """Detiene el worker TTS remoto y cierra este turno local."""
                nonlocal force_sent
                async with force_lock:
                    if force_sent:
                        return
                    force_sent = True
                    try:
                        await websocket.send(json.dumps({"force": True}))
                    except WebSocketException:
                        # La conexión puede haberse cerrado entre la señal y el envío.
                        pass
                    await websocket.close()

            session.set_tts_interrupt_handler(force_current_synthesis)
            if interrupted.is_set():
                await force_current_synthesis()
                return

            sender = asyncio.create_task(_send_agent_text(websocket, tokens))
            try:
                async for raw_message in websocket:
                    if interrupted.is_set():
                        await force_current_synthesis()
                        return
                    try:
                        message = json.loads(raw_message)
                        if message.get("error"):
                            raise TTSConfigurationError(
                                f"Telnyx rechazó la síntesis: {message['error']}"
                            )
                        encoded_audio = message.get("audio")
                        if not encoded_audio:
                            continue
                        yield base64.b64decode(encoded_audio, validate=True)
                    except (TypeError, ValueError, json.JSONDecodeError) as error:
                        logger.warning("Chunk inválido recibido desde TTS de Telnyx: %s", error)
            finally:
                if not sender.done():
                    sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
                session.clear_tts_interrupt_handler(force_current_synthesis)
    except (OSError, WebSocketException) as error:
        raise TTSConfigurationError("No fue posible conectar con TTS streaming de Telnyx") from error


async def run_streaming_tts_pipeline(
    session: CallSession,
    settings: VoiceAgentSettings,
    send_media_payload: SendMediaPayload,
) -> None:
    """Convierte cada turno del agente a RTP L16 y lo entrega a Telnyx."""
    if not settings.is_streaming_tts_configured:
        logger.error("TTS no iniciado: configura TTS_VOICE y TTS_AUDIO_FORMAT=pcm")
        return

    while True:
        first_event = await session.agent_text_queue.get()
        if first_event is None:
            return
        if first_event.is_final or not first_event.text:
            continue

        async def current_response() -> AsyncIterator[str]:
            yield first_event.text
            async for token in _agent_response_events(session):
                yield token

        interrupted = session.begin_output()
        resampler = PCMResampler(settings.tts_pcm_sample_rate, TELNYX_MEDIA_SAMPLE_RATE)
        packer = MediaFramePacker()
        sent_frames = 0
        try:
            async for pcm_chunk in stream_telnyx_tts(
                current_response(),
                settings,
                session=session,
                interrupted=interrupted,
            ):
                if interrupted.is_set():
                    break
                pcm_16khz = resampler.feed(pcm_chunk)
                for l16_frame in packer.feed(pcm_16khz):
                    if interrupted.is_set():
                        break
                    await send_media_payload(l16_frame)
                    sent_frames += 1
            if interrupted.is_set():
                logger.info("TTS interrumpido [sesión=%s]", session.call_session_id)
            else:
                for l16_frame in packer.flush():
                    await send_media_payload(l16_frame)
                    sent_frames += 1
                logger.info(
                    "TTS reproducido [sesión=%s, paquetes_20ms=%d]",
                    session.call_session_id,
                    sent_frames,
                )
        except asyncio.CancelledError:
            raise
        except (TTSConfigurationError, ValueError):
            logger.exception("No fue posible reproducir TTS para sesión %s", session.call_session_id)
        except Exception:
            logger.exception(
                "Error inesperado al enviar audio TTS para sesión %s",
                session.call_session_id,
            )
        finally:
            session.finish_output()
