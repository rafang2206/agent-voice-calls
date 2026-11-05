"""Adaptadores de TTS streaming intercambiables para el agente de voz."""

import asyncio
import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from urllib.parse import urlencode

import websockets
from websockets.exceptions import ConnectionClosedOK, WebSocketException
from websockets.protocol import State

from call_session import CallSession
from tts import TTSConfigurationError, stream_telnyx_tts
from voice_config import VoiceAgentSettings


logger = logging.getLogger(__name__)
ASSEMBLYAI_TTS_HOSTS = {
    "us": "tts.us.assemblyai.com",
    "eu": "tts.eu.assemblyai.com",
}
ELEVENLABS_TTS_WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech"


class StreamingTTSAdapter(ABC):
    """Contrato común: tokens de LLM a PCM S16LE de una sola respuesta."""

    sample_rate: int

    @abstractmethod
    async def stream_audio(
        self,
        tokens: AsyncIterator[str],
        *,
        session: CallSession,
        interrupted: asyncio.Event,
    ) -> AsyncIterator[bytes]:
        """Produce audio de un turno y respeta la señal de barge-in."""

    async def close(self) -> None:
        """Libera recursos al terminar la llamada."""


class TelnyxStreamingTTSAdapter(StreamingTTSAdapter):
    """Adaptador del TTS Telnyx existente; conserva su protocolo actual."""

    def __init__(self, settings: VoiceAgentSettings) -> None:
        self.settings = settings
        self.sample_rate = settings.tts_pcm_sample_rate

    async def stream_audio(
        self,
        tokens: AsyncIterator[str],
        *,
        session: CallSession,
        interrupted: asyncio.Event,
    ) -> AsyncIterator[bytes]:
        async for chunk in stream_telnyx_tts(
            tokens,
            self.settings,
            session=session,
            interrupted=interrupted,
        ):
            yield chunk


class AssemblyAIStreamingTTSAdapter(StreamingTTSAdapter):
    """Cliente de AssemblyAI Streaming TTS mantenido abierto durante la llamada."""

    def __init__(self, settings: VoiceAgentSettings) -> None:
        if not settings.assemblyai_tts_voice_id:
            raise TTSConfigurationError("Falta configurar TTS_ASSEMBLYAI_VOICE_ID")
        if settings.assemblyai_tts_region not in ASSEMBLYAI_TTS_HOSTS:
            raise TTSConfigurationError("TTS_ASSEMBLYAI_REGION debe ser us o eu")
        if not os.getenv("ASSEMBLYAI_API_KEY"):
            raise TTSConfigurationError("Falta configurar ASSEMBLYAI_API_KEY")

        self.settings = settings
        self.sample_rate = settings.tts_pcm_sample_rate
        self._websocket: websockets.ClientConnection | None = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()

    def _url(self) -> str:
        query: dict[str, str | int] = {
            "voice_id": self.settings.assemblyai_tts_voice_id or "",
            "sample_rate": self.sample_rate,
            "encoding": "pcm_s16le",
            # No necesitamos timings de palabra para reproducir RTP.
            "word_boundaries": "false",
        }
        if self.settings.assemblyai_tts_language:
            query["language"] = self.settings.assemblyai_tts_language
        return f"wss://{ASSEMBLYAI_TTS_HOSTS[self.settings.assemblyai_tts_region]}/v1/ws?{urlencode(query)}"

    async def _connect(self) -> websockets.ClientConnection:
        async with self._connect_lock:
            if self._websocket and self._websocket.state is State.OPEN:
                return self._websocket

            websocket = await websockets.connect(
                self._url(),
                additional_headers={"Authorization": os.environ["ASSEMBLYAI_API_KEY"]},
                compression=None,
                open_timeout=10,
                ping_interval=20,
                max_size=2**20,
            )
            try:
                first_frame = json.loads(await websocket.recv())
                if first_frame.get("type") == "Error":
                    raise TTSConfigurationError(
                        f"AssemblyAI TTS rechazó la sesión: {first_frame.get('error')}"
                    )
                if first_frame.get("type") != "Begin":
                    raise TTSConfigurationError("AssemblyAI TTS no devolvió el frame Begin")

                configuration = first_frame.get("configuration", {})
                actual_rate = configuration.get("sample_rate")
                if actual_rate != self.sample_rate:
                    raise TTSConfigurationError(
                        "AssemblyAI TTS devolvió una frecuencia distinta a "
                        f"TTS_PCM_SAMPLE_RATE ({actual_rate!r})"
                    )
                self._websocket = websocket
                logger.info("AssemblyAI TTS conectado: sesión=%s", first_frame.get("id"))
                return websocket
            except Exception:
                await websocket.close()
                raise

    async def _send(self, message: dict[str, str]) -> None:
        websocket = await self._connect()
        async with self._send_lock:
            await websocket.send(json.dumps(message))

    async def stream_audio(
        self,
        tokens: AsyncIterator[str],
        *,
        session: CallSession,
        interrupted: asyncio.Event,
    ) -> AsyncIterator[bytes]:
        websocket = await self._connect()
        sender: asyncio.Task[None] | None = None
        cancelled = False
        async def send_tokens() -> None:
            async for token in tokens:
                if interrupted.is_set():
                    return
                if token:
                    await self._send({"type": "Generate", "text": token})
            if not interrupted.is_set():
                await self._send({"type": "Flush"})

        async def cancel_current_synthesis() -> None:
            nonlocal cancelled
            if cancelled:
                return
            cancelled = True
            if sender and not sender.done():
                sender.cancel()
            try:
                await self._send({"type": "Cancel"})
            except WebSocketException:
                # La llamada se está cerrando; Telnyx clear seguirá vaciando RTP.
                pass

        session.set_tts_interrupt_handler(cancel_current_synthesis)
        try:
            if interrupted.is_set():
                await cancel_current_synthesis()
                return

            sender = asyncio.create_task(send_tokens())
            async for raw_message in websocket:
                message = json.loads(raw_message)
                frame_type = message.get("type")
                if frame_type == "Error":
                    raise TTSConfigurationError(
                        f"AssemblyAI TTS: {message.get('error', 'error desconocido')}"
                    )
                if frame_type == "Audio":
                    if not cancelled:
                        yield base64.b64decode(message["audio"], validate=True)
                elif frame_type == "FlushDone" and not cancelled:
                    return
                elif frame_type == "Cancelled":
                    return
                elif frame_type == "Termination":
                    self._websocket = None
                    return
        except (OSError, WebSocketException, json.JSONDecodeError, ValueError) as error:
            self._websocket = None
            raise TTSConfigurationError("Error en AssemblyAI Streaming TTS") from error
        finally:
            if sender and not sender.done():
                sender.cancel()
            if sender:
                await asyncio.gather(sender, return_exceptions=True)
            session.clear_tts_interrupt_handler(cancel_current_synthesis)

    async def close(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if not websocket:
            return
        try:
            async with self._send_lock:
                await websocket.send(json.dumps({"type": "Terminate"}))
        except WebSocketException:
            pass
        finally:
            await websocket.close()


class ElevenLabsStreamingTTSAdapter(StreamingTTSAdapter):
    """TTS incremental de ElevenLabs, con un WebSocket por turno del agente.

    El WebSocket estándar es el apropiado para una única respuesta de un agente
    que llega por tokens. Al cerrar el texto con ``{"text": ""}``, ElevenLabs
    hace flush y finaliza el turno. No necesitamos multi-contexto porque el
    pipeline procesa un único turno por sesión a la vez.
    """

    def __init__(self, settings: VoiceAgentSettings) -> None:
        if not settings.elevenlabs_tts_voice_id:
            raise TTSConfigurationError("Falta configurar TTS_ELEVENLABS_VOICE_ID")
        if not os.getenv("ELEVENLABS_API_KEY"):
            raise TTSConfigurationError("Falta configurar ELEVENLABS_API_KEY")
        if not settings.elevenlabs_tts_output_format.startswith("pcm_"):
            raise TTSConfigurationError(
                "TTS_ELEVENLABS_OUTPUT_FORMAT debe ser PCM, por ejemplo pcm_24000"
            )
        try:
            self.sample_rate = int(settings.elevenlabs_tts_output_format.split("_", 1)[1])
        except ValueError as error:
            raise TTSConfigurationError(
                "TTS_ELEVENLABS_OUTPUT_FORMAT debe indicar la frecuencia PCM"
            ) from error
        if self.sample_rate != settings.tts_pcm_sample_rate:
            raise TTSConfigurationError(
                "TTS_PCM_SAMPLE_RATE debe coincidir con "
                "TTS_ELEVENLABS_OUTPUT_FORMAT"
            )

        self.settings = settings
        self._websocket: websockets.ClientConnection | None = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()

    def _url(self) -> str:
        query: dict[str, str | int] = {
            "model_id": self.settings.elevenlabs_tts_model,
            "output_format": self.settings.elevenlabs_tts_output_format,
            "inactivity_timeout": self.settings.elevenlabs_tts_inactivity_timeout,
            "sync_alignment": "false",
        }
        if self.settings.elevenlabs_tts_language_code:
            query["language_code"] = self.settings.elevenlabs_tts_language_code
        voice_id = self.settings.elevenlabs_tts_voice_id or ""
        return f"{ELEVENLABS_TTS_WS_URL}/{voice_id}/stream-input?{urlencode(query)}"

    async def _connect(self) -> websockets.ClientConnection:
        async with self._connect_lock:
            if self._websocket and self._websocket.state is State.OPEN:
                return self._websocket
            try:
                self._websocket = await websockets.connect(
                    self._url(),
                    additional_headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                    compression=None,
                    open_timeout=10,
                    ping_interval=20,
                    max_size=2**20,
                )
                logger.info("ElevenLabs TTS conectado")
                return self._websocket
            except (OSError, WebSocketException) as error:
                self._websocket = None
                raise TTSConfigurationError("No fue posible conectar con ElevenLabs TTS") from error

    async def _send(self, message: dict[str, object]) -> None:
        websocket = await self._connect()
        async with self._send_lock:
            await websocket.send(json.dumps(message))

    async def stream_audio(
        self,
        tokens: AsyncIterator[str],
        *,
        session: CallSession,
        interrupted: asyncio.Event,
    ) -> AsyncIterator[bytes]:
        websocket = await self._connect()
        sender: asyncio.Task[None] | None = None
        cancelled = False

        async def send_tokens() -> None:
            # El espacio inicial abre la sesión sin sintetizar texto. La clave se
            # envía en el header, por lo que nunca entra en logs de aplicación.
            await self._send({"text": " "})
            async for token in tokens:
                if interrupted.is_set():
                    return
                if token:
                    await self._send({"text": token})
            if not interrupted.is_set():
                # En el WebSocket estándar este frame solicita el flush final.
                await self._send({"text": ""})

        async def cancel_current_synthesis() -> None:
            nonlocal cancelled
            if cancelled:
                return
            cancelled = True
            if sender and not sender.done():
                sender.cancel()
            try:
                # El protocolo estándar no cuenta con cancel por contexto:
                # cerrar el socket corta generación y audio pendiente.
                await websocket.close()
                self._websocket = None
            except WebSocketException:
                self._websocket = None

        session.set_tts_interrupt_handler(cancel_current_synthesis)
        try:
            if interrupted.is_set():
                await cancel_current_synthesis()
                return

            sender = asyncio.create_task(send_tokens())
            while True:
                try:
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=1)
                except TimeoutError:
                    if sender and sender.done():
                        # No ocultar un fallo del productor de texto: antes se
                        # perdía y el receptor quedaba esperando indefinidamente.
                        sender.result()
                    if cancelled:
                        self._websocket = None
                        return
                    continue

                message = json.loads(raw_message)
                if message.get("error"):
                    raise TTSConfigurationError(
                        f"ElevenLabs TTS: {message['error']}"
                    )
                encoded_audio = message.get("audio")
                if encoded_audio and not cancelled:
                    logger.debug(
                        "ElevenLabs TTS audio [bytes_base64=%d]",
                        len(encoded_audio),
                    )
                    yield base64.b64decode(encoded_audio, validate=True)
                # ElevenLabs documenta snake_case, pero la API también entrega
                # camelCase en algunos modelos/regions; aceptamos ambas formas.
                if message.get("is_final", message.get("isFinal", False)):
                    return
        except ConnectionClosedOK:
            # El barge-in cierra deliberadamente este socket para cortar la
            # síntesis. ``recv`` despierta con 1000/OK: no es un error del
            # proveedor ni debe convertirse en un TTSConfigurationError.
            self._websocket = None
            if cancelled or interrupted.is_set():
                logger.debug("ElevenLabs TTS cerrado por barge-in")
                return
            # ElevenLabs puede cerrar tras entregar su último frame incluso si
            # no envía is_final. El turno ya no puede producir más audio.
            logger.warning("ElevenLabs TTS cerró normalmente antes de is_final")
            return
        except (OSError, WebSocketException, json.JSONDecodeError, ValueError) as error:
            self._websocket = None
            logger.error("Error en ElevenLabs Streaming TTS: %s", error)
            raise TTSConfigurationError("Error en ElevenLabs Streaming TTS") from error
        finally:
            if sender and not sender.done():
                sender.cancel()
            if sender:
                await asyncio.gather(sender, return_exceptions=True)
            session.clear_tts_interrupt_handler(cancel_current_synthesis)
            if self._websocket is websocket:
                self._websocket = None
                if websocket.state is State.OPEN:
                    await websocket.close()

    async def close(self) -> None:
        websocket = self._websocket
        self._websocket = None
        if not websocket:
            return
        try:
            await websocket.close()
        except WebSocketException:
            pass


def create_streaming_tts_adapter(settings: VoiceAgentSettings) -> StreamingTTSAdapter:
    """Fábrica central para seleccionar proveedor sin tocar el pipeline RTP."""
    if settings.tts_provider == "telnyx":
        return TelnyxStreamingTTSAdapter(settings)
    if settings.tts_provider == "assemblyai":
        return AssemblyAIStreamingTTSAdapter(settings)
    if settings.tts_provider == "elevenlabs":
        return ElevenLabsStreamingTTSAdapter(settings)
    raise TTSConfigurationError(f"TTS_PROVIDER no soportado: {settings.tts_provider}")
