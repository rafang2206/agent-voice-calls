"""Punto de entrada para llamadas entrantes de Telnyx.

Variables necesarias:
    TELNYX_API_KEY: API key de Telnyx usada para controlar la llamada.
    TELNYX_PUBLIC_KEY: clave pública Ed25519 del portal de Telnyx para validar webhooks.
    TELNYX_STREAM_URL: URL pública WSS a la que Telnyx enviará el audio.

Opcional:
    TELNYX_STREAM_TRACK: inbound_track (por defecto), outbound_track o both_tracks.
    TELNYX_STREAM_CODEC: L16 (por defecto) u otro codec soportado por Telnyx.
    LLM_PROVIDER y LLM_MODEL: se usarán cuando se conecte el agente LangChain.
    TTS_PROVIDER=telnyx o assemblyai selecciona el adaptador TTS streaming.
    TTS_PCM_SAMPLE_RATE: frecuencia PCM de la voz TTS, 24000 por defecto.
"""

import base64
import binascii
import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from urllib.parse import quote
from uuid import NAMESPACE_URL, uuid5
from dotenv import load_dotenv

import httpx
from agent import run_agent_pipeline
from call_session import CallSession, CallSessionRegistry
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette import status
from tts_pipeline import run_streaming_tts_pipeline
from voice_config import VoiceAgentSettings
from voice_pipeline import VoicePipelineConfigurationError, create_stt_runnable


load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
VOICE_AGENT_SETTINGS = VoiceAgentSettings.from_env()

TELNYX_API_BASE_URL = "https://api.telnyx.com/v2"
VALID_STREAM_TRACKS = {"inbound_track", "outbound_track", "both_tracks"}
VALID_STREAM_CODECS = {"PCMU", "PCMA", "G722", "OPUS", "AMR-WB", "L16", "default"}
WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS = 300

app = FastAPI(title="Telnyx Voice Agent")
CALL_SESSIONS = CallSessionRegistry()


def get_stream_track() -> str:
    """Obtiene y valida el track de audio que Telnyx debe enviar."""
    stream_track = os.getenv("TELNYX_STREAM_TRACK", "inbound_track")
    if stream_track not in VALID_STREAM_TRACKS:
        raise RuntimeError(
            "TELNYX_STREAM_TRACK debe ser inbound_track, outbound_track o both_tracks"
        )
    return stream_track


def get_stream_codec() -> str:
    """Obtiene y valida el codec de audio entregado por Telnyx."""
    configured_codec = os.getenv("TELNYX_STREAM_CODEC", "L16")
    stream_codec = "default" if configured_codec.lower() == "default" else configured_codec.upper()
    if stream_codec not in VALID_STREAM_CODECS:
        raise RuntimeError(
            "TELNYX_STREAM_CODEC debe ser PCMU, PCMA, G722, OPUS, AMR-WB, L16 o default"
        )
    return stream_codec


def l16_to_pcm_s16le(l16_audio: bytes) -> bytes:
    """Normaliza el payload ``L16`` de Media Streaming de Telnyx a PCM S16LE.

    Aunque ``L16`` suele designar muestras big-endian en RTP, Telnyx entrega los
    payloads de Media Streaming como ``s16le``. Invertir bytes en este punto
    convertiría una onda continua en muestras de amplitud errática y el STT no
    podría reconocerla.
    """
    if len(l16_audio) % 2:
        raise ValueError("Un chunk L16 debe contener un número par de bytes")
    return l16_audio


def _g711_to_pcm_s16le(g711_audio: bytes, *, alaw: bool) -> bytes:
    """Decodifica G.711 μ-law o A-law a PCM signed 16-bit little-endian."""
    pcm_s16le = bytearray(len(g711_audio) * 2)

    for index, encoded_sample in enumerate(g711_audio):
        if alaw:
            value = encoded_sample ^ 0x55
            sample = (value & 0x0F) << 4
            segment = (value & 0x70) >> 4
            if segment == 0:
                sample += 8
            elif segment == 1:
                sample += 0x108
            else:
                sample = (sample + 0x108) << (segment - 1)
            sample = sample if value & 0x80 else -sample
        else:
            value = ~encoded_sample & 0xFF
            sample = (((value & 0x0F) << 3) + 0x84) << ((value >> 4) & 0x07)
            sample = 0x84 - sample if value & 0x80 else sample - 0x84

        offset = index * 2
        pcm_s16le[offset : offset + 2] = sample.to_bytes(2, "little", signed=True)

    return bytes(pcm_s16le)


def normalize_telnyx_audio(audio: bytes, encoding: str) -> bytes:
    """Normaliza los codecs soportados por el receptor a PCM S16LE."""
    codec = encoding.upper()
    if codec == "L16":
        return l16_to_pcm_s16le(audio)
    if codec == "PCMU":
        return _g711_to_pcm_s16le(audio, alaw=False)
    if codec == "PCMA":
        return _g711_to_pcm_s16le(audio, alaw=True)
    raise ValueError(f"Codec de stream no soportado aún: {encoding}")


async def queued_audio_stream(audio_queue: asyncio.Queue[bytes | None]):
    """Convierte los chunks del WebSocket de Telnyx en un stream asíncrono."""
    while True:
        audio_chunk = await audio_queue.get()
        if audio_chunk is None:
            return
        yield audio_chunk


async def run_stt_pipeline(
    session: CallSession,
    *,
    input_sample_rate: int,
    on_user_speech: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Ejecuta STT y deja los transcripts finales listos para el agente."""
    try:
        stt_pipeline = create_stt_runnable(input_sample_rate=input_sample_rate)
        async for transcript_event in stt_pipeline.atransform(
            queued_audio_stream(session.audio_queue)
        ):
            if not transcript_event.is_final:
                if on_user_speech and transcript_event.transcript.strip():
                    await on_user_speech()
                logger.debug(
                    "STT parcial [stream=%s, caracteres=%d]",
                    session.stream_id,
                    len(transcript_event.transcript),
                )
                # Los parciales sirven para barge-in; no deben saturar la cola
                # que el agente reserva para turnos finales.
                continue

            # Algunos turnos cortos llegan únicamente como final. También son
            # habla del usuario y deben detener el audio actual antes de que el
            # agente procese el nuevo contexto.
            if on_user_speech and transcript_event.transcript.strip():
                await on_user_speech()

            if not session.enqueue_transcript(transcript_event):
                logger.warning(
                    "Transcript descartado: la sesión %s no está disponible",
                    session.call_session_id,
                )
                continue

            if VOICE_AGENT_SETTINGS.log_transcripts:
                logger.info(
                    "STT final [stream=%s]: %s",
                    session.stream_id,
                    transcript_event.transcript,
                )
            else:
                logger.info(
                    "STT final [stream=%s, caracteres=%d]",
                    session.stream_id,
                    len(transcript_event.transcript),
                )
            # La tarea del agente consume este evento desde transcript_queue.
    except VoicePipelineConfigurationError as error:
        logger.error("STT no iniciado: %s", error)
    except Exception:
        logger.exception("El pipeline STT terminó con un error")


def verify_telnyx_webhook(raw_body: bytes, headers: Mapping[str, str]) -> None:
    """Valida la firma Ed25519 de Telnyx y limita reintentos fuera de tiempo."""
    public_key = os.getenv("TELNYX_PUBLIC_KEY")
    signature = headers.get("telnyx-signature-ed25519")
    timestamp = headers.get("telnyx-timestamp")

    if not public_key:
        raise RuntimeError("Falta configurar TELNYX_PUBLIC_KEY")
    if not signature or not timestamp:
        raise ValueError("Faltan los headers de firma de Telnyx")

    try:
        if abs(time.time() - int(timestamp)) > WEBHOOK_TIMESTAMP_TOLERANCE_SECONDS:
            raise ValueError("El timestamp del webhook está fuera de la ventana permitida")

        public_key_bytes = base64.b64decode(public_key, validate=True)
        signature_bytes = base64.b64decode(signature, validate=True)
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature_bytes,
            timestamp.encode("utf-8") + b"|" + raw_body,
        )
    except (binascii.Error, InvalidSignature, TypeError, ValueError) as error:
        raise ValueError("Firma de webhook Telnyx inválida") from error


async def answer_call_with_stream(
    *, call_control_id: str, event_id: str | None
) -> None:
    """Contesta una llamada y solicita el stream de media en una única orden."""
    api_key = os.getenv("TELNYX_API_KEY")
    stream_url = os.getenv("TELNYX_STREAM_URL")

    if not api_key:
        raise RuntimeError("Falta configurar TELNYX_API_KEY")
    if not stream_url or not stream_url.startswith("wss://"):
        raise RuntimeError("TELNYX_STREAM_URL debe ser una URL pública que empiece por wss://")

    # El mismo command_id para un reintento del webhook hace la operación idempotente.
    command_id = event_id or str(uuid5(NAMESPACE_URL, call_control_id))
    endpoint = (
        f"{TELNYX_API_BASE_URL}/calls/"
        f"{quote(call_control_id, safe='')}/actions/answer"
    )
    command = {
        "command_id": command_id,
        "stream_url": stream_url,
        "stream_track": get_stream_track(),
        # Se solicita L16; el receptor normaliza también el fallback PCMU/PCMA.
        "stream_codec": get_stream_codec(),
        # El mismo WebSocket recibe audio y permite devolver RTP L16 al llamante.
        "stream_bidirectional_mode": "rtp",
        "stream_bidirectional_codec": "L16",
        "stream_bidirectional_sampling_rate": 16_000,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            json=command,
        )
        response.raise_for_status()


@app.get("/")
async def healthcheck() -> dict[str, str | int]:
    return {"status": "ok", "active_sessions": await CALL_SESSIONS.active_count()}


@app.post("/call", status_code=status.HTTP_200_OK)
async def call_receive_endpoint(request: Request) -> dict[str, str]:
    """Recibe eventos de Telnyx y contesta las llamadas entrantes.

    Al responder ``call.initiated`` con ``stream_url``, Telnyx contesta la llamada
    y abre una conexión hacia ``/ws`` (o la URL WSS pública configurada) para
    enviar los frames de audio.
    """
    try:
        raw_body = await request.body()
        verify_telnyx_webhook(raw_body, request.headers)
        event = json.loads(raw_body)
        event_data = event["data"]
        event_type = event_data["event_type"]
    except RuntimeError:
        logger.exception("Configuración de validación de Telnyx incompleta")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuración de Telnyx incompleta",
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Webhook inválido")

    event_payload = event_data.get("payload", {})
    if not isinstance(event_payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Webhook inválido")
    call_session_id = event_payload.get("call_session_id")

    if event_type != "call.initiated":
        if event_type in {"streaming.stopped", "call.hangup"} and isinstance(
            call_session_id, str
        ):
            if await CALL_SESSIONS.close(call_session_id):
                logger.info("Sesión finalizada: %s", call_session_id)

        # Todo evento distinto queda confirmado; no debe volver a contestar la llamada.
        logger.info("Evento Telnyx confirmado: %s", event_type)
        return {"status": "acknowledged"}

    call_control_id = event_payload.get("call_control_id")
    if not call_control_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="call_control_id ausente en call.initiated",
        )

    try:
        await answer_call_with_stream(
            call_control_id=call_control_id,
            event_id=event_data.get("id"),
        )
    except (httpx.HTTPError, RuntimeError):
        # Un 5xx hace que Telnyx reintente el webhook. command_id evita doble respuesta.
        logger.exception("No fue posible contestar la llamada de Telnyx")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No fue posible ejecutar el comando answer de Telnyx",
        )

    if isinstance(call_session_id, str):
        await CALL_SESSIONS.get_or_create(
            call_session_id,
            call_control_id=call_control_id,
        )

    logger.info("Llamada contestada; Telnyx abrirá el stream para %s", call_control_id)
    return {"status": "answered_and_streaming"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Acepta el stream de media de Telnyx.

    Cada ``media.payload`` se decodifica desde Base64 y se normaliza a PCM signed
    16-bit little-endian. L16, PCMU y PCMA quedan preparados para un STT streaming.
    """
    await websocket.accept()
    logger.info("WebSocket de media conectado")

    media_format: dict[str, object] | None = None
    stream_encoding: str | None = None
    session: CallSession | None = None

    try:
        while True:
            message = await websocket.receive_json()
            event_type = message.get("event", "unknown")

            if event_type == "start":
                start_data = message.get("start", {})
                candidate_format = (
                    start_data.get("media_format") if isinstance(start_data, dict) else None
                )
                media_format = candidate_format if isinstance(candidate_format, dict) else None
                configured_encoding = media_format.get("encoding") if media_format else None
                stream_encoding = (
                    configured_encoding.upper()
                    if isinstance(configured_encoding, str)
                    else None
                )
                logger.info(
                    "Stream iniciado: id=%s, formato=%s",
                    message.get("stream_id"),
                    media_format,
                )

                if stream_encoding == "L16":
                    logger.info("Stream L16 confirmado")
                elif stream_encoding in {"PCMU", "PCMA"}:
                    logger.warning(
                        "Telnyx entregó %s; se aplicará conversión G.711 a PCM S16LE",
                        stream_encoding,
                    )
                else:
                    logger.warning(
                        "Telnyx entregó un codec no soportado: %s",
                        stream_encoding or "desconocido",
                    )

                sample_rate = media_format.get("sample_rate") if media_format else None
                call_session_id = (
                    start_data.get("call_session_id") if isinstance(start_data, dict) else None
                )
                call_control_id = (
                    start_data.get("call_control_id") if isinstance(start_data, dict) else None
                )
                stream_id = message.get("stream_id")
                if not isinstance(call_session_id, str) or not isinstance(stream_id, str):
                    logger.warning("Pipeline STT no iniciado: stream sin call_session_id o stream_id")
                elif isinstance(sample_rate, int) and sample_rate in {8_000, 16_000}:
                    session = await CALL_SESSIONS.get_or_create(
                        call_session_id,
                        call_control_id=call_control_id
                        if isinstance(call_control_id, str)
                        else None,
                    )
                    session.attach_stream(
                        stream_id,
                        call_control_id
                        if isinstance(call_control_id, str)
                        else None,
                    )
                    if VOICE_AGENT_SETTINGS.is_llm_configured:
                        agent_task = asyncio.create_task(
                            run_agent_pipeline(session, VOICE_AGENT_SETTINGS)
                        )
                        session.start_agent_task(agent_task)
                    else:
                        logger.warning(
                            "Agente LangChain no iniciado: configura LLM_PROVIDER y LLM_MODEL"
                        )

                    async def interrupt_for_user_speech() -> None:
                        if not await session.interrupt_active_output():
                            return
                        await websocket.send_json({"event": "clear"})
                        logger.info(
                            "Barge-in aplicado: TTS interrumpido y buffer vaciado "
                            "[sesión=%s]",
                            session.call_session_id,
                        )

                    if VOICE_AGENT_SETTINGS.is_streaming_tts_configured:
                        async def send_tts_media(l16_payload: bytes) -> None:
                            await websocket.send_json(
                                {
                                    "event": "media",
                                    "media": {
                                        "payload": base64.b64encode(l16_payload).decode("ascii")
                                    },
                                }
                            )

                        tts_task = asyncio.create_task(
                            run_streaming_tts_pipeline(
                                session,
                                VOICE_AGENT_SETTINGS,
                                send_tts_media,
                            )
                        )
                        session.start_tts_task(tts_task)
                        if (
                            VOICE_AGENT_SETTINGS.agent_greeting
                            and session.enqueue_greeting(VOICE_AGENT_SETTINGS.agent_greeting)
                        ):
                            logger.info(
                                "Saludo inicial encolado [sesión=%s]",
                                session.call_session_id,
                            )
                    else:
                        logger.warning(
                            "TTS no iniciado: configura el proveedor, la voz y "
                            "TTS_AUDIO_FORMAT=pcm"
                        )
                    stt_task = asyncio.create_task(
                        run_stt_pipeline(
                            session,
                            input_sample_rate=sample_rate,
                            on_user_speech=interrupt_for_user_speech,
                        )
                    )
                    session.start_stt_task(stt_task)
                    logger.info("Pipeline STT iniciado para audio a %s Hz", sample_rate)
                else:
                    logger.warning(
                        "Pipeline STT no iniciado: sample rate no soportado (%s)",
                        sample_rate,
                    )
            elif event_type == "media":
                if not stream_encoding:
                    logger.warning("Chunk descartado: aún no se recibió el formato del stream")
                    continue

                try:
                    encoded_audio = base64.b64decode(
                        message["media"]["payload"], validate=True
                    )
                    pcm_s16le = normalize_telnyx_audio(encoded_audio, stream_encoding)
                except (KeyError, TypeError, ValueError, binascii.Error):
                    logger.warning("Chunk de audio inválido o con codec no soportado descartado")
                    continue

                # Siguiente paso: enviar pcm_s16le al proveedor de STT en streaming.
                logger.debug("Chunk PCM S16LE recibido: %d bytes", len(pcm_s16le))
                if session and session.stt_task and not session.stt_task.done():
                    if not session.enqueue_audio(pcm_s16le):
                        logger.warning("Chunk descartado: el pipeline STT está saturado")

                continue
            elif event_type == "stop":
                logger.info("Stream detenido: %s", message.get("stream_id"))
                break
            else:
                logger.debug("Evento de media recibido: %s", event_type)
    except WebSocketDisconnect:
        logger.info("WebSocket de media desconectado")
    finally:
        if session and await CALL_SESSIONS.close(session.call_session_id):
            logger.info("Sesión de media cerrada: %s", session.call_session_id)




