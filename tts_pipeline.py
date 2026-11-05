"""Pipeline común: proveedor TTS streaming -> RTP L16 para Telnyx."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from call_session import CallSession
from tts import (
    MediaFramePacker,
    PCMResampler,
    TELNYX_MEDIA_FRAME_MILLISECONDS,
    TELNYX_MEDIA_SAMPLE_RATE,
    TTSConfigurationError,
)
from tts_adapters import StreamingTTSAdapter, create_streaming_tts_adapter
from voice_config import VoiceAgentSettings


logger = logging.getLogger(__name__)
SendMediaPayload = Callable[[bytes], Awaitable[None]]


async def _wait_for_next_rtp_slot(interrupted: asyncio.Event) -> bool:
    """Marca RTP a tiempo real y sale antes si el usuario interrumpe.

    Los proveedores TTS entregan PCM más rápido que su duración. Enviarlo todo
    de inmediato deja audio acumulado en Telnyx y hace que ``clear`` parezca
    tardío. Cada payload de ``MediaFramePacker`` representa exactamente 20 ms.
    """
    try:
        await asyncio.wait_for(
            interrupted.wait(),
            timeout=TELNYX_MEDIA_FRAME_MILLISECONDS / 1_000,
        )
    except TimeoutError:
        return False
    return True


async def _agent_response_events(session: CallSession) -> AsyncIterator[str]:
    while True:
        event = await session.agent_text_queue.get()
        if event is None or event.is_final:
            return
        if event.text:
            yield event.text


async def run_streaming_tts_pipeline(
    session: CallSession,
    settings: VoiceAgentSettings,
    send_media_payload: SendMediaPayload,
) -> None:
    """Reproduce turnos del agente a través del adaptador TTS configurado."""
    try:
        adapter: StreamingTTSAdapter = create_streaming_tts_adapter(settings)
    except TTSConfigurationError as error:
        logger.error("TTS no iniciado: %s", error)
        return

    try:
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
            resampler = PCMResampler(adapter.sample_rate, TELNYX_MEDIA_SAMPLE_RATE)
            packer = MediaFramePacker()
            sent_frames = 0
            output_started = False
            try:
                async for pcm_chunk in adapter.stream_audio(
                    current_response(),
                    session=session,
                    interrupted=interrupted,
                ):
                    if interrupted.is_set():
                        break
                    pcm_16khz = resampler.feed(pcm_chunk)
                    for l16_frame in packer.feed(pcm_16khz):
                        if interrupted.is_set():
                            break
                        if not output_started:
                            session.mark_output_started()
                            output_started = True
                            logger.info(
                                "TTS %s inició audio [sesión=%s]",
                                settings.tts_provider,
                                session.call_session_id,
                            )
                        await send_media_payload(l16_frame)
                        sent_frames += 1
                        if await _wait_for_next_rtp_slot(interrupted):
                            break

                if interrupted.is_set():
                    logger.info("TTS interrumpido [sesión=%s]", session.call_session_id)
                else:
                    for l16_frame in packer.flush():
                        if not output_started:
                            session.mark_output_started()
                            output_started = True
                            logger.info(
                                "TTS %s inició audio [sesión=%s]",
                                settings.tts_provider,
                                session.call_session_id,
                            )
                        await send_media_payload(l16_frame)
                        sent_frames += 1
                        if await _wait_for_next_rtp_slot(interrupted):
                            break
                    if interrupted.is_set():
                        logger.info("TTS interrumpido [sesión=%s]", session.call_session_id)
                    else:
                        logger.info(
                            "TTS %s reproducido [sesión=%s, paquetes_20ms=%d]",
                            settings.tts_provider,
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
    finally:
        await adapter.close()
