"""Pipeline de STT streaming para el agente de voz.

El transporte Telnyx entrega PCM S16LE normalizado por ``main.py``. Este módulo
lo agrupa en frames aptos para AssemblyAI y expone los resultados como un
``RunnableGenerator`` de LangChain, listo para encadenarlo con agente y TTS.
"""

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator
from urllib.parse import urlencode

import websockets
from langchain_core.runnables import RunnableGenerator
from voice_events import TranscriptEvent


ASSEMBLYAI_STREAM_URL = "wss://streaming.assemblyai.com/v3/ws"
TARGET_SAMPLE_RATE = 16_000
BYTES_PER_PCM_SAMPLE = 2
STT_FRAME_DURATION_MS = 100
MINIMUM_FRAME_DURATION_MS = 50


class VoicePipelineConfigurationError(RuntimeError):
    """La configuración necesaria para ejecutar el pipeline no existe."""


def upsample_pcm_s16le_8khz_to_16khz(pcm_8khz: bytes) -> bytes:
    """Duplica muestras PCM de 8 kHz a 16 kHz sin dependencias obsoletas.

    La telefonía PSTN contiene como máximo 4 kHz de información útil. Duplicar
    muestras preserva dicha información y produce el formato que pide la sesión
    STT. Si se necesita mayor fidelidad, este punto se puede sustituir por SOXR.
    """
    if len(pcm_8khz) % BYTES_PER_PCM_SAMPLE:
        raise ValueError("PCM S16LE debe contener un número par de bytes")

    pcm_16khz = bytearray(len(pcm_8khz) * 2)
    for source_offset in range(0, len(pcm_8khz), BYTES_PER_PCM_SAMPLE):
        target_offset = source_offset * 2
        sample = pcm_8khz[source_offset : source_offset + BYTES_PER_PCM_SAMPLE]
        pcm_16khz[target_offset : target_offset + 2] = sample
        pcm_16khz[target_offset + 2 : target_offset + 4] = sample
    return bytes(pcm_16khz)


class AssemblyAIStreamingSTT:
    """Cliente asíncrono de AssemblyAI Streaming v3 por llamada."""

    def __init__(self, *, input_sample_rate: int) -> None:
        self.input_sample_rate = input_sample_rate
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.speech_model = os.getenv("ASSEMBLYAI_SPEECH_MODEL", "u3-rt-pro")
        self._websocket: websockets.ClientConnection | None = None
        self._audio_buffer = bytearray()
        self._finished = False

    async def connect(self) -> None:
        if not self.api_key:
            raise VoicePipelineConfigurationError("Falta configurar ASSEMBLYAI_API_KEY")

        query = urlencode(
            {
                "sample_rate": TARGET_SAMPLE_RATE,
                "speech_model": self.speech_model,
                "format_turns": "true",
            }
        )
        self._websocket = await websockets.connect(
            f"{ASSEMBLYAI_STREAM_URL}?{query}",
            additional_headers={"Authorization": self.api_key},
            max_size=1_048_576,
        )

    async def send_audio(self, pcm_s16le: bytes) -> None:
        """Normaliza, agrupa y transmite audio PCM al STT."""
        if not self._websocket:
            raise RuntimeError("La sesión STT no está conectada")

        if self.input_sample_rate == 8_000:
            pcm_s16le = upsample_pcm_s16le_8khz_to_16khz(pcm_s16le)
        elif self.input_sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"Sample rate de Telnyx no soportado: {self.input_sample_rate} Hz"
            )

        self._audio_buffer.extend(pcm_s16le)
        frame_size = TARGET_SAMPLE_RATE * BYTES_PER_PCM_SAMPLE * STT_FRAME_DURATION_MS // 1000

        while len(self._audio_buffer) >= frame_size:
            await self._websocket.send(bytes(self._audio_buffer[:frame_size]))
            del self._audio_buffer[:frame_size]

    async def receive_events(self) -> AsyncIterator[TranscriptEvent]:
        """Convierte los eventos del proveedor en eventos STT normalizados."""
        if not self._websocket:
            raise RuntimeError("La sesión STT no está conectada")

        async for raw_message in self._websocket:
            if not isinstance(raw_message, str):
                continue

            message = json.loads(raw_message)
            if message.get("type") == "Termination":
                return
            if message.get("type") != "Turn":
                continue

            transcript = message.get("transcript", "").strip()
            if transcript:
                yield TranscriptEvent(
                    transcript=transcript,
                    is_final=bool(
                        message.get("end_of_turn", message.get("turn_is_formatted", False))
                    ),
                )

    async def finish(self) -> None:
        """Envía los últimos frames y cierra la sesión del proveedor."""
        if not self._websocket or self._finished:
            return

        self._finished = True
        minimum_frame_size = (
            TARGET_SAMPLE_RATE
            * BYTES_PER_PCM_SAMPLE
            * MINIMUM_FRAME_DURATION_MS
            // 1000
        )
        if len(self._audio_buffer) >= minimum_frame_size:
            await self._websocket.send(bytes(self._audio_buffer))
        self._audio_buffer.clear()
        await self._websocket.send(json.dumps({"type": "Terminate"}))

    async def close(self) -> None:
        if self._websocket:
            await self._websocket.close()


async def _assemblyai_stt_stream(
    audio_stream: AsyncIterator[bytes], *, input_sample_rate: int
) -> AsyncIterator[TranscriptEvent]:
    """Etapa LangChain: ``PCM chunks -> TranscriptEvent``."""
    stt = AssemblyAIStreamingSTT(input_sample_rate=input_sample_rate)
    await stt.connect()

    async def send_audio() -> None:
        try:
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            await stt.finish()

    sender_task = asyncio.create_task(send_audio())
    try:
        async for event in stt.receive_events():
            yield event
    finally:
        if not sender_task.done():
            sender_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender_task
        await stt.close()


def create_stt_runnable(*, input_sample_rate: int) -> RunnableGenerator:
    """Crea la etapa STT de LangChain para una llamada concreta."""
    async def stt_stream(audio_stream: AsyncIterator[bytes]) -> AsyncIterator[TranscriptEvent]:
        async for event in _assemblyai_stt_stream(
            audio_stream,
            input_sample_rate=input_sample_rate,
        ):
            yield event

    return RunnableGenerator(stt_stream)
