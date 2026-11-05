"""Configuración no secreta del agente de voz leída desde el entorno."""

import os
from dataclasses import dataclass


DEFAULT_AGENT_SYSTEM_PROMPT = """Eres un asistente telefónico profesional y amable.
Responde siempre en español, de forma breve y natural para ser escuchada por voz.
No uses Markdown, emojis, listas visuales ni frases excesivamente largas.
Confirma los datos importantes antes de realizar una acción y no inventes información.
Si no entiendes al usuario, pídele que repita o reformule su solicitud.
"""


@dataclass(frozen=True)
class VoiceAgentSettings:
    """Ajustes que utilizarán los adaptadores LLM y TTS en las siguientes fases."""

    llm_provider: str
    llm_model: str
    agent_system_prompt: str
    agent_greeting: str | None
    tts_provider: str
    tts_voice: str | None
    tts_audio_format: str
    tts_pcm_sample_rate: int
    tts_language: str
    tts_service_level: str
    assemblyai_tts_voice_id: str | None
    assemblyai_tts_language: str | None
    assemblyai_tts_region: str
    elevenlabs_tts_voice_id: str | None
    elevenlabs_tts_model: str
    elevenlabs_tts_output_format: str
    elevenlabs_tts_language_code: str | None
    elevenlabs_tts_inactivity_timeout: int
    log_transcripts: bool

    @property
    def is_llm_configured(self) -> bool:
        return bool(self.llm_provider and self.llm_model)

    @property
    def is_tts_configured(self) -> bool:
        if self.tts_provider == "telnyx":
            return bool(self.tts_voice)
        if self.tts_provider == "assemblyai":
            return bool(self.assemblyai_tts_voice_id)
        if self.tts_provider == "elevenlabs":
            return bool(self.elevenlabs_tts_voice_id)
        return False

    @property
    def is_streaming_tts_configured(self) -> bool:
        """Indica si la voz puede usarse en el WebSocket TTS con salida PCM."""
        return self.is_tts_configured and self.tts_audio_format == "pcm"

    @classmethod
    def from_env(cls) -> "VoiceAgentSettings":
        try:
            tts_pcm_sample_rate = int(os.getenv("TTS_PCM_SAMPLE_RATE", "24000"))
            elevenlabs_inactivity_timeout = int(
                os.getenv("TTS_ELEVENLABS_INACTIVITY_TIMEOUT", "180")
            )
        except ValueError as error:
            raise ValueError("Las frecuencias y timeouts TTS deben ser enteros") from error

        if tts_pcm_sample_rate <= 0:
            raise ValueError("TTS_PCM_SAMPLE_RATE debe ser mayor que cero")
        if not 1 <= elevenlabs_inactivity_timeout <= 180:
            raise ValueError("TTS_ELEVENLABS_INACTIVITY_TIMEOUT debe estar entre 1 y 180")

        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            agent_system_prompt=os.getenv(
                "AGENT_SYSTEM_PROMPT", DEFAULT_AGENT_SYSTEM_PROMPT
            ).strip(),
            agent_greeting=os.getenv(
                "AGENT_GREETING", "Hola, ¿en qué te puedo ayudar?"
            ).strip()
            or None,
            tts_provider=os.getenv("TTS_PROVIDER", "telnyx").strip().lower(),
            tts_voice=os.getenv("TTS_VOICE", "").strip() or None,
            tts_audio_format=os.getenv("TTS_AUDIO_FORMAT", "pcm").strip().lower(),
            tts_pcm_sample_rate=tts_pcm_sample_rate,
            tts_language=os.getenv("TTS_LANGUAGE", "es-US").strip(),
            tts_service_level=os.getenv("TTS_SERVICE_LEVEL", "premium").strip(),
            assemblyai_tts_voice_id=(
                os.getenv("TTS_ASSEMBLYAI_VOICE_ID", "").strip() or None
            ),
            assemblyai_tts_language=(
                os.getenv("TTS_ASSEMBLYAI_LANGUAGE", "").strip() or None
            ),
            assemblyai_tts_region=os.getenv("TTS_ASSEMBLYAI_REGION", "us")
            .strip()
            .lower(),
            elevenlabs_tts_voice_id=(
                os.getenv("TTS_ELEVENLABS_VOICE_ID", "").strip() or None
            ),
            elevenlabs_tts_model=os.getenv(
                "TTS_ELEVENLABS_MODEL", "eleven_multilingual_v2"
            ).strip(),
            elevenlabs_tts_output_format=os.getenv(
                "TTS_ELEVENLABS_OUTPUT_FORMAT", "pcm_24000"
            )
            .strip()
            .lower(),
            elevenlabs_tts_language_code=(
                os.getenv("TTS_ELEVENLABS_LANGUAGE_CODE", "").strip() or None
            ),
            elevenlabs_tts_inactivity_timeout=elevenlabs_inactivity_timeout,
            log_transcripts=os.getenv("LOG_TRANSCRIPTS", "false").lower() == "true",
        )
