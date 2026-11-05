"""Contratos tipados intercambiados entre las etapas del agente de voz."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioChunkEvent:
    audio: bytes
    sample_rate: int
    encoding: str = "pcm_s16le"


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    transcript: str
    is_final: bool
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class AgentTextEvent:
    text: str
    is_final: bool = False


@dataclass(frozen=True, slots=True)
class TTSChunkEvent:
    audio: bytes
    sample_rate: int
    encoding: str = "pcm_s16le"
