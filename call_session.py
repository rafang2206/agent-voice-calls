"""Estado y recursos efímeros de una conversación telefónica."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from voice_events import AgentTextEvent, TranscriptEvent, TTSChunkEvent


class CallSessionState(StrEnum):
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class CallSession:
    """Recursos aislados de una sola llamada identificada por Telnyx."""

    call_session_id: str
    call_control_id: str | None = None
    stream_id: str | None = None
    state: CallSessionState = CallSessionState.ACTIVE
    audio_queue: asyncio.Queue[bytes | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=100)
    )
    transcript_queue: asyncio.Queue[TranscriptEvent | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=50)
    )
    agent_text_queue: asyncio.Queue[AgentTextEvent | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=200)
    )
    tts_audio_queue: asyncio.Queue[TTSChunkEvent | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=100)
    )
    stt_task: asyncio.Task[None] | None = None
    agent_task: asyncio.Task[None] | None = None
    agent_response_task: asyncio.Task[None] | None = None
    tts_task: asyncio.Task[None] | None = None
    greeting_enqueued: bool = False
    output_active: bool = False
    _output_interrupted: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _tts_interrupt_handler: Callable[[], Awaitable[None]] | None = field(
        default=None, repr=False
    )
    _interruption_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def attach_stream(self, stream_id: str, call_control_id: str | None = None) -> None:
        self.stream_id = stream_id
        if call_control_id:
            self.call_control_id = call_control_id

    def start_stt_task(self, task: asyncio.Task[None]) -> None:
        if self.stt_task and not self.stt_task.done():
            self.stt_task.cancel()
        self.stt_task = task

    def start_agent_task(self, task: asyncio.Task[None]) -> None:
        if self.agent_task and not self.agent_task.done():
            self.agent_task.cancel()
        self.agent_task = task

    def start_agent_response_task(self, task: asyncio.Task[None]) -> None:
        """Registra la generación del turno actual, cancelable por barge-in."""
        if self.agent_response_task and not self.agent_response_task.done():
            self.agent_response_task.cancel()
        # Cada turno conserva su propia señal: el TTS que quedó cerrándose tras
        # un barge-in no puede confundirse con la siguiente respuesta.
        self._output_interrupted = asyncio.Event()
        self.agent_response_task = task

    def clear_agent_response_task(self, task: asyncio.Task[None]) -> None:
        if self.agent_response_task is task:
            self.agent_response_task = None

    def start_tts_task(self, task: asyncio.Task[None]) -> None:
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
        self.tts_task = task

    def begin_output(self) -> asyncio.Event:
        """Prepara un nuevo turno de voz y devuelve su señal de interrupción.

        Un resultado STT parcial puede llegar con retraso tras su final. Por eso
        el turno no se considera audible hasta que TTS entrega el primer RTP.
        """
        self.output_active = False
        return self._output_interrupted

    def mark_output_started(self) -> None:
        """Marca que al menos un paquete de audio del bot salió hacia Telnyx."""
        self.output_active = True

    def finish_output(self) -> None:
        self.output_active = False
        self._tts_interrupt_handler = None

    def set_tts_interrupt_handler(
        self, handler: Callable[[], Awaitable[None]]
    ) -> None:
        self._tts_interrupt_handler = handler

    def clear_tts_interrupt_handler(
        self, handler: Callable[[], Awaitable[None]]
    ) -> None:
        if self._tts_interrupt_handler is handler:
            self._tts_interrupt_handler = None

    async def interrupt_active_output(self) -> bool:
        """Interrumpe el turno actual y descarta texto pendiente del bot.

        El callback registrado por TTS envía ``force`` al proveedor antes de que
        el WebSocket de media vacíe su buffer en Telnyx.
        """
        async with self._interruption_lock:
            response_active = bool(
                self.agent_response_task and not self.agent_response_task.done()
            )
            if (
                self.state is not CallSessionState.ACTIVE
                or self._output_interrupted.is_set()
                or not (response_active or self.output_active)
            ):
                return False

            self._output_interrupted.set()
            self.output_active = False
            self._drain_agent_text_queue()

            if response_active and self.agent_response_task:
                self.agent_response_task.cancel()

            if self._tts_interrupt_handler:
                await self._tts_interrupt_handler()
            return True

    def enqueue_audio(self, audio: bytes) -> bool:
        if self.state is not CallSessionState.ACTIVE:
            return False
        try:
            self.audio_queue.put_nowait(audio)
        except asyncio.QueueFull:
            return False
        return True

    def enqueue_transcript(self, transcript: TranscriptEvent) -> bool:
        if self.state is not CallSessionState.ACTIVE:
            return False
        try:
            self.transcript_queue.put_nowait(transcript)
        except asyncio.QueueFull:
            return False
        return True

    def enqueue_agent_text(self, text: AgentTextEvent) -> bool:
        if self.state is not CallSessionState.ACTIVE:
            return False
        try:
            self.agent_text_queue.put_nowait(text)
        except asyncio.QueueFull:
            return False
        return True

    def enqueue_greeting(self, greeting: str) -> bool:
        """Publica una vez el saludo fijo, sin pasar por el LLM."""
        if self.greeting_enqueued or not greeting.strip():
            return False
        if self.state is not CallSessionState.ACTIVE:
            return False
        if (
            self.agent_text_queue.maxsize
            and self.agent_text_queue.qsize() + 2 > self.agent_text_queue.maxsize
        ):
            return False
        # No hay puntos de espera entre ambos put_nowait; el saludo y su fin
        # quedan en la cola como un único turno consistente.
        self.agent_text_queue.put_nowait(AgentTextEvent(text=greeting.strip()))
        self.agent_text_queue.put_nowait(AgentTextEvent(text="", is_final=True))
        self.greeting_enqueued = True
        return True

    async def close(self) -> None:
        """Detiene las tareas y desbloquea todos los consumidores de colas."""
        if self.state is CallSessionState.CLOSED:
            return

        self.state = CallSessionState.CLOSING
        self._signal_end(self.audio_queue)
        self._signal_end(self.transcript_queue)
        self._signal_end(self.agent_text_queue)
        self._signal_end(self.tts_audio_queue)

        tasks = [
            task
            for task in (
                self.stt_task,
                self.agent_task,
                self.agent_response_task,
                self.tts_task,
            )
            if task and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.state = CallSessionState.CLOSED

    @staticmethod
    def _signal_end(queue: asyncio.Queue[object]) -> None:
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            _ = queue.get_nowait()
            queue.put_nowait(None)

    def _drain_agent_text_queue(self) -> None:
        """Elimina tokens que pertenecen al turno que fue interrumpido."""
        while True:
            try:
                self.agent_text_queue.get_nowait()
            except asyncio.QueueEmpty:
                return


class CallSessionRegistry:
    """Registro local de sesiones activas.

    Para múltiples procesos o réplicas se deberá sustituir por un almacenamiento
    compartido; una sesión de media debe permanecer atendida por la misma réplica.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, CallSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self, call_session_id: str, *, call_control_id: str | None = None
    ) -> CallSession:
        async with self._lock:
            session = self._sessions.get(call_session_id)
            if session is None:
                session = CallSession(
                    call_session_id=call_session_id,
                    call_control_id=call_control_id,
                )
                self._sessions[call_session_id] = session
            elif call_control_id:
                session.call_control_id = call_control_id
            return session

    async def close(self, call_session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(call_session_id, None)
        if not session:
            return False
        await session.close()
        return True

    async def active_count(self) -> int:
        async with self._lock:
            return len(self._sessions)
