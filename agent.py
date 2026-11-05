"""Etapa de razonamiento LangChain para una conversación telefónica."""

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from call_session import CallSession, CallSessionState
from voice_config import VoiceAgentSettings
from voice_events import AgentTextEvent
import asyncio

logger = logging.getLogger(__name__)


class AgentConfigurationError(RuntimeError):
    """El adaptador LLM solicitado no cuenta con configuración válida."""

def create_voice_agent(settings: VoiceAgentSettings):
    """Crea un agente sin herramientas y con memoria aislada por thread ID."""
    if not settings.is_llm_configured:
        raise AgentConfigurationError("Faltan LLM_PROVIDER o LLM_MODEL")
    if settings.llm_provider != "openai":
        raise AgentConfigurationError(
            f"LLM_PROVIDER no soportado: {settings.llm_provider}. Usa openai."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise AgentConfigurationError("Falta configurar OPENAI_API_KEY")

    model = ChatOpenAI(model=settings.llm_model, temperature=0.2)
    return create_agent(
        model=model,
        tools=[],
        system_prompt=settings.agent_system_prompt,
        checkpointer=InMemorySaver(),
    )


def _text_from_content(content: Any) -> str:
    """Extrae texto de contenido simple o de bloques multimodales."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, str):
                text_blocks.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str):
                    text_blocks.append(text)
        return "".join(text_blocks)
    return ""


async def stream_agent_response(
    agent: Any, *, call_session_id: str, transcript: str
) -> AsyncIterator[str]:
    """Genera tokens de respuesta a partir de una transcripción final."""
    stream = agent.astream_events(
        {"messages": [HumanMessage(content=transcript)]},
        config={"configurable": {"thread_id": call_session_id}},
        version="v2",
    )
    async for event in stream:
        if event.get("event") != "on_chat_model_stream":
            continue
        chunk = event.get("data", {}).get("chunk")
        text = _text_from_content(getattr(chunk, "content", ""))
        logger.info("text: %s", text)
        if text:
            yield text


async def run_agent_pipeline(session: CallSession, settings: VoiceAgentSettings) -> None:
    """Consume transcripciones finales y publica tokens en la cola TTS."""
    try:
        agent = create_voice_agent(settings)
    except AgentConfigurationError as error:
        logger.error("Agente no iniciado para sesión %s: %s", session.call_session_id, error)
        return

    last_transcript: str | None = None
    try:
        while True:
            transcript_event = await session.transcript_queue.get()
            if transcript_event is None:
                return
            if not transcript_event.is_final or not transcript_event.transcript.strip():
                continue
            if transcript_event.transcript == last_transcript:
                continue

            last_transcript = transcript_event.transcript
            response_task = asyncio.create_task(
                _generate_agent_response(
                    agent,
                    session=session,
                    transcript=transcript_event.transcript,
                )
            )
            session.start_agent_response_task(response_task)
            try:
                await response_task
            except asyncio.CancelledError:
                if session.state is not CallSessionState.ACTIVE:
                    raise
                # Barge-in cancela solamente este turno; el worker queda listo
                # para escuchar la siguiente transcripción final del usuario.
                logger.info("Respuesta del agente interrumpida [sesión=%s]", session.call_session_id)
            finally:
                session.clear_agent_response_task(response_task)
    except Exception:
        logger.exception("El agente terminó con un error en la sesión %s", session.call_session_id)


async def _generate_agent_response(
    agent: Any, *, session: CallSession, transcript: str
) -> None:
    """Genera un turno y publica el marcador final sólo si termina normalmente."""
    token_count = 0
    async for token in stream_agent_response(
        agent,
        call_session_id=session.call_session_id,
        transcript=transcript,
    ):
        if not session.enqueue_agent_text(AgentTextEvent(text=token)):
            logger.warning(
                "Token del agente descartado: cola TTS saturada en sesión %s",
                session.call_session_id,
            )
            break
        token_count += 1

    session.enqueue_agent_text(AgentTextEvent(text="", is_final=True))
    logger.info(
        "Respuesta del agente terminada [sesión=%s, tokens=%d]",
        session.call_session_id,
        token_count,
    )
