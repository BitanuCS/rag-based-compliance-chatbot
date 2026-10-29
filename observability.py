"""Tracing to Langfuse and LangSmith: one trace per chat turn, in both at once.

The two are independent and either, both or neither may be on:

- **Langfuse** is the primary view: spans for retrieval and rerank, the LLM
  generations, sessions per conversation, and 👍/👎 feedback scored onto the
  trace. On when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set
  (LANGFUSE_TRACING_ENABLED=false switches it off with keys set).
- **LangSmith** keeps the existing setup working: the same turn, retrieve and
  rerank runs, with LangChain's LLM runs nested inside, grouped into threads.
  On when LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 is true, which is also what
  makes LangChain trace to it on its own.

Every other module goes through here rather than importing either SDK, for one
reason: with no keys configured, langfuse still builds a client on every
get_client() call and logs an authentication warning each time. Tracing is
optional — the app must run identically without it — so whatever is off is
never touched at all.

What a trace holds, top to bottom (the same tree in both tools):

    compliance-chat-turn   question in, reply out; session = conversation id
      retrieve             fetch_k, floor, best distance, survivors
        rerank             model, score per kept passage
      condense             (follow-ups only) the LLM rewrite
      ChatHuggingFace      the grounded answer — prompt, tokens, latency
    score: user-feedback   👍/👎 from the UI (Langfuse)
"""

import logging
import os
from contextlib import ExitStack, contextmanager
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

TRACE_NAME = "compliance-chat-turn"
TAGS = ["compliance-chat"]
FEEDBACK_SCORE = "user-feedback"


def langfuse_enabled():
    """True when both Langfuse keys are set and tracing is not switched off."""
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower() == "false":
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def langsmith_enabled():
    """True when LangSmith tracing is switched on via the environment."""
    flag = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    return flag.strip().lower() in {"1", "true", "yes"}


def tracing_enabled():
    """Whether anything is being traced at all."""
    return langfuse_enabled() or langsmith_enabled()


@lru_cache(maxsize=1)
def get_langfuse():
    """The process-wide Langfuse client, or None when Langfuse is off."""
    if not langfuse_enabled():
        return None
    from langfuse import get_client

    return get_client()


def _langsmith_run():
    """The LangSmith run currently open in this context, if any."""
    if not langsmith_enabled():
        return None
    from langsmith.run_helpers import get_current_run_tree

    return get_current_run_tree()


def observe(name, as_type="span", summarize=None):
    """Trace a function as a span in whichever tools are on.

    `summarize` turns the return value into what the trace shows as output.
    Without it, outputs are not captured: retrieval returns Documents whose
    full text would bury the numbers worth reading. Decided once, at import of
    the decorated module — late enough, since app.py copies Streamlit secrets
    into the environment before importing rag.
    """

    def decorate(func):
        wrapped = func
        if langsmith_enabled():
            from langsmith import traceable

            # LangSmith's run types are fewer than Langfuse's; "retriever"
            # exists in both, anything else is a chain.
            wrapped = traceable(
                name=name,
                run_type="retriever" if as_type == "retriever" else "chain",
                process_outputs=(
                    (lambda value: {"output": summarize(value)}) if summarize else (lambda _: {})
                ),
            )(wrapped)
        if langfuse_enabled():
            from langfuse import observe as langfuse_observe

            # Langfuse outermost, so its span is open while LangSmith's runs;
            # summarize is applied through annotate() by the caller instead.
            wrapped = langfuse_observe(name=name, as_type=as_type, capture_output=False)(wrapped)
        return wrapped

    return decorate


def annotate(output=None, metadata=None, level=None, status_message=None):
    """Add output/metadata/level to whichever span is currently open, in both tools.

    Never raises: losing a trace annotation is not worth failing a user's answer.
    """
    client = get_langfuse()
    if client is not None:
        fields = {
            key: value
            for key, value in {
                "output": output,
                "metadata": metadata,
                "level": level,
                "status_message": status_message,
            }.items()
            if value is not None
        }
        try:
            client.update_current_span(**fields)
        except Exception:
            logging.debug("langfuse annotate failed", exc_info=True)

    run = _langsmith_run()
    if run is not None:
        # Outputs come from the decorator's summarize; LangSmith has no level, so
        # a warning travels as metadata where it can still be filtered on.
        extra = dict(metadata or {})
        if level:
            extra["level"] = level
        if status_message:
            extra["status_message"] = status_message
        try:
            if extra:
                run.add_metadata(extra)
        except Exception:
            logging.debug("langsmith annotate failed", exc_info=True)


class Turn:
    """Handle on one chat turn's trace, in whichever tools are recording it."""

    def __init__(self, langfuse_span=None, langsmith_run=None):
        self._span = langfuse_span
        self._run = langsmith_run

    @property
    def trace_id(self):
        """The Langfuse trace id — what feedback is scored against."""
        return self._span.trace_id if self._span else None

    def update(self, output=None, metadata=None, level=None, status_message=None):
        if self._span is not None:
            fields = {"output": output, "metadata": metadata, "level": level, "status_message": status_message}
            try:
                self._span.update(**{k: v for k, v in fields.items() if v is not None})
            except Exception:
                logging.debug("langfuse turn update failed", exc_info=True)
        if self._run is not None:
            try:
                extra = dict(metadata or {})
                if level:
                    extra["level"] = level
                if status_message:
                    extra["status_message"] = status_message
                if output is not None:
                    self._run.add_outputs({"output": output})
                if extra:
                    self._run.add_metadata(extra)
            except Exception:
                logging.debug("langsmith turn update failed", exc_info=True)


@contextmanager
def trace_turn(session_id, question, **metadata):
    """Root span for one chat turn. Yields a Turn, or None when nothing traces.

    Everything opened inside nests under it in both tools: the @observe'd
    retrieval steps, and LangChain's LLM runs — which reach Langfuse through its
    callback and LangSmith through its own context. session_id groups turns into
    a Langfuse session and a LangSmith thread; the conversation id is used, so a
    chat in the sidebar, a session and a thread all share one identifier.
    """
    if not tracing_enabled():
        yield None
        return

    with ExitStack() as stack:
        run = span = None

        if langsmith_enabled():
            import langsmith

            run = stack.enter_context(
                langsmith.trace(
                    TRACE_NAME,
                    "chain",
                    inputs={"question": question},
                    tags=TAGS,
                    # LangSmith threads group on session_id; thread_id and
                    # conversation_id are accepted aliases, so all three are set.
                    metadata={
                        "session_id": session_id,
                        "thread_id": session_id,
                        "conversation_id": session_id,
                        **metadata,
                    },
                )
            )

        client = get_langfuse()
        if client is not None:
            from langfuse import propagate_attributes

            span = stack.enter_context(
                client.start_as_current_observation(
                    name=TRACE_NAME,
                    as_type="chain",
                    input={"question": question},
                    metadata=metadata,
                )
            )
            stack.enter_context(
                propagate_attributes(
                    session_id=session_id,
                    trace_name=TRACE_NAME,
                    tags=TAGS,
                    # Propagated attributes must be strings; these become
                    # filterable dimensions on every observation in the trace.
                    metadata={key: str(value) for key, value in metadata.items()},
                )
            )

        yield Turn(span, run)


def langchain_callbacks():
    """Extra callbacks for a LangChain run config.

    Only Langfuse needs one: LangChain finds LangSmith from the environment by
    itself, and passing a LangSmith tracer here as well would record every run
    twice.
    """
    if get_langfuse() is None:
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def record_feedback(trace_id, positive, comment=None):
    """Attach a 👍/👎 to a turn's Langfuse trace. Returns whether it was sent."""
    client = get_langfuse()
    if client is None or not trace_id:
        return False
    try:
        client.create_score(
            name=FEEDBACK_SCORE,
            value=1 if positive else 0,
            data_type="BOOLEAN",
            trace_id=trace_id,
            comment=comment,
        )
        return True
    except Exception:
        logging.warning("could not record feedback for trace %s", trace_id, exc_info=True)
        return False


def flush():
    """Send buffered spans now. For short-lived CLI runs; the app need not call it."""
    client = get_langfuse()
    if client is not None:
        client.flush()
    if langsmith_enabled():
        from langsmith.run_trees import get_cached_client

        try:
            get_cached_client().flush()
        except Exception:
            logging.debug("langsmith flush failed", exc_info=True)
