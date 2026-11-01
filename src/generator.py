"""Answer generation with a vision LLM via litellm.

``MultimodalGenerator.generate(question, context)`` sends the retrieved text/table evidence and the
retrieved figures (as base64 images) to a vision-capable model and returns a ``GenerationResult``
with the answer, the sources it cited, token usage, cost (when litellm knows the price) and latency.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .retriever import ContextItem, RetrievedContext

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "I couldn't find anything in the indexed documents that answers this question."

DEFAULT_SYSTEM_PROMPT = """\
You answer questions about documents using ONLY the sources provided.

Sources are labelled [S1], [S2], ... Some are text passages or Markdown tables; others are figures \
(charts, diagrams, photos) shown as images, each introduced by its label and caption.

Rules:
- Use both the text sources and the figures. Read values off charts carefully and say when a value \
is approximate because it was read from a chart.
- Cite every factual claim with its source label in square brackets, e.g. "Revenue was $61.0M [S2]." \
Cite figures the same way.
- If sources disagree, say so and cite both.
- If the sources do not contain the answer, say you could not find it. Do not use outside knowledge \
or guess.
- Be concise: answer first, then any brief supporting detail."""

_CITATION_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
_RETRYABLE_HINTS = ("rate", "429", "quota", "timeout", "temporarily", "overloaded", "503", "502", "500")


@dataclass(slots=True, frozen=True)
class TokenUsage:
    """Token counts reported by the provider (0 when not reported)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


@dataclass(slots=True, frozen=True)
class GenerationResult:
    """The answer plus everything needed to evaluate and audit it.

    Attributes:
        answer: Model output (or ``NO_CONTEXT_ANSWER`` when retrieval found nothing).
        model: Model that produced the answer (may be a fallback).
        citations: Context items the answer cited, in first-cited order.
        usage: Tokens for the successful call.
        total_usage: Tokens across all attempts, including failed fallbacks that reported usage.
        cost_usd: Estimated cost from litellm's price table, ``None`` if unknown (e.g. local models).
        llm_latency_ms: Time spent in the successful LLM call.
        total_latency_ms: Generation wall time including image encoding, retries and fallbacks.
        retrieval_latency_ms: Copied from the context, so end-to-end time is one sum away.
        images_sent: Number of figures sent as images.
        attempts: LLM calls made (retries and fallbacks included).
        finish_reason: Provider finish reason, e.g. ``"stop"`` or ``"length"``.
    """

    answer: str
    model: str
    citations: list[ContextItem] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float | None = None
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    images_sent: int = 0
    attempts: int = 0
    finish_reason: str | None = None

    @property
    def end_to_end_latency_ms(self) -> float:
        return self.retrieval_latency_ms + self.total_latency_ms

    def sources_text(self) -> str:
        """Formatted source list for display, e.g. ``[S2] report.pdf, page 2 (table)``."""
        return "\n".join(f"[{c.label}] {c.citation}" for c in self.citations)


class GenerationError(RuntimeError):
    """Raised when every model (primary and fallbacks) failed."""


class MultimodalGenerator:
    """Builds text + image prompts and calls a vision LLM through ``litellm.acompletion``.

    Args:
        model: litellm model string from config, e.g. ``gemini/gemini-3.8-flash`` or
            ``ollama/llama3.2-vision``.
        fallback_models: Tried in order if the primary model keeps failing.
        temperature: Sampling temperature (low keeps answers grounded).
        max_tokens: Maximum answer length.
        max_retries: Attempts per model for retryable errors (rate limits, timeouts, 5xx).
        max_image_side: Longest image side in pixels before downscaling; keeps token use down.
        api_base: Custom endpoint, e.g. ``http://localhost:11434`` for Ollama.
        system_prompt: Override the default grounding instructions.
        timeout: Seconds per LLM call.
        send_images: ``True``/``False`` forces images on/off; ``None`` (default) asks litellm's model
            map and sends images unless the model is known not to support vision.
    """

    def __init__(
        self,
        model: str,
        fallback_models: Sequence[str] = (),
        temperature: float = 0.1,
        max_tokens: int = 1024,
        max_retries: int = 3,
        max_image_side: int = 1568,
        api_base: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout: float = 120.0,
        send_images: bool | None = None,
    ) -> None:
        if not model:
            raise ValueError("model is required (set it in config)")
        self.model = model
        self.fallback_models = list(fallback_models)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_image_side = max_image_side
        self.api_base = api_base
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.send_images = send_images

    # ------------------------------------------------------------------ prompt building

    def build_messages(
        self, question: str, context: RetrievedContext, include_images: bool = True
    ) -> list[dict[str, Any]]:
        """OpenAI-style messages (litellm translates them per provider).

        The user turn holds: the text sources, then each figure as a label + caption text part
        followed by its image part, then the question. With ``include_images=False`` figures are
        given as captions only (for text-only models).
        """
        parts: list[dict[str, Any]] = []
        if context.text_items:
            parts.append({"type": "text", "text": "Text sources:\n\n" + context.text_block()})

        if context.image_items:
            parts.append({"type": "text", "text": "Figure sources:"})
            for item in context.image_items:
                caption = item.content or "(no caption)"
                header = f"[{item.label}] {item.citation}\nCaption: {caption}"
                if include_images and item.image_path:
                    try:
                        data_url = encode_image(item.image_path, self.max_image_side)
                    except Exception:
                        logger.warning(
                            "Could not encode image %s; sending caption only", item.image_path, exc_info=True
                        )
                        parts.append({"type": "text", "text": header + " (image unavailable)"})
                        continue
                    parts.append({"type": "text", "text": header})
                    parts.append({"type": "image_url", "image_url": {"url": data_url}})
                else:
                    parts.append({"type": "text", "text": header + " (image not shown)"})

        parts.append({"type": "text", "text": f"Question: {question}"})
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": parts},
        ]

    # ------------------------------------------------------------------ generation

    async def generate(self, question: str, context: RetrievedContext) -> GenerationResult:
        """Answer ``question`` from ``context``.

        Returns a result without calling the LLM when the context is empty.

        Raises:
            GenerationError: If the primary model and all fallbacks fail.
        """
        start = time.perf_counter()
        if not question.strip():
            raise ValueError("question must not be empty")
        if context.is_empty:
            logger.info("No context retrieved for %r; skipping LLM call", question[:80])
            return GenerationResult(
                answer=NO_CONTEXT_ANSWER,
                model=self.model,
                total_latency_ms=(time.perf_counter() - start) * 1000,
                retrieval_latency_ms=context.latency_ms,
            )

        import litellm

        total_usage = TokenUsage()
        attempts = 0
        errors: list[str] = []
        for model in [self.model, *self.fallback_models]:
            vision = self.send_images if self.send_images is not None else _supports_vision(model)
            if context.image_items and not vision:
                logger.warning("%s may not support images; sending figure captions only", model)
            messages = await asyncio.to_thread(self.build_messages, question, context, vision)
            images_sent = sum(1 for p in messages[1]["content"] if p.get("type") == "image_url")

            for attempt in range(1, self.max_retries + 1):
                attempts += 1
                call_start = time.perf_counter()
                try:
                    response = await litellm.acompletion(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        api_base=self.api_base,
                        timeout=self.timeout,
                    )
                except Exception as exc:  # noqa: BLE001 - provider errors vary; classified below
                    message = f"{model} attempt {attempt}: {type(exc).__name__}: {str(exc)[:200]}"
                    errors.append(message)
                    retryable = any(h in str(exc).lower() for h in _RETRYABLE_HINTS)
                    if retryable and attempt < self.max_retries:
                        delay = 2.0 * 2 ** (attempt - 1)
                        logger.warning("%s; retrying in %.0fs", message, delay)
                        await asyncio.sleep(delay)
                        continue
                    logger.warning("%s; giving up on this model", message)
                    break

                llm_latency_ms = (time.perf_counter() - call_start) * 1000
                usage = _usage(response)
                total_usage = total_usage + usage
                answer = str(response.choices[0].message.content or "").strip()
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                cost = _cost(litellm, response)
                result = GenerationResult(
                    answer=answer,
                    model=model,
                    citations=cited_items(answer, context),
                    usage=usage,
                    total_usage=total_usage,
                    cost_usd=cost,
                    llm_latency_ms=llm_latency_ms,
                    total_latency_ms=(time.perf_counter() - start) * 1000,
                    retrieval_latency_ms=context.latency_ms,
                    images_sent=images_sent,
                    attempts=attempts,
                    finish_reason=finish_reason,
                )
                logger.info(
                    "Generated answer with %s: %d prompt + %d completion tokens, %d images, "
                    "%.0f ms LLM / %.0f ms total, %d attempts, cost=%s, cited=%s",
                    model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    images_sent,
                    llm_latency_ms,
                    result.total_latency_ms,
                    attempts,
                    f"${cost:.5f}" if cost is not None else "n/a",
                    [c.label for c in result.citations],
                )
                if finish_reason == "length":
                    logger.warning("Answer was cut off at max_tokens=%d", self.max_tokens)
                return result

        raise GenerationError("All models failed:\n" + "\n".join(errors))


def encode_image(path: str | Path, max_side: int = 1568) -> str:
    """Return a base64 data URL for an image, downscaled so its longest side is <= ``max_side``."""
    from PIL import Image

    with Image.open(path) as img:
        if max(img.size) <= max_side and img.format in ("PNG", "JPEG"):
            mime = "image/png" if img.format == "PNG" else "image/jpeg"
            data = Path(path).read_bytes()
        else:
            img = img.convert("RGB")
            img.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            mime, data = "image/png", buf.getvalue()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def cited_items(answer: str, context: RetrievedContext) -> list[ContextItem]:
    """Context items referenced as ``[S1]`` / ``[S1, S3]`` in the answer, in first-cited order."""
    seen: dict[str, ContextItem] = {}
    for group in _CITATION_RE.findall(answer):
        for label in re.split(r"\s*,\s*", group):
            item = context.by_label(label)
            if item is not None and label not in seen:
                seen[label] = item
    return list(seen.values())


def _supports_vision(model: str) -> bool:
    """False only when litellm's model map says the model has no vision support.

    Models litellm does not know (new releases, custom names) or that it lists without a vision
    flag (e.g. some Ollama models) are assumed to accept images; set ``send_images`` to override.
    """
    import litellm

    try:
        info = litellm.get_model_info(model=model)
    except Exception:  # noqa: BLE001 - unmapped models raise various errors
        return True
    flag = info.get("supports_vision")
    return flag is not False


def _usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0) or prompt + completion
    return TokenUsage(prompt, completion, total)


def _cost(litellm_module: Any, response: Any) -> float | None:
    try:
        cost = litellm_module.completion_cost(completion_response=response)
    except Exception:  # noqa: BLE001 - model not in litellm's price table (local / new models)
        return None
    return float(cost) if cost else None
