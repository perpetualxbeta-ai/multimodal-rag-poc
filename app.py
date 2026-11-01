"""Command-line entry point for the multimodal RAG PoC.

Examples:
    python app.py --ingest samples/halcyon_fy2025_report.pdf
    python app.py --query "What is the Q3 revenue trend shown in the chart?"
    python app.py --ingest docs/ --reset --query "Summarise the key risks"
    python app.py --query "..." --model ollama/llama3.2-vision --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from src.config import Settings, load_settings, missing_api_keys
from src.generator import GenerationError
from src.logging_setup import setup_logging
from src.pipeline import IngestReport, PipelineError, QueryResult, RAGPipeline

logger = logging.getLogger("app")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Multimodal RAG over PDFs: ingest documents, then ask questions about their text, tables and figures.",
    )
    parser.add_argument(
        "--ingest", nargs="+", metavar="PATH", help="PDF files or folders to extract and index"
    )
    parser.add_argument(
        "--query", "-q", metavar="QUESTION", help="question to answer from the indexed documents"
    )
    parser.add_argument("--reset", action="store_true", help="delete the existing index before ingesting")
    parser.add_argument("--config", metavar="FILE", help="config file (default: config.yaml)")
    parser.add_argument(
        "--model", metavar="MODEL", help="override the answer model, e.g. ollama/llama3.2-vision"
    )
    parser.add_argument("--top-k", type=int, metavar="N", help="number of sources to retrieve")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    verbosity.add_argument("--quiet", action="store_true", help="only warnings and errors")
    args = parser.parse_args(argv)
    if not args.ingest and not args.query:
        parser.error("nothing to do: give --ingest PATH and/or --query QUESTION")
    if args.reset and not args.ingest:
        parser.error("--reset only makes sense together with --ingest")
    return args


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Factory used by ``main``; tests replace it to inject fake components."""
    return RAGPipeline.from_settings(settings)


def models_in_use(settings: Settings, ingest: bool, query: bool) -> list[str]:
    models: list[str] = []
    if (ingest or query) and settings.embedding.text.provider == "litellm":
        models.append(settings.embedding.text.model)
    if ingest and settings.embedding.image_strategy in ("caption", "both"):
        models.append(settings.embedding.caption_model)
    if query:
        models.append(settings.llm.model)
    return models


def check_api_keys(settings: Settings, ingest: bool, query: bool) -> str | None:
    """Return an error message if a model has no API key at all; warn about other missing settings."""
    problems = []
    for model, keys in missing_api_keys(models_in_use(settings, ingest, query)).items():
        api_keys = [k for k in keys if k.endswith("_API_KEY")]
        if api_keys and len(api_keys) == len(keys):
            problems.append(f"  {model}: set {' or '.join(api_keys)}")
        else:
            logger.warning("%s: %s not set; using the provider default", model, ", ".join(keys))
    if not problems:
        return None
    return (
        "Missing API key(s):\n" + "\n".join(problems) + "\n"
        "Add them to a .env file in the project folder (see .env.example), e.g.\n"
        "  GEMINI_API_KEY=your-key-from-aistudio.google.com"
    )


def print_ingest(reports: list[IngestReport]) -> None:
    for r in reports:
        if r.ok:
            print(
                f"Indexed {r.path}: {r.text_chunks} text, {r.table_chunks} table, "
                f"{r.image_chunks} figure chunks ({r.points_indexed} vectors) in {r.seconds:.1f}s"
            )
        else:
            print(f"FAILED  {r.path}: {r.error}")


def print_answer(result: QueryResult) -> None:
    gen = result.generation
    print(f"\nQ: {result.question}\n")
    print(gen.answer)
    if gen.citations:
        print("\nSources:")
        print("\n".join(f"  {line}" for line in gen.sources_text().splitlines()))
    if gen.images_sent:
        print(f"\nFigures shown to the model ({gen.images_sent}):")
        for item in result.context.image_items:
            print(f"  [{item.label}] {item.image_path}")
    cost = f"${gen.cost_usd:.5f}" if gen.cost_usd is not None else "n/a"
    print(
        f"\n{gen.model} | tokens {gen.usage.prompt_tokens} in / {gen.usage.completion_tokens} out | "
        f"cost {cost} | retrieval {gen.retrieval_latency_ms:.0f} ms | LLM {gen.llm_latency_ms / 1000:.1f} s | "
        f"total {gen.end_to_end_latency_ms / 1000:.1f} s"
    )


def result_json(reports: list[IngestReport], result: QueryResult | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ingested": [
            {
                "path": r.path,
                "doc_id": r.doc_id,
                "text_chunks": r.text_chunks,
                "table_chunks": r.table_chunks,
                "image_chunks": r.image_chunks,
                "points_indexed": r.points_indexed,
                "seconds": round(r.seconds, 3),
                "error": r.error,
            }
            for r in reports
        ]
    }
    if result is not None:
        gen = result.generation
        out["query"] = {
            "question": result.question,
            "answer": gen.answer,
            "model": gen.model,
            "citations": [
                {"label": c.label, "source": c.citation, "page": c.page_number, "type": c.chunk_type.value}
                for c in gen.citations
            ],
            "retrieved": [
                {
                    "label": i.label,
                    "source": i.citation,
                    "score": round(i.score, 4),
                    "image_path": i.image_path,
                }
                for i in result.context.items
            ],
            "usage": {
                "prompt_tokens": gen.usage.prompt_tokens,
                "completion_tokens": gen.usage.completion_tokens,
                "total_tokens": gen.usage.total_tokens,
            },
            "cost_usd": gen.cost_usd,
            "latency_ms": {
                "retrieval": round(gen.retrieval_latency_ms, 1),
                "llm": round(gen.llm_latency_ms, 1),
                "generation_total": round(gen.total_latency_ms, 1),
                "end_to_end": round(gen.end_to_end_latency_ms, 1),
            },
            "images_sent": gen.images_sent,
            "attempts": gen.attempts,
        }
    return out


async def run(args: argparse.Namespace, settings: Settings) -> int:
    reports: list[IngestReport] = []
    result: QueryResult | None = None
    async with build_pipeline(settings) as rag:
        if args.ingest:
            reports = await rag.ingest(args.ingest, reset=args.reset)
            if not args.json:
                print_ingest(reports)
        if args.query:
            result = await rag.query(args.query)
            if not args.json:
                print_answer(result)
    if args.json:
        print(json.dumps(result_json(reports, result), indent=2))
    return 1 if any(not r.ok for r in reports) else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        overrides: dict[str, Any] = {}
        settings = load_settings(args.config)
        if args.model or args.top_k:
            llm = settings.llm.model_copy(update={"model": args.model} if args.model else {})
            retrieval = settings.retrieval.model_copy(update={"top_k": args.top_k} if args.top_k else {})
            overrides = {"llm": llm, "retrieval": retrieval}
            settings = settings.model_copy(update=overrides)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    level = "DEBUG" if args.verbose else "WARNING" if args.quiet or args.json else settings.log_level
    setup_logging(level)

    key_error = check_api_keys(settings, bool(args.ingest), bool(args.query))
    if key_error:
        print(key_error, file=sys.stderr)
        return 2

    try:
        return asyncio.run(run(args, settings))
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except GenerationError as exc:
        print(f"The model call failed. Check your API key, model name and network.\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
