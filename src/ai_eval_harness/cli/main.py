"""Small argparse CLI with explicit exit codes."""

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

from ai_eval_harness.datasets import load_dataset
from ai_eval_harness.errors import AiEvalHarnessError, ConfigError
from ai_eval_harness.evaluators.registry import EvaluatorBuildContext, iter_specifications
from ai_eval_harness.judges.base import JudgeProvider
from ai_eval_harness.judges.providers import build_provider
from ai_eval_harness.judges.scripted import load_scripted_provider
from ai_eval_harness.reporting import summary, write_report
from ai_eval_harness.runner.config import EvaluatorConfig, SuiteConfig, load_config
from ai_eval_harness.runner.suite import run_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline-first RAG and LLM evaluation suites")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-evaluators", help="List supported evaluator types")
    run = commands.add_parser("run", help="Evaluate a JSON/JSONL dataset")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--config", type=Path, help="TOML evaluators and quality gates")
    run.add_argument("--evaluator", action="append", help="Evaluator type (repeatable)")
    run.add_argument("--k", type=int, default=3, help="Retrieval cutoff (default: 3)")
    run.add_argument("--report", type=Path, help="Write structured JSON report")
    run.add_argument("--allow-empty-dataset", action="store_true")
    run.add_argument(
        "--allow-live",
        action="store_true",
        help="Allow paid API calls that send evaluation content to a provider",
    )
    args = parser.parse_args(argv)
    if args.command == "list-evaluators":
        for specification in iter_specifications():
            print(f"{specification.type}: {specification.description}")
        return 0
    if args.config and args.evaluator:
        parser.error("--config and --evaluator cannot be combined")
    try:
        config = (
            load_config(args.config)
            if args.config
            else SuiteConfig(
                evaluators=[
                    EvaluatorConfig(
                        type=name,
                        params={"k": args.k} if name in ("recall_at_k", "precision_at_k") else {},
                    )
                    for name in (
                        args.evaluator
                        or [
                            "recall_at_k",
                            "precision_at_k",
                            "reciprocal_rank",
                            "citation_f1",
                            "answer_token_f1",
                        ]
                    )
                ]
            )
        )
        if args.allow_empty_dataset:
            config = config.model_copy(update={"allow_empty_dataset": True})
        if config.judge is not None and not args.allow_live:
            raise ConfigError(
                "Live judge configuration requires --allow-live; API calls cost money "
                "and send evaluation content to an external provider"
            )
        dataset = load_dataset(args.dataset, allow_empty=config.allow_empty_dataset)
        with ExitStack() as stack:
            provider: JudgeProvider | None = None
            if config.judge is not None:
                provider = stack.enter_context(build_provider(config.judge))
            elif config.judge_responses:
                base = args.config.parent if args.config else Path.cwd()
                provider = load_scripted_provider(base / config.judge_responses)
            report = run_suite(dataset, config, context=EvaluatorBuildContext(provider))
        if args.report:
            write_report(report, args.report)
        print(summary(report))
        return {"passed": 0, "failed": 1, "error": 3, "insufficient_data": 1}[report.status]
    except (AiEvalHarnessError, ValueError, OSError) as exc:
        print(f"ai-eval: {exc}", file=sys.stderr)
        return 2


def run_cli() -> None:
    sys.exit(main())
