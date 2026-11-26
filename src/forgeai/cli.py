"""Generic CLI entry point and artifact bundler for ForgeAI factory runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

from forgeai.config import ForgeSettings, load_settings
from forgeai.llm.gemini_client import GeminiClient
from forgeai.retrieval.embeddings import EmbeddingProvider
from forgeai.schemas.intake import BriefInput
from forgeai.workflow import FactoryState, build_factory_graph


def _write_json(path: Path, data: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump_json"):
        text = data.model_dump_json(indent=2)
    else:
        text = json.dumps(data, indent=2, default=str)
    path.write_text(text, encoding="utf-8")
    return str(path)


def persist_factory_run_bundle(
    out_dir: Path,
    *,
    brief_text: str,
    state: FactoryState,
    execution_metadata: dict[str, Any],
) -> dict[str, str]:
    """Persist a complete, standardized artifact bundle for a factory run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    brief_path = out_dir / "brief.txt"
    brief_path.write_text(brief_text.strip() + "\n", encoding="utf-8")
    written["brief"] = str(brief_path)

    if state.get("clarifying_questions") is not None:
        written["intake_questions"] = _write_json(
            out_dir / "intake_questions.json",
            [q.model_dump(mode="json") for q in state["clarifying_questions"]],
        )
    written["unattended_assumptions"] = _write_json(
        out_dir / "unattended_assumptions.json",
        state.get("unattended_assumptions") or [],
    )
    if state.get("requirements") is not None:
        written["requirements"] = _write_json(out_dir / "requirements.json", state["requirements"])
    if state.get("groundability") is not None:
        written["groundability"] = _write_json(out_dir / "groundability.json", state["groundability"])

    baseline_agent = state.get("baseline_agent") or state.get("agent_definition")
    if baseline_agent is not None:
        written["baseline_agent"] = _write_json(out_dir / "baseline_agent.json", baseline_agent)
        written["retrieval_and_chunking"] = _write_json(
            out_dir / "retrieval_and_chunking.json",
            {
                "chunking": baseline_agent.chunking.model_dump(mode="json"),
                "retrieval": baseline_agent.retrieval.model_dump(mode="json"),
            },
        )
        written["selected_tools"] = _write_json(
            out_dir / "selected_tools.json",
            [t.model_dump(mode="json") for t in baseline_agent.tools],
        )
        written["output_contract"] = _write_json(
            out_dir / "output_contract.json",
            baseline_agent.output_contract.model_dump(mode="json"),
        )

    if state.get("eval_cases") is not None:
        written["eval_cases"] = _write_json(
            out_dir / "eval_cases.json",
            [c.model_dump(mode="json") for c in state["eval_cases"]],
        )
    if state.get("baseline_results") is not None:
        written["baseline_results"] = _write_json(
            out_dir / "baseline_results.json",
            [r.model_dump(mode="json") for r in state["baseline_results"]],
        )
    if state.get("baseline_scorecard") is not None:
        written["baseline_scorecard"] = _write_json(
            out_dir / "baseline_scorecard.json",
            state["baseline_scorecard"],
        )

    logs = state.get("optimization_logs") or []
    written["optimization_history"] = _write_json(
        out_dir / "optimization_history.json",
        [log.model_dump(mode="json") for log in logs],
    )

    opt_dir_str = state.get("optimization_run_dir")
    if opt_dir_str:
        opt_dir = Path(opt_dir_str)
        for p in sorted(opt_dir.glob("iteration_*.json")):
            dest = out_dir / p.name
            dest.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            written[p.stem] = str(dest)

    final_agent = state.get("best_agent") or state.get("agent_definition")
    if final_agent is not None:
        written["final_agent"] = _write_json(out_dir / "final_agent.json", final_agent)
        written["selected_tools"] = _write_json(
            out_dir / "selected_tools.json",
            [t.model_dump(mode="json") for t in final_agent.tools],
        )
    if state.get("final_report") is not None:
        written["final_report"] = _write_json(out_dir / "final_report.json", state["final_report"])

    meta = {
        **execution_metadata,
        "run_id": state.get("run_id"),
        "stop_reason": state.get("stop_reason"),
        "eval_suite_sha256": state.get("eval_suite_sha256"),
        "document_count": len(state.get("documents") or []),
        "unit_count": len(state.get("document_units") or []),
        "artifact_files": sorted(Path(v).name for v in written.values()),
    }
    written["run_metadata"] = _write_json(out_dir / "run_metadata.json", meta)
    return written


def run_factory_cli(
    *,
    brief_text: str,
    docs_dir: Path,
    out_dir: Path,
    run_id: str,
    unattended_policy: Literal["stop_for_clarification", "proceed_with_assumptions"] = "proceed_with_assumptions",
    max_iterations: int = 3,
    min_eval_cases: int = 20,
    settings: ForgeSettings | None = None,
    gemini_client: GeminiClient | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> FactoryState:
    """Execute the full LangGraph factory pipeline and persist the output bundle."""
    cfg = settings or load_settings()
    graph = build_factory_graph(
        settings=cfg,
        gemini_client=gemini_client,
        embedding_provider=embedding_provider,
    )
    brief = BriefInput(brief_text=brief_text, documents_dir=docs_dir, run_id=run_id)
    state = graph.invoke(
        {
            "run_id": run_id,
            "brief_input": brief,
            "unattended_policy": unattended_policy,
            "max_iterations": min(max(max_iterations, 0), 3),
            "min_eval_cases": min_eval_cases,
            "target_pass_rate": cfg.target_pass_rate,
        }
    )
    persist_factory_run_bundle(
        out_dir,
        brief_text=brief_text,
        state=state,
        execution_metadata=execution_metadata
        or {
            "execution_mode": "live_gemini",
            "live_gemini_verified": True,
            "gemini_model": cfg.gemini_model,
            "embedding_model": cfg.embedding_model,
            "unattended_policy": unattended_policy,
        },
    )
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ForgeAI Generic Agent Factory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the factory on a brief and document folder.")
    run_p.add_argument("--brief", required=True, help="Path to brief text file or literal brief text.")
    run_p.add_argument("--docs", required=True, type=Path, help="Path to source document folder.")
    run_p.add_argument("--out", required=True, type=Path, help="Output artifact bundle directory.")
    run_p.add_argument("--run-id", default="forgeai_cli_run", help="Identifier for the factory run.")
    run_p.add_argument(
        "--unattended-policy",
        choices=["stop_for_clarification", "proceed_with_assumptions"],
        default="proceed_with_assumptions",
    )
    run_p.add_argument("--max-iterations", type=int, default=3)
    run_p.add_argument("--min-eval-cases", type=int, default=20)

    args = parser.parse_args(argv)
    if args.command == "run":
        brief_arg = Path(args.brief)
        brief_text = brief_arg.read_text(encoding="utf-8") if brief_arg.exists() else args.brief
        try:
            state = run_factory_cli(
                brief_text=brief_text,
                docs_dir=args.docs,
                out_dir=args.out,
                run_id=args.run_id,
                unattended_policy=args.unattended_policy,
                max_iterations=args.max_iterations,
                min_eval_cases=args.min_eval_cases,
            )
        except Exception as exc:
            args.out.mkdir(parents=True, exist_ok=True)
            _write_json(
                args.out / "run_error.json",
                {"run_id": args.run_id, "error": str(exc), "error_type": type(exc).__name__},
            )
            print(f"ForgeAI run failed: {exc}", file=sys.stderr)
            return 1

        report = state.get("final_report")
        if report is not None:
            print(
                f"Run {args.run_id} finished: pass_rate={report.final_pass_rate:.0%}, "
                f"stop_reason={report.stop_reason}"
            )
        else:
            print(f"Run {args.run_id} paused (awaiting_user_answers={state.get('awaiting_user_answers')}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

