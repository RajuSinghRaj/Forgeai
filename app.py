"""Streamlit entry point for ForgeAI — The Agent That Builds Agents.

Phase 4 adds automatic evaluation-set generation (20+ cases), hybrid judging,
scorecard reporting, and evaluation artifact export while preserving Phase 1–3.
Importing or starting this app never triggers Gemini API calls.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import streamlit as st

from forgeai.config import load_settings
from forgeai.retrieval.loaders import load_document_folder
from forgeai.schemas import BriefInput, GroundabilityVerdict
from forgeai.workflow import build_factory_graph

st.set_page_config(
    page_title="ForgeAI — Agent Factory",
    page_icon="⚒️",
    layout="wide",
)


def _init_session_state() -> None:
    defaults = {
        "factory_state": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    _init_session_state()
    settings = load_settings()
    status = settings.status_summary()
    ready, issues = settings.gemini_readiness()

    st.title("ForgeAI — The Agent That Builds Agents")
    st.caption(
        "Phase 5: Intake, Groundability, Agent Generator, Eval & Judge, Critic–Optimizer & Final Report"
    )

    with st.sidebar:
        st.header("Environment Readiness")
        st.write(f"**Gemini Configured:** {'Yes' if ready else 'No'}")
        st.write(f"**Gemini Model:** `{status['gemini_model']}`")
        st.write(f"**Embedding Model:** `{status['embedding_model']}`")
        st.write(f"**Chroma Store:** `{status['chroma_persist_dir']}`")
        st.write(f"**Artifacts Dir:** `{status['artifacts_dir']}`")
        st.write(f"**CSV Rows/Unit:** `{settings.csv_rows_per_unit}`")
        st.write(f"**Min Eval Cases:** `{settings.min_eval_cases}`")
        st.write(f"**Target Pass Rate:** `{settings.target_pass_rate:.0%}`")
        st.write(f"**Max Opt Iterations:** `{settings.max_optimization_iterations}` (0–3)")
        if not ready:
            st.warning("Gemini API is not yet configured for live runs.")
            for issue in issues:
                st.caption(f"- {issue}")

    col_left, col_right = st.columns([1.15, 1.0])

    with col_left:
        st.subheader("1. Business Brief & Document Folder")
        brief_text = st.text_area(
            "Unseen Business Brief",
            placeholder="Describe the agent's purpose, audience, tone, must-answer topics, and must-refuse boundaries...",
            height=150,
        )
        docs_dir_input = st.text_input(
            "Document Folder Path",
            value="demos",
            help="Path containing PDF, CSV, TXT, or MD grounding documents.",
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Inspect Document Folder", type="secondary"):
                st.session_state["last_error"] = None
                target_dir = Path(docs_dir_input)
                if not target_dir.exists() or not target_dir.is_dir():
                    st.error(f"Folder not found: `{target_dir}`")
                else:
                    docs, units = load_document_folder(
                        target_dir, csv_rows_per_unit=settings.csv_rows_per_unit
                    )
                    if not docs:
                        st.info("No supported documents (.pdf, .csv, .txt, .md) found in that folder.")
                    else:
                        st.success(f"Found {len(docs)} document(s) across {len(units)} source unit(s).")
                        st.dataframe(
                            [
                                {
                                    "doc_id": d.doc_id,
                                    "file_name": d.file_name,
                                    "file_type": d.file_type,
                                    "units": d.unit_count,
                                    "bytes": d.size_bytes,
                                }
                                for d in docs
                            ],
                            use_container_width=True,
                        )

        with btn_col2:
            if st.button("Start Intake Interview", type="primary", disabled=not ready):
                st.session_state["last_error"] = None
                try:
                    brief = BriefInput(brief_text=brief_text, documents_dir=Path(docs_dir_input))
                    graph = build_factory_graph(settings=settings)
                    result_state = graph.invoke(
                        {
                            "brief_input": brief,
                            "csv_rows_per_unit": settings.csv_rows_per_unit,
                            "stop_after_phase2": True,
                        }
                    )
                    st.session_state["factory_state"] = result_state
                except Exception as exc:
                    st.session_state["last_error"] = str(exc)

        if st.session_state["last_error"]:
            st.error(st.session_state["last_error"])

        state = st.session_state.get("factory_state")
        if state and state.get("clarifying_questions"):
            st.subheader("2. Clarifying Questions (Max 5)")
            questions = state["clarifying_questions"]
            answers_map: dict[str, str] = {}
            with st.form("clarifying_answers_form"):
                for idx, q in enumerate(questions, start=1):
                    st.markdown(f"**Q{idx} (`{q.target_field}`):** {q.question}")
                    st.caption(f"Why asked: {q.rationale}")
                    answers_map[q.question_id] = st.text_input(
                        f"Answer for Q{idx} (leave blank to skip / partial answers allowed)",
                        key=f"ans_{q.question_id}",
                    )
                approve_gaps = st.checkbox(
                    "Allow proceeding if unsupported topics remain (requester override)",
                    value=False,
                )
                submitted = st.form_submit_button("Synthesize Requirements & Run Groundability Guard")

            if submitted:
                st.session_state["last_error"] = None
                try:
                    graph = build_factory_graph(settings=settings)
                    updated_state = graph.invoke(
                        {
                            **state,
                            "clarifying_answers": answers_map,
                            "interview_completed": True,
                            "approved_with_gaps": approve_gaps,
                            "stop_after_phase2": True,
                        }
                    )
                    st.session_state["factory_state"] = updated_state
                    st.rerun()
                except Exception as exc:
                    st.session_state["last_error"] = str(exc)

    with col_right:
        st.subheader("3. Requirements, Agent & Evaluation Output")
        state = st.session_state.get("factory_state")

        if not state:
            st.info(
                "Run the Intake Interview to generate requirements, groundability findings, "
                "a declarative agent, and automated evaluations."
            )
            return

        docs = state.get("documents") or []
        st.caption(f"Ingested documents: {len(docs)} | Units: {len(state.get('document_units') or [])}")

        req = state.get("requirements")
        if req is not None:
            st.markdown("#### Structured Requirements")
            st.write(f"**Domain:** {req.domain_name}")
            if req.purpose:
                st.write(f"**Purpose:** {req.purpose}")
            st.write(f"**Scope:** {req.scope}")
            st.write(f"**Audience:** {req.audience} | **Tone:** {req.tone}")
            if req.expected_response_behavior:
                st.write(f"**Response Behaviour:** {req.expected_response_behavior}")
            st.write("**Must-Answer Topics:**", req.must_answer_topics)
            st.write("**Must-Refuse Topics:**", req.must_refuse_topics)
            if req.constraints:
                st.write("**Constraints:**", req.constraints)
            if req.assumptions:
                st.write("**Assumptions:**", req.assumptions)

        report = state.get("groundability")
        if report is not None:
            st.markdown("#### Groundability Guard Report")
            verdict_label = report.overall_verdict.value
            if report.can_proceed and not report.ungrounded_topics:
                st.success(f"Overall Verdict: `{verdict_label}` — Approved for Agent Generation.")
            elif report.can_proceed:
                st.warning(
                    f"Overall Verdict: `{verdict_label}` — Approved with gaps: {report.ungrounded_topics}"
                )
            else:
                st.error(
                    f"Overall Verdict: `{verdict_label}` — Blocked before Agent Generation. "
                    f"Unsupported topics: {report.ungrounded_topics}"
                )

            for finding in report.findings:
                badge = {
                    GroundabilityVerdict.SUPPORTED: "SUPPORTED",
                    GroundabilityVerdict.PARTIALLY_SUPPORTED: "PARTIAL",
                    GroundabilityVerdict.UNSUPPORTED: "UNSUPPORTED",
                }[finding.verdict]
                scope_tag = " [OUT-OF-SCOPE]" if finding.is_out_of_scope else ""
                with st.expander(f"[{badge}]{scope_tag} {finding.topic}"):
                    st.write(f"**Evidence Summary:** {finding.evidence_summary}")
                    st.write(f"**Supporting Sources:** {finding.supporting_sources or ['None']}")
                    if finding.gap_description:
                        st.write(f"**Missing / Gap:** {finding.gap_description}")

            if report.recommendations:
                st.markdown("**Recommendations:**")
                for rec in report.recommendations:
                    st.markdown(f"- {rec}")

            st.markdown("#### Declarative Agent Generator (Phase 3)")
            can_generate = ready and report.can_proceed
            if st.button("Generate Agent", type="primary", disabled=not can_generate):
                st.session_state["last_error"] = None
                try:
                    graph = build_factory_graph(settings=settings)
                    agent_state = graph.invoke(
                        {
                            **state,
                            "stop_after_phase2": False,
                            "stop_after_phase3": True,
                        }
                    )
                    st.session_state["factory_state"] = agent_state
                    st.rerun()
                except Exception as exc:
                    st.session_state["last_error"] = str(exc)

        agent = state.get("agent_definition")
        if agent is not None:
            artifact_path = state.get("agent_artifact_path")
            st.success(f"Generated Agent: `{agent.name}` (`{agent.agent_id}`, v{agent.version})")
            if artifact_path:
                st.caption(f"Persisted to: `{artifact_path}`")

            with st.expander("System Prompt"):
                st.text(agent.system_prompt)
            with st.expander("Retrieval & Chunking Settings"):
                st.json({"chunking": agent.chunking.model_dump(), "retrieval": agent.retrieval.model_dump()})
            with st.expander("Selected Tools & Reasons"):
                for tool in agent.tools:
                    reason = tool.config.get("reason", tool.purpose)
                    st.markdown(f"- **`{tool.name}`**: {tool.purpose} *(Reason: {reason})*")
            with st.expander("Output Contract"):
                st.json(agent.output_contract.model_dump())

            agent_json = agent.model_dump_json(indent=2)
            st.download_button(
                "Export Agent JSON",
                data=agent_json,
                file_name=f"{agent.agent_id}_v{agent.version}.json",
                mime="application/json",
            )

            st.markdown("#### Evaluation Generator & Hybrid Judge (Phase 4)")
            eval_col1, eval_col2 = st.columns(2)
            with eval_col1:
                if st.button("Generate Evaluation Set", type="secondary", disabled=not ready):
                    st.session_state["last_error"] = None
                    try:
                        graph = build_factory_graph(settings=settings)
                        eval_state = graph.invoke(
                            {
                                **state,
                                "stop_after_phase2": False,
                                "stop_after_phase3": False,
                                "stop_after_eval_generation": True,
                                "stop_after_phase4": True,
                            }
                        )
                        st.session_state["factory_state"] = eval_state
                        st.rerun()
                    except Exception as exc:
                        st.session_state["last_error"] = str(exc)

            eval_cases = state.get("eval_cases") or []
            with eval_col2:
                if st.button(
                    "Run Evaluations & Judge",
                    type="primary",
                    disabled=not (ready and len(eval_cases) >= settings.min_eval_cases),
                ):
                    st.session_state["last_error"] = None
                    try:
                        graph = build_factory_graph(settings=settings)
                        judged_state = graph.invoke(
                            {
                                **state,
                                "stop_after_phase2": False,
                                "stop_after_phase3": False,
                                "stop_after_eval_generation": False,
                                "stop_after_phase4": True,
                            }
                        )
                        st.session_state["factory_state"] = judged_state
                        st.rerun()
                    except Exception as exc:
                        st.session_state["last_error"] = str(exc)

            if eval_cases:
                st.info(f"Generated Evaluation Cases: **{len(eval_cases)}**")
                grouped = defaultdict(list)
                for case in eval_cases:
                    grouped[case.category.value].append(case)
                for category_name, items in grouped.items():
                    with st.expander(f"Category `{category_name}` ({len(items)} cases)"):
                        for item in items:
                            st.markdown(f"**{item.case_id}:** {item.question}")
                            st.caption(
                                f"Behavior: `{item.expected_behavior.value}` | "
                                f"Sources: {item.source_documents or ['None']} | "
                                f"Expected Facts: {item.expected_facts or ['Refuse']}"
                            )
                cases_json = json.dumps([c.model_dump(mode="json") for c in eval_cases], indent=2)
                st.download_button(
                    "Export Evaluation Cases JSON",
                    data=cases_json,
                    file_name=f"{agent.agent_id}_eval_cases.json",
                    mime="application/json",
                )

            eval_results = state.get("eval_results") or []
            scorecards = state.get("scorecards") or []
            if scorecards:
                latest_sc = scorecards[-1]
                st.markdown("#### Evaluation Scorecard")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pass Rate", f"{latest_sc.pass_rate:.0%}")
                m2.metric("Passed / Total", f"{latest_sc.passed_cases}/{latest_sc.total_cases}")
                m3.metric("Grounding Rate", f"{latest_sc.grounding_rate:.0%}")
                m4.metric("Citation Accuracy", f"{latest_sc.citation_accuracy:.0%}")
                st.json(latest_sc.model_dump(mode="json"))

            if eval_results:
                failures = [r for r in eval_results if not r.passed]
                if failures:
                    st.error(f"Failed Cases: {len(failures)} / {len(eval_results)}")
                    for fail in failures:
                        with st.expander(
                            f"[FAIL: {fail.failure_classification.value}] {fail.case_id}: {fail.question}"
                        ):
                            st.write(f"**Agent Answer:** {fail.agent_answer}")
                            st.write(f"**Judge Reasoning:** {fail.judge_reasoning}")
                            if fail.deterministic_issues:
                                st.write(f"**Deterministic Issues:** {fail.deterministic_issues}")
                            if fail.disagreements:
                                st.write(f"**Disagreements:** {fail.disagreements}")
                else:
                    st.success("All evaluation cases passed!")

                with st.expander(f"All Individual Judge Results ({len(eval_results)})"):
                    for res in eval_results:
                        status_tag = "PASS" if res.passed else f"FAIL ({res.failure_classification.value})"
                        st.markdown(f"- **`{res.case_id}` [{status_tag}]** — {res.question}")
                        st.caption(
                            f"Grounding: {res.grounding_score:.2f} | Correctness: {res.correctness_score:.2f} | "
                            f"Cited: {res.cited_sources or ['None']}"
                        )

                results_json = json.dumps([r.model_dump(mode="json") for r in eval_results], indent=2)
                st.download_button(
                    "Export Evaluation Results JSON",
                    data=results_json,
                    file_name=f"{agent.agent_id}_eval_results.json",
                    mime="application/json",
                )

            st.markdown("#### Critic–Optimizer & Honest Final Report (Phase 5)")
            opt_col1, opt_col2 = st.columns([1, 1])
            with opt_col1:
                max_iters_ui = st.number_input(
                    "Optimization Budget (0–3 iterations)",
                    min_value=0,
                    max_value=3,
                    value=settings.max_optimization_iterations,
                    step=1,
                )
            with opt_col2:
                used_iters = len(state.get("optimization_logs") or [])
                st.metric("Remaining Budget", f"{max(0, int(max_iters_ui) - used_iters)} / {int(max_iters_ui)}")

            if st.button(
                "Optimize Agent",
                type="primary",
                disabled=not (ready and len(eval_cases) >= settings.min_eval_cases and bool(eval_results)),
            ):
                st.session_state["last_error"] = None
                try:
                    graph = build_factory_graph(settings=settings)
                    opt_state = graph.invoke(
                        {
                            **state,
                            "max_iterations": int(max_iters_ui),
                            "stop_after_phase2": False,
                            "stop_after_phase3": False,
                            "stop_after_eval_generation": False,
                            "stop_after_phase4": False,
                        }
                    )
                    st.session_state["factory_state"] = opt_state
                    st.rerun()
                except Exception as exc:
                    st.session_state["last_error"] = str(exc)

            opt_logs = state.get("optimization_logs") or []
            if opt_logs:
                st.markdown("##### Per-Iteration Critic–Optimizer History")
                for log in opt_logs:
                    badge = "ACCEPTED" if log.accepted else "REJECTED"
                    with st.expander(
                        f"Iteration {log.iteration} [{badge}] — Categories: {[c.value for c in log.observed_failure_categories]}"
                    ):
                        st.write(f"**Diagnosis:** {log.diagnosis_summary}")
                        st.write(f"**Failed Case IDs:** {log.failed_case_ids}")
                        st.write("**Proposed Changes:**", [c.model_dump() for c in log.proposed_changes])
                        st.write("**Applied Changes:**", [c.model_dump() for c in log.targeted_changes])
                        if log.rejected_changes:
                            st.write("**Rejected Changes:**", [r.model_dump() for r in log.rejected_changes])
                        if log.after_scorecard is not None:
                            st.write(
                                f"**Scorecard:** {log.before_scorecard.pass_rate:.0%} -> "
                                f"{log.after_scorecard.pass_rate:.0%} | **Deltas:** {log.metric_deltas}"
                            )
                        st.write(f"**Decision:** {log.acceptance_reason}")
                        st.write(f"**Remaining Failures:** {log.remaining_failed_case_ids}")

            final_report = state.get("final_report")
            if final_report is not None:
                st.markdown("##### Honest Final Report")
                if final_report.passing_criteria_met:
                    st.success(
                        f"Passing criteria met ({final_report.final_pass_rate:.0%} >= "
                        f"{final_report.target_pass_rate:.0%}). Stop reason: `{final_report.stop_reason}`"
                    )
                else:
                    st.warning(
                        f"Passing criteria NOT met ({final_report.final_pass_rate:.0%} < "
                        f"{final_report.target_pass_rate:.0%}) after {final_report.iterations_used} iteration(s). "
                        f"Stop reason: `{final_report.stop_reason}`"
                    )

                b_sc = final_report.baseline_scorecard
                f_sc = final_report.final_scorecard
                if b_sc and f_sc:
                    rm1, rm2, rm3, rm4 = st.columns(4)
                    rm1.metric(
                        "Final Pass Rate",
                        f"{f_sc.pass_rate:.0%}",
                        delta=f"{final_report.metric_deltas.get('pass_rate', 0.0):+.0%}",
                    )
                    rm2.metric(
                        "Grounding Rate",
                        f"{f_sc.grounding_rate:.0%}",
                        delta=f"{final_report.metric_deltas.get('grounding_rate', 0.0):+.0%}",
                    )
                    rm3.metric(
                        "Citation Accuracy",
                        f"{f_sc.citation_accuracy:.0%}",
                        delta=f"{final_report.metric_deltas.get('citation_accuracy', 0.0):+.0%}",
                    )
                    rm4.metric(
                        "Contract Compliance",
                        f"{f_sc.contract_compliance_rate:.0%}",
                        delta=f"{final_report.metric_deltas.get('contract_compliance_rate', 0.0):+.0%}",
                    )

                if final_report.honest_limitations:
                    with st.expander("Honest Limitations & Remaining Failures", expanded=True):
                        for item in final_report.honest_limitations:
                            st.markdown(f"- {item}")

                rep_col1, rep_col2 = st.columns(2)
                with rep_col1:
                    st.download_button(
                        "Export Final Report JSON",
                        data=final_report.model_dump_json(indent=2),
                        file_name=f"{final_report.run_id}_final_report.json",
                        mime="application/json",
                    )
                with rep_col2:
                    st.download_button(
                        "Export Final Agent JSON",
                        data=final_report.final_agent.model_dump_json(indent=2),
                        file_name=f"{final_report.final_agent.agent_id}_final_v{final_report.final_agent.version}.json",
                        mime="application/json",
                    )


if __name__ == "__main__":
    main()
