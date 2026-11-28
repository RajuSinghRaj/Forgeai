"""Pure-Python PDF evaluation report generator for ForgeAI.

Builds a clean, multi-page, standards-compliant PDF 1.4 document using only
Python's standard library so deployment requires no extra binary or PDF
rendering dependencies.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from typing import Any

from forgeai.schemas.agent import DeclarativeAgentDefinition
from forgeai.schemas.evaluation import (
    EvalCaseResult,
    IterationScorecard,
    OptimizationIterationLog,
)

_PAGE_WIDTH = 612
_PAGE_HEIGHT = 792
_MARGIN_X = 48
_TOP_Y = 744
_BOTTOM_Y = 54
_CONTENT_WIDTH = _PAGE_WIDTH - 2 * _MARGIN_X


def _sanitize_pdf_text(text: str) -> str:
    """Normalize Unicode punctuation to clean WinAnsi/ASCII-safe text."""
    replacements = {
        "\u2014": "--",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2192": "->",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00a0": " ",
        "\r": " ",
        "\n": " ",
    }
    out = str(text or "")
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out.encode("latin-1", errors="replace").decode("latin-1")


def _escape_pdf_literal(text: str) -> str:
    """Escape characters special to PDF string literals."""
    safe = _sanitize_pdf_text(text)
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class _PdfCanvas:
    """Minimal multi-page PDF 1.4 writer with Type1 Helvetica fonts."""

    def __init__(self) -> None:
        self._pages: list[list[str]] = [[]]
        self.y: float = _TOP_Y

    @property
    def current(self) -> list[str]:
        return self._pages[-1]

    def new_page(self) -> None:
        self._pages.append([])
        self.y = _TOP_Y

    def ensure_space(self, needed_pt: float) -> None:
        if self.y - needed_pt < _BOTTOM_Y:
            self.new_page()

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill_rgb: tuple[float, float, float] | None = None,
        stroke_rgb: tuple[float, float, float] | None = None,
        line_width: float = 0.75,
    ) -> None:
        cmds: list[str] = ["q"]
        if fill_rgb is not None:
            cmds.append(f"{fill_rgb[0]:.3f} {fill_rgb[1]:.3f} {fill_rgb[2]:.3f} rg")
        if stroke_rgb is not None:
            cmds.append(f"{line_width:.2f} w")
            cmds.append(f"{stroke_rgb[0]:.3f} {stroke_rgb[1]:.3f} {stroke_rgb[2]:.3f} RG")
        op = "B" if (fill_rgb and stroke_rgb) else ("f" if fill_rgb else "S")
        cmds.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op}")
        cmds.append("Q")
        self.current.append("\n".join(cmds))

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        rgb: tuple[float, float, float] = (0.82, 0.85, 0.90),
        width: float = 0.75,
    ) -> None:
        self.current.append(
            f"q {width:.2f} w {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} RG "
            f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S Q"
        )

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        font: str = "F1",
        size: float = 10.0,
        rgb: tuple[float, float, float] = (0.08, 0.10, 0.16),
    ) -> None:
        escaped = _escape_pdf_literal(content)
        self.current.append(
            f"BT /{font} {size:.1f} Tf {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} rg "
            f"1 0 0 1 {x:.2f} {y:.2f} Tm ({escaped}) Tj ET"
        )

    def wrapped_paragraph(
        self,
        content: str,
        *,
        x: float = _MARGIN_X,
        width_chars: int = 92,
        font: str = "F1",
        size: float = 9.5,
        leading: float = 13.0,
        rgb: tuple[float, float, float] = (0.20, 0.24, 0.32),
        bullet: str = "",
    ) -> None:
        clean = _sanitize_pdf_text(content).strip()
        if not clean:
            return
        lines = textwrap.wrap(clean, width=width_chars, break_long_words=True, break_on_hyphens=False)
        for idx, line in enumerate(lines):
            self.ensure_space(leading + 2)
            prefix = bullet if idx == 0 and bullet else ("  " if bullet else "")
            self.text(x, self.y, f"{prefix}{line}", font=font, size=size, rgb=rgb)
            self.y -= leading

    def section_heading(self, title: str) -> None:
        self.ensure_space(34)
        self.y -= 8
        self.text(_MARGIN_X, self.y, title, font="F2", size=12.0, rgb=(0.24, 0.22, 0.72))
        self.y -= 5
        self.line(_MARGIN_X, self.y, _PAGE_WIDTH - _MARGIN_X, self.y, rgb=(0.82, 0.85, 0.94), width=0.9)
        self.y -= 13

    def to_bytes(self, footer_label: str) -> bytes:
        total_pages = len(self._pages)
        for idx, page_cmds in enumerate(self._pages, start=1):
            # Subtle footer on every page
            footer_cmds = [
                f"q 0.5 w 0.84 0.86 0.91 RG {_MARGIN_X:.2f} 38.00 m {_PAGE_WIDTH - _MARGIN_X:.2f} 38.00 l S Q",
                (
                    f"BT /F1 8.0 Tf 0.45 0.49 0.56 rg 1 0 0 1 {_MARGIN_X:.2f} 25.00 Tm "
                    f"({_escape_pdf_literal(footer_label)}) Tj ET"
                ),
                (
                    f"BT /F1 8.0 Tf 0.45 0.49 0.56 rg 1 0 0 1 {_PAGE_WIDTH - _MARGIN_X - 56:.2f} 25.00 Tm "
                    f"(Page {idx} of {total_pages}) Tj ET"
                ),
            ]
            page_cmds.extend(footer_cmds)

        objects: list[bytes] = []

        def _add_obj(body: bytes) -> int:
            objects.append(body)
            return len(objects)

        # 1: Catalog, 2: Pages (placeholders, replaced after pages built)
        _add_obj(b"")
        _add_obj(b"")
        # 3: Helvetica Regular, 4: Helvetica Bold
        font_reg_id = _add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        font_bold_id = _add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")

        page_ids: list[int] = []
        for page_cmds in self._pages:
            stream_data = "\n".join(page_cmds).encode("latin-1", errors="replace")
            content_obj = (
                f"<< /Length {len(stream_data)} >>\nstream\n".encode("ascii")
                + stream_data
                + b"\nendstream"
            )
            content_id = _add_obj(content_obj)
            page_obj = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_PAGE_WIDTH} {_PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_reg_id} 0 R /F2 {font_bold_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
            page_ids.append(_add_obj(page_obj))

        kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
        objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
        objects[1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_ids)} >>".encode("ascii")

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: list[int] = [0]
        for obj_num, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out.extend(f"{obj_num} 0 obj\n".encode("ascii"))
            out.extend(body)
            out.extend(b"\nendobj\n")

        xref_offset = len(out)
        out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        out.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
                "ascii"
            )
        )
        return bytes(out)


def _format_eval_date(value: str | datetime | None) -> str:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def build_evaluation_report_pdf(
    *,
    agent: DeclarativeAgentDefinition,
    scorecard: IterationScorecard,
    results: list[EvalCaseResult],
    eval_suite_sha256: str = "",
    evaluation_date: str | datetime | None = None,
    optimization_logs: list[OptimizationIterationLog] | None = None,
    stop_reason: str | None = None,
    honest_limitations: list[str] | None = None,
    run_id: str | None = None,
    gemini_model: str | None = None,
) -> bytes:
    """Generate a professionally formatted PDF evaluation report for the retained best scorecard."""
    canvas = _PdfCanvas()
    date_str = _format_eval_date(evaluation_date)
    suite_full = eval_suite_sha256.strip() or "n/a"
    logs = list(optimization_logs or [])
    failures = [r for r in results if not r.passed]
    computed_stop = (stop_reason or "").strip() or (
        "Baseline evaluation (no optimization iterations run)"
        if not logs
        else "Optimization completed"
    )

    limitations = list(honest_limitations or [])
    if not limitations:
        limitations.extend(agent.groundability_limitations)
        for fail in failures:
            limitations.append(
                f"Remaining failure {fail.case_id} [{fail.failure_classification.value}]: {fail.question}"
            )
        limitations = list(dict.fromkeys(limitations))

    # Top Banner
    canvas.rect(
        _MARGIN_X,
        704,
        _CONTENT_WIDTH,
        52,
        fill_rgb=(0.24, 0.22, 0.72),
        stroke_rgb=(0.24, 0.22, 0.72),
    )
    canvas.text(_MARGIN_X + 16, 734, "ForgeAI -- Evaluation Quality Report", font="F2", size=15.0, rgb=(1.0, 1.0, 1.0))
    canvas.text(
        _MARGIN_X + 16,
        716,
        "Retained Best Complete Scorecard | Evidence-Grounded Agent Assessment",
        font="F1",
        size=9.5,
        rgb=(0.86, 0.90, 0.99),
    )
    canvas.y = 686

    # Metadata Card
    meta_rows = [
        ("Agent Name & Version:", f"{agent.name} ({agent.agent_id}) v{agent.version}"),
        ("Evaluation Date:", date_str),
        ("Suite SHA-256:", suite_full),
        ("Iteration Evaluated:", f"Iteration {scorecard.iteration}"),
    ]
    if run_id:
        meta_rows.append(("Run ID:", run_id))
    if gemini_model:
        meta_rows.append(("Gemini Model:", gemini_model))

    box_h = 16 + len(meta_rows) * 15
    canvas.rect(
        _MARGIN_X,
        canvas.y - box_h + 10,
        _CONTENT_WIDTH,
        box_h,
        fill_rgb=(0.96, 0.97, 0.99),
        stroke_rgb=(0.84, 0.87, 0.94),
    )
    row_y = canvas.y - 4
    for label, val in meta_rows:
        canvas.text(_MARGIN_X + 12, row_y, label, font="F2", size=9.0, rgb=(0.25, 0.28, 0.36))
        canvas.text(_MARGIN_X + 140, row_y, val, font="F1", size=9.0, rgb=(0.10, 0.12, 0.18))
        row_y -= 15
    canvas.y = canvas.y - box_h - 6

    # Scorecard Metrics Summary (matches UI retained-best scorecard exactly)
    canvas.section_heading("Retained Best Complete Scorecard")
    kpi_cards = [
        ("Pass Rate", f"{scorecard.pass_rate:.0%}", f"{scorecard.passed_cases}/{scorecard.total_cases} passed"),
        ("Grounding Rate", f"{scorecard.grounding_rate:.0%}", f"Avg score {scorecard.avg_grounding_score:.2f}"),
        ("Citation Accuracy", f"{scorecard.citation_accuracy:.0%}", f"Avg correctness {scorecard.avg_correctness_score:.2f}"),
        ("Contract Compliance", f"{scorecard.contract_compliance_rate:.0%}", f"Failed cases: {scorecard.failed_cases}"),
    ]
    card_gap = 8
    card_w = (_CONTENT_WIDTH - 3 * card_gap) / 4
    card_h = 54
    canvas.ensure_space(card_h + 24)
    card_top_y = canvas.y
    for idx, (lbl, big_val, sub) in enumerate(kpi_cards):
        cx = _MARGIN_X + idx * (card_w + card_gap)
        canvas.rect(
            cx,
            card_top_y - card_h,
            card_w,
            card_h,
            fill_rgb=(0.97, 0.98, 1.00),
            stroke_rgb=(0.80, 0.84, 0.94),
        )
        canvas.text(cx + 10, card_top_y - 15, lbl, font="F2", size=8.0, rgb=(0.35, 0.38, 0.52))
        canvas.text(cx + 10, card_top_y - 33, big_val, font="F2", size=14.0, rgb=(0.10, 0.12, 0.20))
        canvas.text(cx + 10, card_top_y - 46, sub, font="F1", size=7.5, rgb=(0.40, 0.44, 0.54))
    canvas.y = card_top_y - card_h - 14

    # Case Breakdown
    canvas.wrapped_paragraph(
        f"Total Cases: {scorecard.total_cases} | Passed Cases: {scorecard.passed_cases} | "
        f"Failed Cases: {scorecard.failed_cases} | "
        f"Answerable Accuracy: {scorecard.answerable_accuracy:.0%} | "
        f"Out-of-Scope Refusal: {scorecard.out_of_scope_refusal_accuracy:.0%} | "
        f"Edge Case Accuracy: {scorecard.edge_case_accuracy:.0%}",
        size=9.0,
        leading=13.0,
    )

    # Optimization Summary
    canvas.section_heading("Optimization Summary")
    canvas.wrapped_paragraph(
        f"Optimization Iterations: {len(logs)} | Stop Reason: {computed_stop}",
        font="F2",
        size=9.5,
        leading=13.5,
    )
    for log in logs:
        status = "EARLY-STOPPED / PARTIAL (REJECTED)" if log.early_stopped else ("ACCEPTED" if log.accepted else "REJECTED")
        canvas.wrapped_paragraph(
            f"Iteration {log.iteration} [{status}]: {log.acceptance_reason}",
            bullet="- ",
            size=9.0,
            leading=12.5,
        )

    # Individual Failure Cases
    canvas.section_heading(f"Individual Failure Cases ({len(failures)})")
    if not failures:
        canvas.wrapped_paragraph(
            "All evaluated cases in the retained best complete scorecard passed.",
            size=9.5,
            rgb=(0.06, 0.50, 0.34),
        )
    else:
        for fail in failures:
            canvas.ensure_space(46)
            canvas.wrapped_paragraph(
                f"{fail.case_id} [{fail.failure_classification.value}] -- {fail.question}",
                font="F2",
                size=9.5,
                leading=13.0,
                rgb=(0.68, 0.14, 0.20),
            )
            explanation = fail.judge_reasoning or "No judge explanation recorded."
            canvas.wrapped_paragraph(
                f"Explanation: {explanation}",
                x=_MARGIN_X + 10,
                width_chars=88,
                size=8.8,
                leading=12.0,
            )
            if fail.deterministic_issues:
                canvas.wrapped_paragraph(
                    f"Deterministic Issues: {'; '.join(fail.deterministic_issues)}",
                    x=_MARGIN_X + 10,
                    width_chars=88,
                    size=8.5,
                    leading=11.5,
                    rgb=(0.42, 0.30, 0.12),
                )
            canvas.y -= 4

    # Honest Limitations & Remaining Failures
    canvas.section_heading("Honest Limitations & Remaining Failures")
    if not limitations:
        canvas.wrapped_paragraph("No unresolved corpus or evaluation limitations reported.", size=9.2)
    else:
        for lim in limitations:
            canvas.wrapped_paragraph(lim, bullet="- ", size=9.0, leading=12.5)

    return canvas.to_bytes(f"ForgeAI Evaluation Report | {agent.agent_id} v{agent.version}")


def build_evaluation_report_pdf_from_state(state: dict[str, Any]) -> bytes:
    """Build a PDF report strictly from the retained best complete scorecard in workflow state."""
    scorecards = list(state.get("scorecards") or [])
    best_sc: IterationScorecard | None = state.get("best_scorecard")
    if best_sc is None and scorecards:
        complete_scs = [sc for sc in scorecards if not getattr(sc, "is_partial", False)]
        best_sc = max(complete_scs, key=lambda sc: sc.pass_rate) if complete_scs else scorecards[-1]

    final_report = state.get("final_report")
    if best_sc is None and final_report is not None:
        best_sc = final_report.final_scorecard

    agent: DeclarativeAgentDefinition | None = state.get("best_agent") or state.get("agent_definition")
    if agent is None and final_report is not None:
        agent = final_report.final_agent

    if agent is None or best_sc is None:
        raise ValueError("Cannot build PDF evaluation report without a retained-best agent and scorecard.")

    results: list[EvalCaseResult] = list(state.get("best_results") or state.get("eval_results") or [])
    if not results and final_report is not None:
        results = list(final_report.final_results)

    opt_logs = list(state.get("optimization_logs") or [])
    if not opt_logs and final_report is not None:
        opt_logs = list(final_report.optimization_logs)

    suite_hash = str(state.get("eval_suite_sha256") or "")
    if not suite_hash and final_report is not None:
        suite_hash = final_report.eval_suite_sha256

    stop_reason = state.get("stop_reason")
    if not stop_reason and final_report is not None:
        stop_reason = final_report.stop_reason

    honest_limitations = (
        list(final_report.honest_limitations)
        if final_report is not None and final_report.honest_limitations
        else None
    )
    eval_date = final_report.created_at if final_report is not None else None
    run_id = state.get("run_id") or (final_report.run_id if final_report is not None else None)
    gemini_model = final_report.gemini_model if final_report is not None else None

    return build_evaluation_report_pdf(
        agent=agent,
        scorecard=best_sc,
        results=results,
        eval_suite_sha256=suite_hash,
        evaluation_date=eval_date,
        optimization_logs=opt_logs,
        stop_reason=stop_reason,
        honest_limitations=honest_limitations,
        run_id=run_id,
        gemini_model=gemini_model,
    )

