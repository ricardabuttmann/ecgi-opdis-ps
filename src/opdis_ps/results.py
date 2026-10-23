"""Create a reviewer-friendly CSV and HTML table and optionally open it."""

from __future__ import annotations

import csv
import html
import json
import math
import webbrowser
from pathlib import Path
from typing import Any

COLUMNS = (
    ("method", "Method"),
    ("relative_error_global", "RE ↓"),
    ("activation_time_error_ms", "ATE [ms] ↓"),
    ("localization_error_mm", "LE [mm] ↓"),
    ("spatial_cc", "sCC ↑"),
    ("temporal_cc", "tCC ↑"),
)

METHOD_GROUPS = (
    ("Variational regularization", ("tikhonov", "st-tikhonov", "tv")),
    ("Amortized inference", ("transformer", "palette")),
    ("Posterior sampling", ("dps", "ddrm", "opdis-ps")),
    ("OpDis-PS ablations", ("opdis-ps-marginal", "opdis-ps-marginal-isotropic")),
)
METHOD_LABELS = {
    "tikhonov": "Tikhonov (spatial)",
    "st-tikhonov": "Tikhonov (spatio-temporal)",
    "tv": "Total Variation",
    "transformer": "Transformer",
    "palette": "Palette",
    "dps": "DPS",
    "ddrm": "DDRM",
    "opdis-ps": "OpDis-PS",
    "opdis-ps-marginal": "OpDis-PS (marginal)",
    "opdis-ps-marginal-isotropic": "OpDis-PS (marginal, matched isotropic)",
}
METHOD_ORDER = {
    method: index
    for index, method in enumerate(method for _, methods in METHOD_GROUPS for method in methods)
}


def _display(value: Any) -> str:
    if isinstance(value, float):
        return "—" if not math.isfinite(value) else f"{value:.4f}"
    return str(value)


def collect_results(output_root: str | Path) -> list[dict[str, Any]]:
    output_root = Path(output_root)
    rows = []
    for path in output_root.glob("*/metrics.json"):
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise ValueError(f"{path} must contain a JSON object")
        rows.append(record)
    return sorted(rows, key=lambda row: METHOD_ORDER.get(str(row.get("method")), len(METHOD_ORDER)))


def _paper_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    method = str(result.get("method", ""))
    result["method"] = METHOD_LABELS.get(method, method)
    return result


def write_results_table(
    rows: list[dict[str, Any]], output_root: str | Path, *, open_browser: bool = True
) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("cannot create a results table without result rows")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "results.csv"
    html_path = output_root / "results.html"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[label for _, label in COLUMNS], extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(
            {label: _paper_row(row).get(key, "") for key, label in COLUMNS} for row in rows
        )

    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in COLUMNS)
    rows_by_method = {str(row.get("method")): row for row in rows}
    body_parts = []
    for group, methods in METHOD_GROUPS:
        group_rows = [rows_by_method[method] for method in methods if method in rows_by_method]
        if not group_rows:
            continue
        body_parts.append(
            f"<tr class='group'><th colspan='{len(COLUMNS)}'>{html.escape(group)}</th></tr>"
        )
        for raw_row in group_rows:
            row = _paper_row(raw_row)
            body_parts.append(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(_display(row.get(key, '')))}</td>"
                    for key, _ in COLUMNS
                )
                + "</tr>"
            )
    body = "".join(body_parts)
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>ECGI example results</title><style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;color:#18212f}"
        "table{border-collapse:collapse;width:100%;max-width:1100px}"
        "th,td{padding:.65rem .8rem;border-bottom:1px solid #d8dee8;text-align:right}"
        "th:first-child,td:first-child{text-align:left}"
        "th{background:#eef3f8;position:sticky;top:0}"
        "tr:hover{background:#f7f9fb}.group th{background:#dce7f2;text-align:left;position:static}"
        ".note{color:#536273}</style></head><body>"
        "<h1>ECGI example results</h1>"
        "<p class='note'>Generated automatically from the per-method metrics files.</p>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
        "</body></html>\n",
        encoding="utf-8",
    )
    if open_browser:
        webbrowser.open(html_path.resolve().as_uri())
    return csv_path, html_path


def refresh_results_table(output_root: str | Path, *, open_browser: bool = True) -> tuple[Path, Path]:
    return write_results_table(collect_results(output_root), output_root, open_browser=open_browser)
