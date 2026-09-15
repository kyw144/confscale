from __future__ import annotations

import csv
import html
import json
import math
import statistics
from pathlib import Path


# Paths are assigned by the local CLI before use. Evidence is bundled; no network.
ROOT = Path(__file__).resolve().parents[1] / "evidence"
OUT = Path(__file__).resolve().parents[1] / "generated" / "supporting_entities"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fmt_num(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return f"{value:,.{digits}f}"


def fmt_pct(value, digits=1):
    return f"{value:.{digits}f}%"


def fmt_cov(mean, sd=None):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "-"
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {sd:.3f}"


def parse_markdown_table(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def table_1_calibration():
    src = ROOT / "data" / "p3_runs" / "results" / "real" / "tables" / "table_3_calibration.md"
    header, rows = parse_markdown_table(src)
    wanted = []
    for row in rows:
        method = row[0].replace("confscale-scp-online", "rolling-origin")
        wanted.append(
            [
                f"`{method}`",
                row[1].replace(chr(177), "+/-"),
                row[2],
                row[3],
                row[5],
            ]
        )
    text = [
        "# Table 1 - Pretrained UQ Calibration on A-D Baseline",
        "",
        "Empirical interval coverage against the 90% target. The two narrowest split-conformal heads are also the most under-covering heads, which is the calibration-as-frugality pitfall. The online split-conformal head (`confscale-scp-online` in the source artifact) is the rolling-origin recalibration variant; it is labeled `rolling-origin` here for consistency with the drift tables.",
        "",
        "| method | coverage | gap to 90% | median width (RPS) | spike coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    text += ["| " + " | ".join(row) + " |" for row in wanted]
    text += [
        "",
        f"Source: `{src.relative_to(ROOT).as_posix()}`.",
    ]
    write_text(TABLES / "table_1_uq_calibration.md", "\n".join(text))


def table_2_tuned_hpa():
    src = ROOT / "data" / "p3_runs" / "results" / "ev7_tuned_baseline_20260601_203020" / "ev7_cost_analysis.json"
    data = load_json(src)
    rows = []
    rows.append(
        {
            "config": "ConfScale-SCP",
            "cpu": "-",
            "downscale": "-",
            "overhead": data["scp"]["overhead_per_hour_mean"],
            "sd": data["scp"]["sd"],
            "mean_repl": data["scp"]["mean_replicas"],
            "p95": data["scp"]["e2e_p95_ms"],
            "viol": data["scp"]["e2e_violation"],
            "gate": "reference",
        }
    )
    for row in data["config_table"]:
        rows.append(
            {
                "config": row["config"],
                "cpu": row["cpu_target"],
                "downscale": f'{row["downscale_stab_s"]} s',
                "overhead": row["overhead_per_hour_mean"],
                "sd": row["overhead_per_hour_sd"],
                "mean_repl": row["mean_replicas"],
                "p95": row["e2e_p95_ms"],
                "viol": row["e2e_violation"],
                "gate": "yes" if row["meets_matched_slo"] else "no",
            }
        )
    order = {
        "hpa-tuned-u70-s60": 0,
        "ConfScale-SCP": 1,
        "hpa-tuned-u50-s60": 2,
        "hpa-tuned-u70-s120": 3,
        "hpa-tuned-u50-s120": 4,
        "hpa-anchor-u50-s300": 5,
    }
    rows.sort(key=lambda r: order[r["config"]])
    text = [
        "# Table 2 - Tuned HPA Pattern-D Cost Baseline",
        "",
        "Pattern D re-measured at matched end-to-end SLO. The old 71.9% saving disappears once HPA downscale stabilization is tuned.",
        "",
        "| config | CPU target | downscale | overhead rs/h | mean replicas | e2e p95 (ms) | e2e violation | SLO gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        text.append(
            "| "
            + " | ".join(
                [
                    f"`{row['config']}`" if row["config"] != "ConfScale-SCP" else "**ConfScale-SCP**",
                    str(row["cpu"]),
                    str(row["downscale"]),
                    f"{fmt_num(row['overhead'], 0)} +/- {fmt_num(row['sd'], 0)}",
                    fmt_num(row["mean_repl"], 2),
                    fmt_num(row["p95"], 0),
                    fmt_num(row["viol"], 3),
                    row["gate"],
                ]
            )
            + " |"
        )
    text += [
        "",
        "Strict matched-SLO comparison: ConfScale-SCP is +5.8% cheaper than `hpa-tuned-u50-s60`, within n=5 noise. Cheapest SLO-meeting comparison: ConfScale-SCP is 43.3% more expensive than `hpa-tuned-u70-s60`.",
        f"Source: `{src.relative_to(ROOT).as_posix()}`.",
    ]
    write_text(TABLES / "table_2_tuned_hpa_pattern_d.md", "\n".join(text))


def drift_summary_from_table_a():
    src = ROOT / "data" / "p3_runs" / "results" / "post_reframe" / "tables" / "table_a_coverage_cost_drift.md"
    header, rows = parse_markdown_table(src)
    data = {}
    for row in rows:
        method = row[0].strip("`")
        pattern = row[1]
        cov = row[2].replace(chr(9888) + chr(65039), "").replace(chr(177), "+/-").strip()
        if cov == "-" or not cov[:1].isdigit():
            cov = "-"
        cost = row[3].replace(chr(177), "+/-").strip()
        data[(method, pattern)] = {"cov": cov, "cost": cost}
    return data, src


def table_3_drift():
    data, src = drift_summary_from_table_a()
    methods = [
        "confscale-pid",
        "confscale-pid-laddered",
        "confscale-aci",
        "confscale-aci-laddered",
        "confscale-rolling-origin-laddered",
        "hpa-qr-monitored",
        "hpa-reactive",
    ]
    text = [
        "# Table 3 - Drift Coverage and Cost Geometry",
        "",
        "F is volatility drift, G is level drift, and H is regime change. Coverage below 0.85 is below the operational floor. Raw G/H correction notes are in Table 4.",
        "",
        "| method | F coverage | G coverage | H coverage | F rs/h | G rs/h | H rs/h |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = [f"`{method}`"]
        for pattern in ["F", "G", "H"]:
            row.append(data.get((method, pattern), {}).get("cov", "-"))
        for pattern in ["F", "G", "H"]:
            row.append(data.get((method, pattern), {}).get("cost", "-"))
        text.append("| " + " | ".join(row) + " |")
    text += [
        "",
        f"Source: `{src.relative_to(ROOT).as_posix()}`.",
    ]
    write_text(TABLES / "table_3_drift_coverage_cost.md", "\n".join(text))


def table_4_rescue_k8():
    ev3 = load_json(ROOT / "data" / "p3_runs" / "results" / "variance_recheck_20260530_175923" / "ev3_analysis.json")
    acih = load_json(ROOT / "data" / "p3_runs" / "results" / "ev7_tuned_baseline_20260601_203020" / "ev3_acih_analysis.json")
    rows = []
    for key, short in [("confscale-aci|G", "ACI on G"), ("confscale-pid|G", "PID on G")]:
        item = ev3["rederived"][key]
        p_holm = next(
            row["p_holm"]
            for row in ev3["holm"]["family"]
            if row["comparison"] == item["comparison"] and row["pattern"] == item["pattern"]
        )
        rows.append(
            [
                short,
                item["raw_K"],
                fmt_cov(item["raw_mean"], item["raw_sd"]),
                item["ladder_n"],
                fmt_cov(item["ladder_mean"], item["ladder_sd"]),
                f"{(item['ladder_mean'] - item['raw_mean']) * 100:.1f}",
                f"{p_holm:.2e}",
            ]
        )
    item = acih["rederived"]
    rows.append(
        [
            "ACI on H",
            item["raw_K"],
            fmt_cov(item["raw_mean"], item["raw_sd"]),
            item["ladder_n"],
            fmt_cov(item["ladder_mean"], item["ladder_sd"]),
            f"{(item['ladder_mean'] - item['raw_mean']) * 100:.1f}",
            f"{acih['holm_authoritative_family']['acih']['p_holm']:.2e}",
        ]
    )
    text = [
        "# Table 4 - K=8 Coverage-Rescue Cells",
        "",
        "The statistically defensible rescue family after the variance rechecks. These are the strongest ladder cells; other drift cells are directional or scoped.",
        "",
        "| comparison | raw K | raw coverage | ladder n | ladder coverage | delta (pp) | p_Holm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    text += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    text += [
        "",
        "Sources: `data/p3_runs/results/variance_recheck_20260530_175923/ev3_analysis.json`; `data/p3_runs/results/ev7_tuned_baseline_20260601_203020/ev3_acih_analysis.json`.",
    ]
    write_text(TABLES / "table_4_k8_ladder_rescue.md", "\n".join(text))


def table_5_perservice():
    summary = load_json(ROOT / "data" / "p3_runs" / "results" / "ev8b_pass2_full_20260601_210028" / "pass2_full_summary.json")
    recal = load_json(ROOT / "data" / "p3_runs" / "results" / "ev8b_pass2_full_20260601_210028" / "pass2_full_recal.json")
    anchor = load_json(ROOT / "data" / "p3_runs" / "results" / "ev8b_perservice_20260601_024537" / "recal_volatility.json")
    recovery = {}
    for item in recal["services"]:
        recovery[item["service"]] = {
            "frozen_cov": item["methods"]["frozen"]["cov_h0"][0],
            "frozen_sd": item["methods"]["frozen"]["cov_h0"][1],
            "aci_cov": item["methods"]["aci"]["cov_h0"][0],
            "aci_sd": item["methods"]["aci"]["cov_h0"][1],
            "pid_cov": item["methods"]["pid"]["cov_h0"][0],
            "pid_sd": item["methods"]["pid"]["cov_h0"][1],
            "aci_width_x": item["methods"]["aci"]["width_x_frozen"],
            "pid_width_x": item["methods"]["pid"]["width_x_frozen"],
        }
    recovery["MS_7129"] = {
        "frozen_cov": anchor["methods"]["frozen"]["cov_h0"][0],
        "frozen_sd": anchor["methods"]["frozen"]["cov_h0"][1],
        "aci_cov": anchor["methods"]["aci"]["cov_h0"][0],
        "aci_sd": anchor["methods"]["aci"]["cov_h0"][1],
        "pid_cov": anchor["methods"]["pid"]["cov_h0"][0],
        "pid_sd": anchor["methods"]["pid"]["cov_h0"][1],
        "aci_width_x": round(anchor["methods"]["aci"]["mean_width_h0"] / anchor["methods"]["frozen"]["mean_width_h0"], 1),
        "pid_width_x": round(anchor["methods"]["pid"]["mean_width_h0"] / anchor["methods"]["frozen"]["mean_width_h0"], 1),
    }
    order = ["MS_21558", "MS_7129", "MS_41763", "MS_7420"]
    # Gap = 90 - deployment-window coverage; the masking ratio uses a separate distribution.
    text = [
        "# Table 5 - Real Per-Service Volatility Coverage",
        "",
        "Frozen split conformal prediction under-covers every sanity-passing volatility cell; ACI and PID recover each testable service to approximately target coverage. Gap is the deploy-window shortfall below the 90% target (gap = 90 - frozen coverage). Coverage cells are means +/- s.d. over R=6 replicates; each cell validates n=59 deploy-window decisions per replicate.",
        "",
        "| service | frozen coverage | gap vs 90% (pp) | ACI coverage | PID coverage | ACI/PID width x frozen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    deploy_gaps = []
    for service in order:
        rec = recovery[service]
        gap = round(90.0 - rec["frozen_cov"], 1)
        deploy_gaps.append(gap)
        text.append(
            "| "
            + " | ".join(
                [
                    f"`{service}`",
                    f"{fmt_pct(rec['frozen_cov'], 1)} +/- {fmt_num(rec['frozen_sd'], 1)}",
                    fmt_num(gap, 1),
                    f"{fmt_pct(rec['aci_cov'], 1)} +/- {fmt_num(rec['aci_sd'], 1)}",
                    f"{fmt_pct(rec['pid_cov'], 1)} +/- {fmt_num(rec['pid_sd'], 1)}",
                    f"{fmt_num(rec['aci_width_x'], 1)} / {fmt_num(rec['pid_width_x'], 1)}",
                ]
            )
            + " |"
        )
    dist = summary["volatility"]["distribution"]
    deploy_median = round(statistics.median(deploy_gaps), 1)
    text += [
        "",
        f"Deploy-window gaps (90 - frozen coverage): median {fmt_num(deploy_median, 1)} pp; all four cells exceed 10 pp.",
        (
            "Aggregate-vs-per-service masking is a separate cut, the gap-distribution "
            f"analysis: aggregate +{summary['volatility']['vs_v2_aggregate_pp']} pp vs per-service "
            f"median +{dist['median']} pp (range +{dist['min']} to +{dist['max']} pp), a "
            f"{summary['volatility']['granularity_effect_x']}x granularity effect. That analysis uses "
            "different windows, so its per-service magnitudes differ slightly from the deploy-window gaps above."
        ),
        "Sources: `data/p3_runs/results/ev8b_pass2_full_20260601_210028/pass2_full_summary.json`; `pass2_full_recal.json`; `data/p3_runs/results/ev8b_perservice_20260601_024537/recal_volatility.json`.",
    ]
    write_text(TABLES / "table_5_perservice_volatility.md", "\n".join(text))


def table_6_h1_binds():
    src = ROOT / "data" / "p3_runs" / "results" / "ev8b_perservice_20260601_024537" / "h1_binds.json"
    data = load_json(src)
    text = [
        "# Table 6 - h0-Only Recalibration Reach",
        "",
        "Fraction of cycles where the frozen h1 upper bound wins the max-over-horizon decision, making h0 recalibration invisible to scaling on that cycle.",
        "",
        "| method | h1 binds (%) | h0 binds (%) | read band | up0_recal/up1_frozen |",
        "|---|---:|---:|---|---:|",
    ]
    for method in ["aci", "pid", "aci-lad", "pid-lad"]:
        item = data["methods"][method]
        text.append(
            "| "
            + " | ".join(
                [
                    f"`{method}`",
                    f"{fmt_num(item['h1_binds_pct']['mean'], 1)} +/- {fmt_num(item['h1_binds_pct']['sd'], 1)}",
                    fmt_num(item["h0_binds_pct_mean"], 1),
                    item["read_band"].split(":")[0],
                    fmt_num(item["decomposition_mean_rps"]["ci_upper0_recal"] / item["decomposition_mean_rps"]["ci_upper1_frozen"], 2),
                ]
            )
            + " |"
        )
    text += [
        "",
        f"Source: `{src.relative_to(ROOT).as_posix()}`.",
    ]
    write_text(TABLES / "table_6_h1_binds.md", "\n".join(text))


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933} .title{font-size:18px;font-weight:700} .label{font-size:12px} .tiny{font-size:10px} .axis{stroke:#65717e;stroke-width:1} .grid{stroke:#d8dee6;stroke-width:1} .note{fill:#5a6570;font-size:11px}</style>',
    ]


def text_el(x, y, text, cls="label", anchor="start"):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{html.escape(str(text))}</text>'


def rect_el(x, y, w, h, fill, stroke="none"):
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}"/>'


def circle_el(x, y, r, fill, stroke="#1f2933"):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'


def figure_cost_vs_coverage():
    data, _src = drift_summary_from_table_a()
    points = []
    for (method, pattern), vals in data.items():
        cov_s = vals["cov"]
        if cov_s == "-":
            continue
        try:
            cov = float(cov_s.split()[0])
            cost = float(vals["cost"].split()[0])
        except ValueError:
            continue
        points.append((method, pattern, cov, cost))
    colors = {
        "confscale-pid": "#1b6ca8",
        "confscale-pid-laddered": "#2a9d8f",
        "confscale-aci": "#7b2cbf",
        "confscale-aci-laddered": "#e76f51",
        "confscale-rolling-origin-laddered": "#f4a261",
        "hpa-qr-monitored": "#6c757d",
    }
    width, height = 980, 430
    svg = svg_header(width, height)
    # No baked-in figure number/title: the document caption carries those.
    svg.append(text_el(24, 20, "y-axis: overhead replica-seconds/hour, log scale; x-axis: empirical coverage", "note"))
    panels = {"F": 50, "G": 365, "H": 680}
    for pattern, x0 in panels.items():
        y0, pw, ph = 58, 260, 270
        svg.append(text_el(x0 + pw / 2, 48, f"Pattern {pattern}", "label", "middle"))
        svg.append(rect_el(x0, y0, pw, ph, "#ffffff", "#c9d1da"))
        for tick in [0.0, 0.5]:
            x = x0 + tick * pw
            svg.append(f'<line x1="{x:.1f}" y1="{y0+ph}" x2="{x:.1f}" y2="{y0+ph+4}" class="axis"/>')
            svg.append(text_el(x, y0 + ph + 16, f"{tick:g}", "tiny", "middle"))
        for ref, anchor, dx in [(0.85, "end", -3), (0.90, "start", 3)]:
            x = x0 + ref * pw
            svg.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+ph}" stroke="#9aa6b2" stroke-dasharray="4,4"/>')
            svg.append(text_el(x + dx, y0 + ph + 16, f"{ref:.2f}", "tiny", anchor))
        min_log, max_log = math.log10(20000), math.log10(65000)
        for tick in [20000, 30000, 45000, 60000]:
            y = y0 + ph - (math.log10(tick) - min_log) / (max_log - min_log) * ph
            svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" class="grid"/>')
            svg.append(text_el(x0 - 5, y + 4, f"{tick//1000}k", "tiny", "end"))
        for method, pat, cov, cost in points:
            if pat != pattern or method not in colors:
                continue
            x = x0 + cov * pw
            y = y0 + ph - (math.log10(cost) - min_log) / (max_log - min_log) * ph
            svg.append(circle_el(x, y, 5.5, colors[method]))
        svg.append(text_el(x0 + pw / 2, y0 + ph + 34, "coverage", "tiny", "middle"))
    legend_x, legend_y = 55, 388
    for i, (method, color) in enumerate(colors.items()):
        x = legend_x + (i % 3) * 295
        y = legend_y + (i // 3) * 22
        svg.append(circle_el(x, y - 4, 5, color))
        svg.append(text_el(x + 12, y, method, "tiny"))
    svg.append("</svg>")
    write_text(FIGURES / "figure_1_cost_vs_coverage.svg", "\n".join(svg))


def nice_scale(raw_max):
    """Smallest 1/2/5 x 10^k step that yields 3-6 axis ticks above raw_max."""
    for k in range(0, 7):
        for m in (1, 2, 5):
            step = m * 10 ** k
            n = math.ceil(raw_max / step)
            if 3 <= n <= 6:
                return step, n
    return raw_max / 4, 4


def bar_chart(path, labels, values, colors, ylabel="", target_lines=None, height=420, value_prefix=""):
    width = 900
    two_line_labels = any(len(str(l).split()) > 1 for l in labels)
    margin_l, margin_r, margin_t = 90, 30, 30
    margin_b = 95 if two_line_labels else 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    raw_max = max(values + ([v for v, _ in target_lines] if target_lines else [0])) * 1.12
    step, n_ticks = nice_scale(raw_max)
    max_val = step * n_ticks
    svg = svg_header(width, height)
    svg.append(f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}" class="axis"/>')
    svg.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" class="axis"/>')
    for i in range(n_ticks + 1):
        tick = step * i
        y = margin_t + plot_h - tick / max_val * plot_h
        svg.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l+plot_w}" y2="{y:.1f}" class="grid"/>')
        svg.append(text_el(margin_l - 8, y + 4, fmt_num(tick, 0), "tiny", "end"))
    bar_gap = 16
    bar_w = (plot_w - bar_gap * (len(values) + 1)) / len(values)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = margin_l + bar_gap + i * (bar_w + bar_gap)
        h = value / max_val * plot_h
        y = margin_t + plot_h - h
        svg.append(rect_el(x, y, bar_w, h, color))
        svg.append(text_el(x + bar_w / 2, y - 5, f"{value_prefix}{fmt_num(value, 1 if value < 100 else 0)}", "tiny", "middle"))
        parts = label.split()
        if len(parts) > 1:
            svg.append(text_el(x + bar_w / 2, margin_t + plot_h + 18, parts[0], "tiny", "middle"))
            svg.append(text_el(x + bar_w / 2, margin_t + plot_h + 32, " ".join(parts[1:]), "tiny", "middle"))
        else:
            svg.append(text_el(x + bar_w / 2, margin_t + plot_h + 22, label, "tiny", "middle"))
    if target_lines:
        # Drawn after the bars so the reference line and its label stay visible.
        for value, label in target_lines:
            y = margin_t + plot_h - value / max_val * plot_h
            svg.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l+plot_w}" y2="{y:.1f}" stroke="#b42318" stroke-dasharray="5,4"/>')
            svg.append(
                f'<text x="{margin_l + plot_w - 6}" y="{y - 6:.1f}" class="tiny" text-anchor="end" '
                f'style="fill:#b42318;paint-order:stroke;stroke:#ffffff;stroke-width:3px">{html.escape(str(label))}</text>'
            )
    if ylabel:
        cy = margin_t + plot_h / 2
        svg.append(
            f'<text x="22" y="{cy:.1f}" class="note" text-anchor="middle" '
            f'transform="rotate(-90 22 {cy:.1f})">{html.escape(str(ylabel))}</text>'
        )
    svg.append("</svg>")
    write_text(path, "\n".join(svg))


def figure_tuned_hpa():
    data = load_json(ROOT / "data" / "p3_runs" / "results" / "ev7_tuned_baseline_20260601_203020" / "ev7_cost_analysis.json")
    items = [
        ("HPA 70/60", data["config_table"][0]["overhead_per_hour_mean"]),
        ("ConfScale SCP", data["scp"]["overhead_per_hour_mean"]),
        ("HPA 50/60", data["config_table"][1]["overhead_per_hour_mean"]),
        ("HPA 50/300", data["anchor_reproduction"]["overhead_per_hour_mean"]),
    ]
    bar_chart(
        FIGURES / "figure_2_tuned_hpa_pattern_d.svg",
        [x[0] for x in items],
        [x[1] for x in items],
        ["#2a9d8f", "#1b6ca8", "#7b2cbf", "#e76f51"],
        ylabel="rs/h",
        height=420,
    )


def figure_perservice():
    summary = load_json(ROOT / "data" / "p3_runs" / "results" / "ev8b_pass2_full_20260601_210028" / "pass2_full_summary.json")
    services = summary["volatility"]["distribution"]["services"]
    labels = [s for s, _g in services]
    values = [g for _s, g in services]
    bar_chart(
        FIGURES / "figure_3_perservice_volatility_gap.svg",
        labels,
        values,
        ["#1b6ca8"] * len(labels),
        ylabel="coverage gap, pp",
        target_lines=[(summary["volatility"]["vs_v2_aggregate_pp"], "aggregate +3.6 pp")],
        height=430,
        value_prefix="+",
    )


def figure_ladder_rescue():
    ev3 = load_json(ROOT / "data" / "p3_runs" / "results" / "variance_recheck_20260530_175923" / "ev3_analysis.json")
    acih = load_json(ROOT / "data" / "p3_runs" / "results" / "ev7_tuned_baseline_20260601_203020" / "ev3_acih_analysis.json")
    labels = ["ACI G raw", "ACI G ladder", "PID G raw", "PID G ladder", "ACI H raw", "ACI H ladder"]
    values = [
        ev3["rederived"]["confscale-aci|G"]["raw_mean"],
        ev3["rederived"]["confscale-aci|G"]["ladder_mean"],
        ev3["rederived"]["confscale-pid|G"]["raw_mean"],
        ev3["rederived"]["confscale-pid|G"]["ladder_mean"],
        acih["rederived"]["raw_mean"],
        acih["rederived"]["ladder_mean"],
    ]
    bar_chart(
        FIGURES / "figure_4_k8_ladder_rescue.svg",
        labels,
        [v * 100 for v in values],
        ["#9aa6b2", "#2a9d8f", "#9aa6b2", "#1b6ca8", "#9aa6b2", "#e76f51"],
        ylabel="coverage %",
        target_lines=[(85, "0.85 floor"), (90, "0.90 target")],
        height=430,
    )


def figure_h1_binds():
    data = load_json(ROOT / "data" / "p3_runs" / "results" / "ev8b_perservice_20260601_024537" / "h1_binds.json")
    methods = ["aci", "pid", "aci-lad", "pid-lad"]
    labels = ["ACI", "PID", "ACI ladder", "PID ladder"]
    values = [data["methods"][m]["h1_binds_pct"]["mean"] for m in methods]
    bar_chart(
        FIGURES / "figure_5_h1_binds.svg",
        labels,
        values,
        ["#7b2cbf", "#1b6ca8", "#e76f51", "#2a9d8f"],
        ylabel="% cycles",
        target_lines=[(15, "material"), (40, "prominent")],
        height=420,
    )


def figure_control_loop():
    """MAPE-K dataflow diagram of one 30 s control cycle."""
    width, height = 980, 600
    svg = svg_header(width, height)
    svg.append(
        '<defs><marker id="arr" markerWidth="9" markerHeight="8" refX="8" refY="3.5" orient="auto">'
        '<path d="M0,0 L8,3.5 L0,7 z" fill="#1f2933"/></marker></defs>'
    )

    def arrow(points, label=None, lx=None, ly=None, dashed=False, anchor="middle"):
        d = "M" + " L".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        svg.append(f'<path d="{d}" fill="none" stroke="#1f2933" stroke-width="1.3"{dash} marker-end="url(#arr)"/>')
        if label:
            svg.append(text_el(lx, ly, label, "tiny", anchor))

    def block(x, y, w, h, lines, fill="#ffffff"):
        svg.append(rect_el(x, y, w, h, fill, "#65717e"))
        cy = y + 17
        for i, line in enumerate(lines):
            svg.append(text_el(x + w / 2, cy, line, "label" if i == 0 else "tiny", "middle"))
            cy += 15

    svg.append(text_el(24, 26, "Calibration-aware control cycle (MAPE-K)", "title"))

    svg.append(rect_el(30, 42, 920, 32, "#eef2f6", "#c9d1da"))
    svg.append(
        text_el(
            490,
            62,
            "Knowledge: pretrained GRU + UQ heads · capacity C, utilization ρ · coverage target T, miscoverage α* · ladder constants w, c",
            "tiny",
            "middle",
        )
    )

    svg.append(text_el(100, 132, "MONITOR", "label", "middle"))
    block(30, 140, 140, 90, ["Prometheus", "realized RPS y_t,", "end-to-end p95"])

    svg.append(rect_el(200, 110, 370, 310, "#f4f7fa", "#c9d1da"))
    svg.append(text_el(385, 130, "ANALYZE", "label", "middle"))
    block(215, 145, 155, 60, ["GRU predictor", "μ_t(τ), τ ∈ {h0, h1}"])
    block(400, 145, 155, 60, ["UQ head", "interval [L_t, U_t]"])
    block(215, 265, 155, 90, ["Coverage monitor", "pending FIFO, validate", "coverage C_t — eq. (4)"])
    block(400, 265, 155, 90, ["Recalibrator", "ACI / PID / rolling-origin", "α_t, q_t — eqs. (5)–(6)"])

    svg.append(rect_el(600, 110, 200, 310, "#f4f7fa", "#c9d1da"))
    svg.append(text_el(700, 130, "PLAN", "label", "middle"))
    block(615, 145, 170, 70, ["Escalation ladder", "L0 / L1 / L2 — eq. (7)", "reads C_t only"])
    block(615, 265, 170, 90, ["Replica planner", "p_target — eq. (8)", "60 s scale-up cooldown"])

    svg.append(text_el(890, 132, "EXECUTE", "label", "middle"))
    block(830, 140, 120, 220, ["Kubernetes API", "patch replicas"])

    block(
        30,
        480,
        920,
        80,
        [
            "kind cluster — four-service chain",
            "load generator → frontend → processor → compute worker (scaled tier) · Redis cache",
        ],
        fill="#eef2f6",
    )

    arrow([(170, 170), (215, 170)], "history", 192, 162)
    arrow([(370, 175), (400, 175)], "μ_t(τ)", 385, 167)
    arrow([(555, 175), (615, 175)], "[L_t, U_t]", 585, 167)
    arrow([(700, 215), (700, 265)], "widened interval (7)", 708, 243, anchor="start")
    arrow([(785, 310), (830, 310)], "p_target (8)", 807, 302)
    arrow([(890, 360), (890, 480)], "scale", 897, 424, anchor="start")
    arrow([(100, 480), (100, 230)], "served traffic, metrics", 106, 380, anchor="start")

    arrow([(170, 210), (190, 210), (190, 300), (215, 300)], "y_t validate", 188, 252, anchor="start")
    arrow([(420, 205), (340, 265)], "enqueue pending [L, U]", 380, 230, dashed=True)
    arrow([(370, 310), (400, 310)], "miss, residual", 385, 332)
    arrow([(478, 265), (478, 205)], "q_t (6)", 485, 238, anchor="start")
    arrow([(290, 355), (290, 445), (810, 445), (810, 180), (788, 180)], "trailing coverage C_t (4)", 550, 439)

    svg.append("</svg>")
    write_text(FIGURES / "figure_control_loop.svg", "\n".join(svg))


def copy_existing_pdf_note():
    src = ROOT / "data" / "p3_runs" / "results" / "post_reframe" / "figures" / "figure_a_cost_vs_coverage.pdf"
    # Keep the generated SVG as the reviewer-facing figure, but copy the original PDF for provenance if present.
    if src.exists():
        dest = FIGURES / "source_figure_a_cost_vs_coverage.pdf"
        dest.write_bytes(src.read_bytes())


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    table_1_calibration()
    table_2_tuned_hpa()
    table_3_drift()
    table_4_rescue_k8()
    table_5_perservice()
    table_6_h1_binds()
    figure_cost_vs_coverage()
    figure_tuned_hpa()
    figure_perservice()
    figure_ladder_rescue()
    figure_h1_binds()
    figure_control_loop()
    copy_existing_pdf_note()


if __name__ == "__main__":
    main()
