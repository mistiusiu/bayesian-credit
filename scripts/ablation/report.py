#!/usr/bin/env python3
"""PDF report of an ablation study run (scripts/ablation/study.py) - Times New Roman 12 pt, 1.5 spacing, A4.

    pip install reportlab          # the only dependency (not part of the project environment)
    python3 scripts/ablation/report.py data/runs/ablation-pilot [--author "..."] [--supervisor "..."]

Every number and comparative statement is computed from ``<run>/results.json``, so the report is
correct for any scale or reproduction. Fonts: Times New Roman when fontconfig finds it (bold and
italic faces fall back to the metric-compatible Liberation Serif), else Liberation Serif, else
reportlab's built-in Times.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        KeepTogether,
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError:  # pragma: no cover - optional dependency
    sys.exit("report.py needs reportlab: pip install reportlab")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study import POLICIES, SCALES, SUBSETS, TRAIN_END

LABEL = {s["key"]: s["label"] for s in SUBSETS}
PLABEL = {"thompson": "Thompson Sampling", "thompson_se": "Thompson (σ/√n)", "ucb1": "UCB1",
          "epsilon_greedy": "ε-greedy", "greedy": "Greedy", "historical": "Historical (LC)", "random": "Random"}
DIM = {"grade": "grade", "fico_band": "FICO band", "purpose_group": "purpose", "home_ownership": "home",
       "income_band": "income band", "dti_band": "DTI band", "emp_band": "employment band"}
ASCII = {"σ": "sigma", "√": "sqrt ", "ε": "epsilon", "×": "x", "≈": "~", "→": "->", "÷": "/"}


def font_file(pattern: str, want: str) -> str | None:
    if not shutil.which("fc-match"):
        return None
    out = subprocess.run(["fc-match", "-f", "%{file}|%{family}", pattern], capture_output=True, text=True, check=False).stdout
    path, _, family = out.partition("|")
    return path if want.lower() in family.lower() and path.lower().endswith((".ttf", ".otf")) else None


def register_fonts() -> tuple[str, bool]:
    """Register the family 'Body'; returns (family, unicode_ok)."""
    regular = font_file("Times New Roman", "Times New Roman") or font_file("Liberation Serif", "Liberation Serif")
    if not regular:
        return "Times-Roman", False
    faces = {"normal": regular}
    for face, style in (("bold", "Bold"), ("italic", "Italic"), ("boldItalic", "Bold Italic")):
        f = font_file(f"Times New Roman:style={style}", "Times New Roman")
        if not f or f == regular:  # fontconfig answers a missing face with the regular file
            f = font_file(f"Liberation Serif:style={style}", "Liberation Serif")
        faces[face] = f or regular
    names = {}
    for face, path in faces.items():
        names[face] = f"Body-{face}"
        pdfmetrics.registerFont(TTFont(names[face], path))
    pdfmetrics.registerFontFamily("Body", normal=names["normal"], bold=names["bold"], italic=names["italic"],
                                  boldItalic=names["boldItalic"])
    return names["normal"], True


def build(run: Path, author: str, supervisor: str | None, out_pdf: Path) -> Path:
    r = json.loads((run / "results.json").read_text())
    scale = r["scale"]
    body_font, unicode_ok = register_fonts()
    bold = "Body-bold" if unicode_ok else "Times-Bold"
    italic = "Body-italic" if unicode_ok else "Times-Italic"

    def tx(s: str) -> str:
        if unicode_ok:
            return s
        for k, v in ASCII.items():
            s = s.replace(k, v)
        return s

    body = ParagraphStyle("body", fontName=body_font, fontSize=12, leading=18, alignment=TA_JUSTIFY, spaceAfter=8)
    title = ParagraphStyle("title", parent=body, fontName=bold, fontSize=16, leading=24, alignment=TA_CENTER)
    sub = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, spaceAfter=2)
    h1 = ParagraphStyle("h1", parent=body, fontName=bold, fontSize=13, leading=20, spaceBefore=12, spaceAfter=6,
                        alignment=0, keepWithNext=1)
    h2 = ParagraphStyle("h2", parent=h1, fontSize=12, leading=18, spaceBefore=8, spaceAfter=4)
    cell = ParagraphStyle("cell", fontName=body_font, fontSize=10, leading=15)
    cellb = ParagraphStyle("cellb", parent=cell, fontName=bold)
    caption = ParagraphStyle("caption", parent=body, fontName=italic, fontSize=10, leading=15, spaceBefore=4,
                             spaceAfter=10)

    def P(t, st=body):
        return Paragraph(tx(t), st)

    def bullets(items):
        return ListFlowable([ListItem(P(t), leftIndent=14, value="•", bulletFontName=body_font) for t in items],
                            bulletType="bullet", start="•", leftIndent=14, bulletFontName=body_font)

    def table(rows, widths, cap, highlight=()):
        data = [[P(str(v), cellb if i == 0 else cell) for v in row] for i, row in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1)
        style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9a9a9a")),
                 ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e4e4")),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
        style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f3f3")) for i in highlight]
        t.setStyle(TableStyle(style))
        return [t, P(cap, caption)]

    keys = [s["key"] for s in reversed(SUBSETS) if s["key"] in r["fits"]]
    C = {(x["subset"], x["mode"], x["policy"]): x for x in r["policies"]}
    runs, fits = r["runs"], r["fits"]
    pct = lambda v: f"{v:.1%}"
    mult = lambda v: f"{v:.2f}×"

    def share(k, mode, p="thompson"):
        return C[(k, mode, p)]["oracle_share"]

    def wc(metric, fmt, k, p):
        w, c = C[(k, "warm", p)][metric], C[(k, "cold", p)][metric]
        return fmt(w) if p == "historical" else f"{fmt(w)} / {fmt(c)}"

    cold = [share(k, "cold") for k in keys]
    warm = {k: share(k, "warm") for k in keys}
    best = {}
    for k in keys:
        for mode in ("warm", "cold"):
            p = max(POLICIES, key=lambda p: C[(k, mode, p)]["cum_reward"])
            best[p] = best.get(p, 0) + 1
    top, top_n = max(best.items(), key=lambda kv: kv[1])
    mins = [v["seconds"] / 60 for v in runs.values() if v.get("seconds")]
    peaks = [v["peak_mb"] / 1024 for v in runs.values() if v.get("peak_mb")]
    thin = "thin_file" if "thin_file" in keys else keys[-1]
    sc = SCALES[scale]
    ds = next(iter(r["datasets"].values()))
    decisions = next(iter(C.values()))["n_decisions"]
    m = r.get("manifest", {})

    story = [
        P(f"{sc['name']} Ablation Study: Bayesian Contextual Bandits for Lending on Limited Borrower Data", title),
        P(f"DSA4900 Research Project · {author}", sub),
        P((f"Prepared for {supervisor} · " if supervisor else "") + date.today().strftime("%d %B %Y"), sub),
        Spacer(1, 10),
        P("1. Summary", h1),
        P("The framework (hierarchical Two-Stage Beta-Beta risk model, Markov credit-state chain, contextual "
          "Thompson Sampling, shadow lending engine and PSI Sentinel) is run on four nested variable subsets, "
          "removing the bureau score, Lending Club's grade and finally all credit-report data. Each subset is run "
          "<i>warm</i> (policies start from a risk model trained on six years of history) and fully <i>cold</i> "
          "(no offline model)."),
        bullets([
            (f"Cold Thompson Sampling captures {pct(min(cold))}–{pct(max(cold))} of the hindsight-optimal reward "
            f"across the subsets ({LABEL[thin]}: {pct(share(thin, 'cold'))})."),
            "Warm Thompson Sampling: " + "; ".join(f"{LABEL[k]} {pct(warm[k])}" for k in keys) + ".",
            (f"On the {LABEL[thin].lower()} Thompson Sampling earns {mult(C[(thin, 'warm', 'thompson')]['x_historical'])}"
            f" (warm) and {mult(C[(thin, 'cold', 'thompson')]['x_historical'])} (cold) the historical policy's "
            "expected reward."),
            f"{PLABEL[top]} has the highest expected reward in {top_n} of {2 * len(keys)} settings"
            + (f"; one of the two Thompson variants is highest in {best.get('thompson', 0) + best.get('thompson_se', 0)}"
               if top in ("thompson", "thompson_se") else "") + ".",
            (f"Each simulation ({decisions:,} decisions) took {min(mins):.0f}–{max(mins):.0f} minutes"
             + (f", peaking at {min(peaks):.1f}–{max(peaks):.1f} GB of worker memory." if peaks else ".")),
        ]),
        P("2. Design", h1),
        P(f"Lending Club 2007–2020Q3 export (loan statuses observed {ds.get('snapshot_date') or '2020-10-31'}); "
          f"each dataset holds {ds.get('n_loans', 0):,} loans issued June 2007 – September 2020, identical across "
          f"subsets. Warm models train on the loans issued before {TRAIN_END}; the bandit lab replays "
          + (f"{sc['sim_loans']:,} borrowers" if sc["sim_loans"] else "every borrower")
          + f" month by month from {TRAIN_END} to 2020-09 with all seven policies, hybrid feedback and the PSI "
          "Sentinel auto-acknowledging trips with a resegmentation capped at two added dimensions. Cold runs fit "
          "their own replay world on the same pre-2013 loans with the same segmentation."),
    ]
    rows = [["Subset", "Segmentation", "AUC", "Brier", "ECE"]]
    for k in keys:
        f = fits[k]
        rows.append([LABEL[k], " × ".join(DIM.get(d, d) for d in f["dims"]), f"{f['auc']:.3f}", f"{f['brier']:.4f}",
                     f"{f['ece']:.3f}"])
    story += [P("3. Results", h1), P("3.1 Risk model", h2), KeepTogether(table(
        rows, [1.6 * inch, 2.75 * inch, 0.6 * inch, 0.65 * inch, 0.65 * inch],
        "Table 1. Default prediction of the warm models on hold-out loans whose term elapsed by the snapshot."))]
    oracle = {k: (C[(k, "warm", "thompson")]["cum_reward"] + C[(k, "warm", "thompson")]["cum_regret"]) / 1e6
              for k in keys}
    story += [P("3.2 Lending-policy performance", h2),
              P("Each subset is scored by a replay world fitted under its own segmentation, so dollar totals are not "
                "comparable across subsets; the tables report the share of the same world's hindsight-oracle reward "
                "(reward ÷ (reward + regret)) and the multiple of the historical policy. Oracle reward per world: "
                + ", ".join(f"{LABEL[k]} ${oracle[k]:.1f}M" for k in keys) + ".")]
    widths = [1.5 * inch] + [4.75 * inch / len(keys)] * len(keys)
    rows = [["Policy", *[LABEL[k] for k in keys]]]
    rows += [[PLABEL[p], *[wc("oracle_share", pct, k, p) for k in keys]] for p in POLICIES]
    story.append(KeepTogether(table(rows, widths, "Table 2. Share of the hindsight-oracle reward, warm / cold.", (1,))))
    rows = [["Policy", *[LABEL[k] for k in keys]]]
    rows += [[PLABEL[p], *[wc("x_historical", mult, k, p) for k in keys]] for p in POLICIES if p != "historical"]
    story.append(KeepTogether(table(rows, widths, "Table 3. Multiple of the historical policy's expected reward, "
                                    "warm / cold.", (1,))))
    rows = [["Run", "Trips", "Months", "Minutes", "Peak GB"]]
    for k in keys:
        for mode in ("warm", "cold"):
            v = runs.get(f"{k}:{mode}", {})
            rows.append([f"{LABEL[k]} · {mode}", str(v.get("trips")), ", ".join(v.get("trip_epochs") or []),
                         f"{(v.get('seconds') or 0) / 60:.1f}", f"{(v.get('peak_mb') or 0) / 1024:.1f}"])
    story += [P("3.3 Monitoring and computation", h2), KeepTogether(table(
        rows, [1.75 * inch, 0.5 * inch, 2.55 * inch, 0.7 * inch, 0.75 * inch],
        "Table 4. PSI Sentinel breaker trips, run time and worker memory (includes page cache)."))]
    story += [P("4. Limitations", h1), bullets([
        "Rewards are expected values under each world's fitted model, not realised profits.",
        ("There is no borrower take-up or adverse-selection response beyond the within-group policy effect, so the "
        "uplift over the historical policy is likely overstated."),
        ("Loans issued in 2019–2020 are mostly unresolved at the October 2020 snapshot; their outcomes in hybrid "
        "feedback are largely simulated."),
        "One sample and one seed (42); compare reproductions with study.py compare.",
    ]), P("5. Reproduction", h1),
        P(f"Code commit {m.get('git_commit') or 'unknown'}; data {m.get('data', {}).get('file', 'Loan_status_2007-2020Q3.csv')}"
          f" (SHA-256 verified: {m.get('data', {}).get('verified', 'not checked')}); "
          f"<font face='Courier' size='9'>scripts/reproduce_ablation.sh &lt;docker|cli|k8s&gt; {scale}</font> "
          "(docs/REPRODUCE.md).", ParagraphStyle("left", parent=body, alignment=0))]

    def glue(flowables):  # headings never end a page alone
        out = []
        for f in flowables:
            if isinstance(f, KeepTogether):
                lead = []
                while out and isinstance(out[-1], Paragraph) and out[-1].style.name in ("h1", "h2"):
                    lead.insert(0, out.pop())
                f = KeepTogether([*lead, *f._content])
            out.append(f)
        return out

    def page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont(body_font, 10)
        canvas.drawCentredString(A4[0] / 2, 0.55 * inch, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=inch, rightMargin=inch, topMargin=inch,
                            bottomMargin=inch, title=f"{sc['name']} ablation study", author=author)
    doc.build(glue(story), onFirstPage=page_number, onLaterPages=page_number)
    return out_pdf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="study output directory (holds results.json)")
    ap.add_argument("--author", default="Misati Randy Nyambane · USIU-Africa")
    ap.add_argument("--supervisor", default="Prof. Gabriel Okello")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    r = json.loads((args.run / "results.json").read_text())
    out = args.out or args.run / f"{SCALES[r['scale']]['name']}_Ablation_Study_Results.pdf"
    print(build(args.run, args.author, args.supervisor or None, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
