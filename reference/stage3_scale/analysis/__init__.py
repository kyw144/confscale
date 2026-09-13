"""P3 Analysis Pipeline — transform experiment outputs into paper-ready results.

Contract:
    report.py reads all run directories, produces:
        tables/  — 4 LaTeX + markdown tables
        figures/ — 8 publication-quality PDF figures
        statistics/ — hypothesis tests, effect sizes, diagnostics

Usage:
    python analysis/report.py --input outputs/ --output results/
"""
