"""Test: the final outputs (tables, run_full_nhanes, report_final.html, main_final.tex) are complete and consistent."""
import re
from pathlib import Path

import pandas as pd
import pytest

EXP = Path(__file__).resolve().parents[1]
RES, T5 = EXP / "results", EXP / "results" / "tables"
SEEDS = range(5)


def test_headline_guarantees():
    h = pd.read_csv(T5 / "headline_tiered.csv").set_index("dataset")
    assert int(h.loc["nhanes_pooled", "n_high_risk_total"]) == 15002
    assert (h.tier1_real_violations == 0).all() and (h.contraindicated_cards == 0).all()
    assert ((h.tier1_mean + h.tier2_mean + h.tier3_mean) - 100).abs().max() < 1e-6


@pytest.mark.parametrize("variant", ["nhanes_pooled", "nhanes_pooled__ontology_as_features"])
def test_full_rerun_is_complete(variant):
    for s in SEEDS:
        d = RES / "run_full_nhanes" / variant / f"seed{s}"
        assert (d / "DONE").exists(), d
        assert not (d / "partial_checkpoint.pkl").exists()


def test_rerun_patient_counts_match_headline():
    n = sum(len(pd.read_csv(RES / "run_full_nhanes" / "nhanes_pooled" / f"seed{s}" / "patients.csv")) for s in SEEDS)
    assert n == int(pd.read_csv(T5 / "headline_tiered.csv").set_index("dataset").loc["nhanes_pooled", "n_high_risk_total"])


def test_run_view_has_no_dangling_links():
    links = [p for p in (RES / "run_view").rglob("*") if p.is_symlink()]
    assert links and all(p.exists() for p in links)


def test_report_content():
    html = (RES / "report_final.html").read_text()
    assert "Patient #139" in html
    assert "Fairness of infeasibility" not in html
    assert "Kaggle CVD-8k (8,000-patient stratified subsample of the Kaggle Cardiovascular Disease dataset, Ulianova 2019)" in html
    assert "randomly sampled" not in html


def test_paper_figures_exist_and_numbers_current():
    tex = (EXP / "paper" / "main_final.tex").read_text()
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    assert figs and all((EXP / "paper" / f).exists() for f in figs)
    assert "15,002" in tex and "1,000 high-risk" not in tex
