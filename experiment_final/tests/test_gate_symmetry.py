"""Test: gate_symmetry -- entry gate == interior edge predicate (CARE-LG B1); SRA keeps only source-admissible nodes (P6/P7)."""
import numpy as np
import scipy.sparse as sp
import torch

from experiment_final.src.graph import build_edge_table, query_edges
from experiment_final.src.knowledge_layer import hard_violates
from experiment_final.src.latent import encode, fit_vae
from experiment_final.src.prep import make_split
from experiment_final.src.tier1 import Tier1Arm, run_arm, target_mask
from src.graph.query_attachment import make_cell_matrix
from src.graph.source_relative_admissibility import restrict_to_source_admissible


def _setup(b, sch):
    sp_ = make_split(b, 0)
    vae = fit_vae(sp_, 0, epochs=3)
    Z = encode(vae, sp_.Xtr)
    et = build_edge_table(vae, Z, sp_.Xtr_raw, sch, 20)
    return sp_, vae, Z, et


def test_entry_gate_equals_interior_predicate(bundles, schemas):
    b, sch = bundles["uci"], schemas["uci"]
    sp_, vae, Z, et = _setup(b, sch)
    # interior: edge flags == predicate on recorded values
    assert (et.violates == hard_violates(sch, sp_.Xtr_raw[et.edge_i], sp_.Xtr_raw[et.edge_j], horizon=True)).all()
    # entry: a TRAIN node attached as a query gets exactly the interior flags towards its k-NN neighbours
    for i in range(10):
        qe = query_edges(vae, Z[i], sp_.Xtr_raw[i], Z, sp_.Xtr_raw, sch)
        m = et.edge_i == i
        assert (qe.violates[et.edge_j[m]] == et.violates[m]).all()


def test_sra_restricts_every_path_node_to_source_admissible(bundles, schemas):
    b, sch = bundles["uci"], schemas["uci"]
    sp_, vae, Z, et = _setup(b, sch)
    base = {"riemannian": make_cell_matrix(et, "riemannian", "hard"), "euclidean": make_cell_matrix(et, "euclidean", "hard")}
    risk = np.random.default_rng(0).uniform(0, 1, len(Z))
    ctx = {"X_raw": sp_.Xtr_raw, "N": len(Z), "schema": sch, "base_csr": base, "train_risk": risk}
    n_ok = 0
    for i in range(len(sp_.te)):
        x0 = sp_.Xte_raw[i]
        qe = query_edges(vae, encode(vae, sp_.Xte[i:i + 1])[0], x0, Z, sp_.Xtr_raw, sch)
        keep = ~qe.violates
        R = restrict_to_source_admissible(base["riemannian"], keep).tocoo()
        assert keep[R.row].all() and keep[R.col].all()
        r = run_arm(Tier1Arm("main"), ctx, qe, target_mask(risk, 0.5, 0.9), x0, 0.9)
        if r["tier1_success"]:
            n_ok += 1
            assert not r["real_violation_any_node"] and not r["real_violation_endpoint"]
            assert (~hard_violates(sch, x0, sp_.Xtr_raw[r["path"]], horizon=True)).all()
    assert n_ok > 0
