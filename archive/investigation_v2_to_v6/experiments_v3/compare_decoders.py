"""
Phase B diagnostic: direct before/after comparison of the v2 original VAE
decoder vs. the v3 type-aware decoder, same seed/data/epoch budget for both.
Writes experiments_v3/results/vae_reconstruction_error_v3.csv.

Usage: .venv/bin/python -m experiments_v3.compare_decoders
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

import experiments_v3.lib.dataset_registration  # noqa: F401
from experiments_v3.lib.seeding import set_all_seeds
from configs.dataset_config import get_dataset_config
from src.data.loader import get_dataloaders
from src.vae.model import train_vae as train_vae_original
from experiments_v3.lib.vae_v3 import train_vae_type_aware
from experiments_v3.lib.graph_build import unscale_matrix

RESULTS_DIR = REPO_ROOT / "experiments_v3" / "results"


def main():
    rows = []
    for ds in ["uci", "nhanes_real"]:
        set_all_seeds(0)
        fm, ew, _ = get_dataset_config(ds)
        train_loader, test_loader, scaler, feature_cols = get_dataloaders(dataset_name=ds, batch_size=64, seed=0)
        input_dim = len(feature_cols)
        all_x = [Xb.numpy() for Xb, _ in train_loader]
        X_train = np.vstack(all_x)

        set_all_seeds(0)
        vae_orig = train_vae_original(train_loader, input_dim=input_dim, latent_dim=4, epochs=25, device=torch.device("cpu"))
        with torch.no_grad():
            Z = vae_orig.encode(torch.tensor(X_train, dtype=torch.float32))[0]
            Xhat_orig = vae_orig.decode(Z).numpy()

        set_all_seeds(0)
        vae_new = train_vae_type_aware(train_loader, input_dim=input_dim, feature_cols=feature_cols,
                                        feature_metadata=fm, epochs=25, device=torch.device("cpu"))
        with torch.no_grad():
            Zn = vae_new.encode(torch.tensor(X_train, dtype=torch.float32))[0]
            Xhat_new = vae_new.decode(Zn).numpy()

        unscaled_true = unscale_matrix(X_train, scaler, feature_cols, fm)
        unscaled_orig = unscale_matrix(Xhat_orig, scaler, feature_cols, fm)
        unscaled_new = unscale_matrix(Xhat_new, scaler, feature_cols, fm)
        binary_cols = fm.get("immutable", []) + fm.get("categorical_mutable", [])

        for col in feature_cols:
            err_orig = np.abs(unscaled_orig[col] - unscaled_true[col])
            err_new = np.abs(unscaled_new[col] - unscaled_true[col])
            row = {"dataset": ds, "feature": col,
                   "mae_v2_original_decoder": round(float(err_orig.mean()), 4),
                   "mae_v3_typeaware_decoder": round(float(err_new.mean()), 4),
                   "is_binary_or_categorical": col in binary_cols}
            if col in binary_cols:
                uniq = sorted(np.unique(np.round(X_train[:, feature_cols.index(col)], 6)))

                def to_class(vals, uniq=uniq):
                    vals = np.asarray(vals).reshape(-1, 1)
                    u = np.array(uniq).reshape(1, -1)
                    return u[0, np.argmin(np.abs(vals - u), axis=1)]

                true_scaled = X_train[:, feature_cols.index(col)]
                row["decoded_accuracy_v2_original_decoder"] = round(
                    float((to_class(Xhat_orig[:, feature_cols.index(col)]) == to_class(true_scaled)).mean()), 4)
                row["decoded_accuracy_v3_typeaware_decoder"] = round(
                    float((to_class(Xhat_new[:, feature_cols.index(col)]) == to_class(true_scaled)).mean()), 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "vae_reconstruction_error_v3.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
