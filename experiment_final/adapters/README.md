# adapters/ — dataset adapters

One adapter per dataset; each returns the `DatasetBundle` described in ../DATASET_ADAPTER.md. A new dataset needs one adapter plus one line in ../config.yaml.

- `__init__.py` — package marker.
- `base.py` — DatasetBundle + contract validation shared by every adapter (DATASET_ADAPTER.md, PIPELINE P1).
- `cardio.py` — Kaggle Cardiovascular Disease adapter (uses `archive/experiments_v10/data_cardio.py`; the final report uses the 8,000-patient subsample).
- `eicu_demo.py` — optional eICU demo adapter (stub, not in `config.datasets`).
- `nhanes.py` — Adapter: nhanes -> DatasetBundle (see DATASET_ADAPTER.md). PIPELINE P1 + P2 inputs.
- `nhanes_pooled.py` — pooled NHANES 1999-2023 adapter: downloads (see NHANES_POOLED.md), harmonises per cycle, adds mortality fields.
- `nhanes_pooled_download.py` — Downloader for pooled NHANES 1999-2023 (NHANES_POOLED.md; PIPELINE P1).
- `uci.py` — Adapter: uci -> DatasetBundle (see DATASET_ADAPTER.md). PIPELINE P1.
