"""
Dataset configuration metadata and clinical effort weights for CARE-LG framework.
Supports multi-dataset configuration for UCI Heart Disease and NHANES Cardiovascular Cohort.
"""

DATASET_CONFIGS = {
    'uci': {
        'FEATURE_METADATA': {
            'continuous_mutable': ['resting_bp', 'cholesterol', 'max_heart_rate', 'oldpeak'],
            'categorical_mutable': ['exercise_angina', 'slope'],
            'non_decreasing': ['age'],
            'immutable': ['sex', 'fasting_blood_sugar'],
            'directional_reduce': ['resting_bp', 'cholesterol', 'oldpeak', 'exercise_angina', 'slope'],
            'directional_increase': ['max_heart_rate'],
            'target': 'heart_disease_risk'
        },
        # NOTE (archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT.md issue #9, resolved
        # experiments_v6/CHANGES.md "Phase 7"): these weights are illustrative,
        # hand-set constants with no cited derivation -- git history shows no
        # earlier rationale beyond the comment "representing the difficulty/
        # cost of changing each feature." They are NOT derived from a patient
        # survey, published treatment-burden score, or any other clinical
        # burden-of-change instrument. Every clinical-effort number reported
        # anywhere in this project (experiments_v2 through v6, including
        # Phase 6's preference-elicitation cost reweighting) inherits this
        # limitation.
        'CLINICAL_EFFORT_WEIGHTS': {
            'resting_bp': 1.5,
            'cholesterol': 2.0,
            'max_heart_rate': 1.2,
            'oldpeak': 2.5,
            'exercise_angina': 3.0,
            'slope': 2.2
        },
        'dataset_path': 'data/uci_heart.csv'
    },
    'nhanes': {
        'FEATURE_METADATA': {
            'continuous_mutable': ['systolic_bp', 'diastolic_bp', 'cholesterol', 'bmi', 'glycemic_hba1c'],
            'categorical_mutable': [],
            'non_decreasing': ['age'],
            'immutable': ['sex'],
            'directional_reduce': ['systolic_bp', 'diastolic_bp', 'cholesterol', 'bmi', 'glycemic_hba1c'],
            'directional_increase': [],
            'target': 'cvd_risk_flag'
        },
        # NOTE: same caveat as 'uci' above (archive/investigation_v2_to_v6/experiments_v6_audit/DEEP_AUDIT.md
        # issue #9) -- illustrative, uncited weights, not derived from clinical
        # burden-of-change data.
        'CLINICAL_EFFORT_WEIGHTS': {
            'systolic_bp': 1.5,
            'diastolic_bp': 1.5,
            'cholesterol': 2.0,
            'bmi': 2.5,
            'glycemic_hba1c': 3.0
        },
        'dataset_path': 'data/nhanes.csv'
    }
}


def get_dataset_config(dataset_name='uci'):
    """
    Returns (feature_metadata, effort_weights, dataset_path) for given dataset_name ('uci' or 'nhanes').
    """
    key = dataset_name.lower()
    if key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported datasets: {list(DATASET_CONFIGS.keys())}")
    cfg = DATASET_CONFIGS[key]
    return cfg['FEATURE_METADATA'], cfg['CLINICAL_EFFORT_WEIGHTS'], cfg['dataset_path']


# Default top-level exports for backward compatibility (defaults to UCI dataset)
FEATURE_METADATA = DATASET_CONFIGS['uci']['FEATURE_METADATA']
CLINICAL_EFFORT_WEIGHTS = DATASET_CONFIGS['uci']['CLINICAL_EFFORT_WEIGHTS']
