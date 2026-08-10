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
            'target': 'heart_disease_risk'
        },
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
            'target': 'cvd_risk_flag'
        },
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
