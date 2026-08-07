"""
Dataset configuration metadata and clinical effort weights for CARE-LG framework.
"""

FEATURE_METADATA = {
    'continuous_mutable': ['resting_bp', 'cholesterol', 'max_heart_rate', 'oldpeak'],
    'categorical_mutable': ['exercise_angina', 'slope'],
    'non_decreasing': ['age'],
    'immutable': ['sex', 'fasting_blood_sugar'],
    'target': 'heart_disease_risk'
}

# Clinical effort weights representing the difficulty/cost of changing each feature
CLINICAL_EFFORT_WEIGHTS = {
    # Continuous mutable feature weights
    'resting_bp': 1.5,
    'cholesterol': 2.0,
    'max_heart_rate': 1.2,
    'oldpeak': 2.5,
    # Categorical mutable feature weights
    'exercise_angina': 3.0,
    'slope': 2.2
}
