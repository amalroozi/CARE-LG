"""
Benchmark Metrics Engine for CARE-LG research framework.
Evaluates Constraint Violation Rate (CVR %), KDE log-density, clinical effort cost, and latency.
"""

import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from src.graph.clinical_constraints import check_clinical_violations, compute_clinical_effort, unscale_features


def compute_cvr(source_x, recourse_x, feature_metadata, scaler, feature_cols):
    """
    Computes Constraint Violation Rate (CVR) for a recourse recommendation:
    Returns 1.0 (100%) if any clinical violation occurs (age decrease or immutable feature change), else 0.0 (0%).

    Args:
        source_x (np.ndarray or torch.Tensor): Original query patient feature vector.
        recourse_x (np.ndarray or torch.Tensor): Recommended counterfactual feature vector.
        feature_metadata (dict): Feature schema configuration.
        scaler (StandardScaler): Fitted scaler object.
        feature_cols (list): List of feature column names.

    Returns:
        float: 1.0 if violation exists, 0.0 otherwise.
    """
    has_violation = check_clinical_violations(source_x, recourse_x, feature_metadata, scaler, feature_cols)
    return 1.0 if has_violation else 0.0


def compute_kde_density(recourse_points, background_data, bandwidth=0.5):
    """
    Fits a Kernel Density Estimator (KDE) on background training data and evaluates the mean log-density
    of generated counterfactual recourse points. Higher log-density indicates points lie in high-density data manifolds.

    Args:
        recourse_points (np.ndarray): Generated counterfactual samples of shape (M, D).
        background_data (np.ndarray): Training set feature samples of shape (N, D).
        bandwidth (float): KDE kernel bandwidth (default: 0.5).

    Returns:
        float: Mean log-likelihood density score.
    """
    recourse_points = np.atleast_2d(recourse_points)
    background_data = np.atleast_2d(background_data)

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(background_data)

    log_densities = kde.score_samples(recourse_points)
    return float(np.mean(log_densities))


def compute_cost_and_latency(recourse_paths, search_times, clinical_effort_weights, feature_metadata, scaler, feature_cols):
    """
    Computes average clinical effort cost and execution latency across query instances.

    Args:
        recourse_paths (list): List of (source_x, recourse_x) tuples.
        search_times (list): List of search latencies in seconds.
        clinical_effort_weights (dict): Weight mapping per mutable feature.
        feature_metadata (dict): Feature metadata configuration.
        scaler (StandardScaler): Fitted scaler object.
        feature_cols (list): Feature column names.

    Returns:
        mean_cost (float): Average clinical effort cost.
        mean_latency (float): Average search latency in seconds.
    """
    costs = []
    for src, rec in recourse_paths:
        cost = compute_clinical_effort(src, rec, clinical_effort_weights, feature_metadata, scaler, feature_cols)
        costs.append(cost)

    mean_cost = float(np.mean(costs)) if len(costs) > 0 else 0.0
    mean_latency = float(np.mean(search_times)) if len(search_times) > 0 else 0.0
    return mean_cost, mean_latency
