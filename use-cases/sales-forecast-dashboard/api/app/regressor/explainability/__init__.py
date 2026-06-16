"""
Explainability Module.

Provides SHAP-based model interpretation utilities:
- SHAP value computation and visualization
- Top contributor extraction
- Feature importance aggregation

Usage:
    from app.regressor.explainability import (
        compute_shap_values,
        plot_shap_summary,
        build_contributor_strings,
        get_top_features_by_shap,
    )

    # Compute SHAP values
    shap_values = compute_shap_values(model, X)

    # Get top features
    top_features = get_top_features_by_shap(shap_values, feature_names, top_n=10)

    # Generate plots
    plot_shap_summary(shap_values, X, "output/shap_summary.png")

    # Build contributor strings for each row
    contributors = build_contributor_strings(shap_values, feature_names, X, top_k=3)
"""

from .shap_analysis import (
    compute_shap_values,
    get_top_features_by_shap,
    plot_shap_summary,
    plot_shap_importance,
    plot_shap_dependence,
    plot_horizon_sensitivity,
    plot_cohort_importance,
    plot_fit_scatter,
)

from .contributors import (
    build_contributor_strings,
    get_top_contributors_dataframe,
    aggregate_feature_importance,
    filter_contributors_by_direction,
)


__all__ = [
    # SHAP analysis
    "compute_shap_values",
    "get_top_features_by_shap",
    "plot_shap_summary",
    "plot_shap_importance",
    "plot_shap_dependence",
    "plot_horizon_sensitivity",
    "plot_cohort_importance",
    "plot_fit_scatter",
    # Contributors
    "build_contributor_strings",
    "get_top_contributors_dataframe",
    "aggregate_feature_importance",
    "filter_contributors_by_direction",
]
