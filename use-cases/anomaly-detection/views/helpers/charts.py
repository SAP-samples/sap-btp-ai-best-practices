from __future__ import annotations

"""Chart helper functions for Streamlit views."""

import pandas as pd
import plotly.express as px


def build_top_materials_chart(anomalies: pd.DataFrame):
    data = anomalies.groupby("Material Description").size().reset_index(name="Count").nlargest(10, "Count")
    fig = px.bar(
        data,
        x="Count",
        y="Material Description",
        orientation="h",
        title="Top 10 Materials with Anomalies",
    )
    fig.update_layout(height=400)
    return fig


def build_anomaly_score_distribution(daily_data: pd.DataFrame):
    fig = px.histogram(
        daily_data,
        x="anomaly_score",
        color="predicted_anomaly",
        nbins=30,
        title="Anomaly Score Distribution",
        labels={"predicted_anomaly": "Is Anomaly", "anomaly_score": "Anomaly Score"},
    )
    fig.update_layout(height=400)
    return fig
