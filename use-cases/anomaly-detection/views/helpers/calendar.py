from __future__ import annotations

"""Calendar heatmap helper for monthly dashboard."""

import calendar
from datetime import date

import pandas as pd
import streamlit as st


def render_calendar_heatmap(year: int, month: int, daily_metrics_df: pd.DataFrame) -> None:
    cal = calendar.Calendar()
    month_days = cal.monthdatescalendar(year, month)

    header_html = "<tr>" + "".join(
        f"<th style='text-align:center;'>{day}</th>" for day in calendar.weekheader(2).split()
    ) + "</tr>"

    rows_html = []
    for week in month_days:
        row_html = "<tr>"
        for day_date in week:
            if day_date.month == month:
                metric_for_day = daily_metrics_df[daily_metrics_df["date"].dt.date == day_date]
                bg_color = "#f0f0f0"
                cell_text_color = "black"
                anomaly_info = ""

                if not metric_for_day.empty:
                    anomaly_rate = metric_for_day["anomaly_rate"].iloc[0]
                    anomaly_count = metric_for_day["anomaly_count"].iloc[0]

                    if anomaly_rate > 0:
                        if anomaly_rate > 0.2:
                            bg_color = "#ff7f7f"
                        elif anomaly_rate > 0.1:
                            bg_color = "#ffb2b2"
                        else:
                            bg_color = "#ffe5e5"
                        if anomaly_rate > 0.15:
                            cell_text_color = "white"
                    else:
                        bg_color = "#e6ffe6"

                    anomaly_info = f"<br><span style='font-size:0.7em;'>{anomaly_count} ({anomaly_rate:.0%})</span>"

                row_html += (
                    f"<td style='text-align:center; padding:5px; border:1px solid #ccc;"
                    f" background-color:{bg_color}; color:{cell_text_color}; height:60px; vertical-align:top;'>"
                    f"{day_date.day}{anomaly_info}</td>"
                )
            else:
                row_html += "<td style='background-color:#e8e8e8; border:1px solid #ccc;'></td>"
        row_html += "</tr>"
        rows_html.append(row_html)

    calendar_html = f"""
    <h5 style='text-align:center;'>Anomaly Heatmap for {calendar.month_name[month]} {year}</h5>
    <table style='width:100%; border-collapse: collapse;'>
        <thead>{header_html}</thead>
        <tbody>{''.join(rows_html)}</tbody>
    </table>
    <style>table td:hover {{ background-color: #f0f2f6 !important; }}</style>
    """

    st.markdown(calendar_html, unsafe_allow_html=True)
