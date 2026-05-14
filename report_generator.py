import os
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from predictive_insights import build_predictive_insights


REPORTS_FOLDER = "generated_reports"
os.makedirs(REPORTS_FOLDER, exist_ok=True)


def safe_count(df, column, value):
    if df.empty or column not in df.columns:
        return 0
    return int((df[column] == value).sum())


def build_metrics(df):
    if df.empty:
        return {
            "Unique Tickets": 0,
            "Operational Rows": 0,
            "High Priority": 0,
            "Medium Priority": 0,
            "Low Priority": 0,
            "Immediate": 0,
            "Open": 0,
            "In Progress": 0,
            "Closed": 0,
            "Pending Review": 0,
            "Pending Approval": 0,
            "Approved Tickets": 0,
            "Risk-Flagged Tickets": 0,
            "Average Response Time (hrs)": 0
        }

    numeric_response = (
        pd.to_numeric(df["response_time_hours"], errors="coerce")
        if "response_time_hours" in df.columns
        else pd.Series(dtype=float)
    )

    avg_response = round(numeric_response.mean(), 1) if not numeric_response.dropna().empty else 0

    risk_count = 0
    if "risk_flags" in df.columns:
        risk_count = df[
            df["risk_flags"].fillna("").str.strip().ne("")
            & ~df["risk_flags"].fillna("").str.contains("No major risk detected", case=False, na=False)
        ]["TicketID"].nunique() if "TicketID" in df.columns else 0

    return {
        "Unique Tickets": df["TicketID"].nunique() if "TicketID" in df.columns else len(df),
        "Operational Rows": len(df),
        "High Priority": safe_count(df, "priority", "High"),
        "Medium Priority": safe_count(df, "priority", "Medium"),
        "Low Priority": safe_count(df, "priority", "Low"),
        "Immediate": safe_count(df, "urgency", "Immediate"),
        "Open": safe_count(df, "Status", "Open"),
        "In Progress": safe_count(df, "Status", "In Progress"),
        "Closed": safe_count(df, "Status", "Closed"),
        "Pending Review": safe_count(df, "Status", "Pending Review"),
        "Pending Approval": safe_count(df, "Status", "Pending Approval"),
        "Approved Tickets": safe_count(df, "approval_status", "Approved"),
        "Risk-Flagged Tickets": int(risk_count),
        "Average Response Time (hrs)": avg_response
    }


def build_summary(df, report_type="executive", filter_value="All"):
    if df.empty:
        return "No ticket data is available for the selected report."

    metrics = build_metrics(df)

    total_rows = metrics["Operational Rows"]
    unique_tickets = metrics["Unique Tickets"]
    high_count = metrics["High Priority"]
    immediate_count = metrics["Immediate"]
    open_count = metrics["Open"]
    closed_count = metrics["Closed"]
    pending_review = metrics["Pending Review"]
    pending_approval = metrics["Pending Approval"]
    risk_flagged = metrics["Risk-Flagged Tickets"]
    avg_response = metrics["Average Response Time (hrs)"]

    summary_parts = [
        f"This report contains {unique_tickets} unique ticket(s) across {total_rows} operational row(s)."
    ]

    if report_type == "department":
        summary_parts.append(
            f"This department report focuses on {filter_value} and summarises workload, service pressure, risk visibility, approval status, and expected operational demand for that area."
        )
    elif report_type == "priority":
        summary_parts.append(
            f"This priority report focuses on {filter_value}-priority tickets and highlights service urgency, operational significance, governance considerations, and workflow status."
        )
    else:
        summary_parts.append(
            "This executive report provides a high-level business view of operational demand, urgency levels, service responsiveness, risk monitoring, approval workflow performance, and trend-based future outlook."
        )

    if total_rows > 0:
        high_pct = round((high_count / total_rows) * 100, 1)
        summary_parts.append(
            f"High-priority tickets account for {high_pct}% of the recorded workload."
        )

    if immediate_count > 0:
        summary_parts.append(
            f"There are {immediate_count} immediate ticket(s), indicating time-sensitive service requests that require closer monitoring."
        )

    if risk_flagged > 0:
        summary_parts.append(
            f"The system identified {risk_flagged} risk-flagged ticket(s), supporting the Week 6 compliance and risk monitoring layer."
        )
    else:
        summary_parts.append(
            "No major risk-flagged tickets were identified in this report selection."
        )

    if pending_review > 0:
        summary_parts.append(
            f"There are {pending_review} ticket(s) pending manual review, showing that uncertain classifications are being handled through a human review process."
        )

    if pending_approval > 0:
        summary_parts.append(
            f"There are {pending_approval} ticket(s) pending approval, demonstrating the Week 7 approval workflow before selected tickets continue through the process."
        )

    if avg_response > 24:
        summary_parts.append(
            f"The average response time is {avg_response} hours, which suggests service delays that may require process improvement."
        )
    else:
        summary_parts.append(
            f"The average response time is {avg_response} hours, which is within an acceptable demonstration range."
        )

    if open_count > closed_count:
        summary_parts.append(
            "Open tickets currently exceed closed tickets, suggesting a possible backlog."
        )
    else:
        summary_parts.append(
            "Closed tickets meet or exceed open tickets, indicating relatively stable resolution performance."
        )

    return " ".join(summary_parts)


def build_governance_summary(df):
    if df.empty:
        return "No governance summary is available because there is no ticket data."

    metrics = build_metrics(df)

    risk_flagged = metrics["Risk-Flagged Tickets"]
    pending_review = metrics["Pending Review"]
    pending_approval = metrics["Pending Approval"]
    approved = metrics["Approved Tickets"]

    parts = [
        "The governance layer supports responsible ticket handling by recording risk flags, transparency notes, manual review status, and approval workflow outcomes."
    ]

    if risk_flagged > 0:
        parts.append(
            f"{risk_flagged} ticket(s) contain risk flags, meaning they may involve sensitive HR matters, financial risk, low-confidence classification, or response-quality concerns."
        )
    else:
        parts.append(
            "No major risk flags were detected in this report selection."
        )

    if pending_review > 0:
        parts.append(
            f"{pending_review} ticket(s) remain in pending review, supporting human oversight for unclear or unclassified requests."
        )

    if pending_approval > 0:
        parts.append(
            f"{pending_approval} ticket(s) are waiting for approval before further processing."
        )

    if approved > 0:
        parts.append(
            f"{approved} ticket(s) have already passed through the approval workflow."
        )

    parts.append(
        "These controls demonstrate ethical awareness, transparency, and workflow accountability across the platform."
    )

    return " ".join(parts)


def build_recommendations(df, report_type="executive", filter_value="All"):
    recommendations = []

    if df.empty:
        return ["No recommendations available because there is no data in the selected report."]

    metrics = build_metrics(df)

    high_count = metrics["High Priority"]
    open_count = metrics["Open"]
    immediate_count = metrics["Immediate"]
    pending_review = metrics["Pending Review"]
    pending_approval = metrics["Pending Approval"]
    risk_flagged = metrics["Risk-Flagged Tickets"]
    avg_response = metrics["Average Response Time (hrs)"]

    if report_type == "department":
        recommendations.append(f"Continue monitoring workload concentration within the {filter_value} department.")

    if high_count >= 3:
        recommendations.append("Prioritise faster handling of high-priority tickets to reduce operational risk.")

    if immediate_count >= 2:
        recommendations.append("Review escalation procedures for immediate tickets to improve urgent response handling.")

    if risk_flagged > 0:
        recommendations.append("Review all risk-flagged tickets to ensure sensitive or compliance-related matters receive appropriate human oversight.")

    if pending_review > 0:
        recommendations.append("Clear pending review tickets regularly so unclassified requests do not remain unresolved.")

    if pending_approval > 0:
        recommendations.append("Monitor pending approval tickets to prevent workflow delays, especially for finance-related requests.")

    if avg_response > 24:
        recommendations.append("Investigate response-time delays and consider process improvements or better workload distribution.")

    if open_count >= 3:
        recommendations.append("Monitor unresolved open tickets closely to prevent backlog growth.")

    if not recommendations:
        recommendations.append("Current ticket activity appears stable; continue monitoring trends, risks, approvals, and response consistency.")

    return recommendations


def build_six_month_trend(df):
    if df.empty or "date" not in df.columns:
        return "No six-month trend could be generated because date data is unavailable."

    temp_df = df.copy()
    temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce")
    temp_df = temp_df.dropna(subset=["date"])

    if temp_df.empty:
        return "No six-month trend could be generated because valid date data is unavailable."

    latest_date = temp_df["date"].max()
    six_month_start = latest_date - pd.DateOffset(months=6)
    six_month_df = temp_df[temp_df["date"] >= six_month_start].copy()

    if six_month_df.empty:
        return "No six-month trend could be generated because there is insufficient historical data."

    monthly_counts = (
        six_month_df.groupby(six_month_df["date"].dt.to_period("M"))["TicketID"]
        .nunique()
        .sort_index()
    )

    if monthly_counts.empty:
        return "No six-month trend could be generated because there is insufficient historical data."

    first_val = monthly_counts.iloc[0]
    last_val = monthly_counts.iloc[-1]

    if len(monthly_counts) == 1:
        trend_label = "stable"
    elif last_val > first_val:
        trend_label = "increasing"
    elif last_val < first_val:
        trend_label = "decreasing"
    else:
        trend_label = "stable"

    month_text = ", ".join([f"{str(period)}: {value}" for period, value in monthly_counts.items()])

    return (
        f"Six-month ticket trend is {trend_label}. Monthly ticket volumes were: {month_text}. "
        f"This trend supports longer-term workload planning and capacity decisions."
    )


def build_department_prediction_block(df, department_name):
    if df.empty:
        return [
            f"No department-specific predictions are available for {department_name} because there is no data."
        ]

    top_priority = (
        df["priority"].mode().iloc[0]
        if "priority" in df.columns and not df["priority"].dropna().empty
        else "N/A"
    )

    avg_response = 0
    if "response_time_hours" in df.columns:
        numeric_response = pd.to_numeric(df["response_time_hours"], errors="coerce")
        avg_response = round(numeric_response.mean(), 1) if not numeric_response.dropna().empty else 0

    insights = build_predictive_insights(df)

    return [
        f"{department_name} is projected to continue handling approximately {insights['overall']['prediction']} ticket(s) in the next immediate reporting period based on recent trend patterns.",
        f"The dominant priority level in {department_name} is currently {top_priority}, which helps indicate the expected intensity of departmental workload.",
        f"The department trend is {insights['department']['trend'].lower()}, while the average response time is {avg_response} hours."
    ]


def build_risk_table(df):
    if df.empty or "risk_flags" not in df.columns:
        return [["Risk Area", "Count"], ["No risk data available", "0"]]

    risk_df = df.copy()
    risk_df["risk_flags"] = risk_df["risk_flags"].fillna("").astype(str)

    flagged = risk_df[
        risk_df["risk_flags"].str.strip().ne("")
        & ~risk_df["risk_flags"].str.contains("No major risk detected", case=False, na=False)
    ]

    if flagged.empty:
        return [["Risk Area", "Count"], ["No major risk detected", "0"]]

    risk_counts = flagged["risk_flags"].value_counts().reset_index()
    risk_counts.columns = ["Risk Area", "Count"]

    return [["Risk Area", "Count"]] + risk_counts.astype(str).values.tolist()


def generate_pdf_report(df, report_type="executive", filter_value="All"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filter = str(filter_value).replace(" ", "_")
    filename = f"{report_type}_{safe_filter}_{timestamp}.pdf"
    file_path = os.path.join(REPORTS_FOLDER, filename)

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=22,
        leading=28,
        textColor=colors.black,
        spaceAfter=14
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        textColor=colors.grey,
        spaceAfter=14
    )

    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontSize=15,
        leading=20,
        textColor=colors.black,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=16
    )

    elements = []

    if report_type == "department":
        report_title = f"Department Report: {filter_value}"
    elif report_type == "priority":
        report_title = f"Priority Report: {filter_value}"
    else:
        report_title = "Executive Ticket Performance Report"

    elements.append(Paragraph(report_title, title_style))
    elements.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %B %Y at %H:%M')} | Filter: {filter_value}",
            subtitle_style
        )
    )

    metrics = build_metrics(df)
    summary = build_summary(df, report_type, filter_value)
    governance_summary = build_governance_summary(df)
    recommendations = build_recommendations(df, report_type, filter_value)
    predictive = build_predictive_insights(df)

    elements.append(Paragraph("Key Metrics", section_style))

    metric_data = [["Metric", "Value"]]
    for key, value in metrics.items():
        metric_data.append([key, str(value)])

    metric_table = Table(metric_data, colWidths=[250, 200])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ff69b4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9d9d9")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(metric_table)
    elements.append(Spacer(1, 18))

    elements.append(Paragraph("Automated Executive Summary", section_style))
    elements.append(Paragraph(summary, normal_style))
    elements.append(Spacer(1, 18))

    elements.append(Paragraph("Risk, Compliance & Approval Overview", section_style))
    elements.append(Paragraph(governance_summary, normal_style))
    elements.append(Spacer(1, 12))

    risk_table_data = build_risk_table(df)
    risk_table = Table(risk_table_data, colWidths=[330, 120])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4a6c1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9d9d9")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 18))

    if report_type == "executive":
        elements.append(Paragraph("Six-Month Trend Overview", section_style))
        elements.append(Paragraph(build_six_month_trend(df), normal_style))
        elements.append(Spacer(1, 18))

    if report_type == "department":
        elements.append(Paragraph("Department-Specific Forecast", section_style))
        for line in build_department_prediction_block(df, filter_value):
            elements.append(Paragraph(line, normal_style))
            elements.append(Spacer(1, 6))
        elements.append(Spacer(1, 12))

    elements.append(Paragraph("Recommendations", section_style))
    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", normal_style))
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 18))
    elements.append(Paragraph("Ticket Detail Snapshot", section_style))

    detail_columns = [
        "TicketID",
        "category",
        "priority",
        "urgency",
        "Status",
        "approval_status",
        "risk_flags",
        "date"
    ]

    available_columns = [col for col in detail_columns if col in df.columns]

    if available_columns:
        preview_df = df[available_columns].head(12).fillna("")
        detail_data = [available_columns] + preview_df.astype(str).values.tolist()
    else:
        detail_data = [["No ticket detail data available"]]

    detail_table = Table(detail_data, repeatRows=1)
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9d9d9")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(detail_table)
    elements.append(Spacer(1, 18))

    elements.append(Paragraph("Predictive Insights", section_style))
    elements.append(Paragraph(f"Overall Forecast: {predictive['overall']['message']}", normal_style))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Department Trend: {predictive['department']['message']}", normal_style))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Priority Trend: {predictive['priority']['message']}", normal_style))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"Response Time Forecast: {predictive['response_time']['message']}", normal_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("7-Day Forecast", section_style))
    forecast = predictive["forecast_7_days"]

    if forecast["forecast_dates"]:
        forecast_data = [["Date", "Projected Tickets"]]
        for d, v in zip(forecast["forecast_dates"], forecast["forecast_values"]):
            forecast_data.append([d, str(v)])

        forecast_table = Table(forecast_data, colWidths=[220, 180])
        forecast_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ff69b4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9d9d9")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(forecast_table)
        elements.append(Spacer(1, 10))

        elements.append(Paragraph(
            "This 7-day forecast provides a short-term view of expected ticket demand to support resource planning and workload preparation.",
            normal_style
        ))
    else:
        elements.append(Paragraph(
            "No 7-day forecast could be generated due to limited historical data.",
            normal_style
        ))

    doc.build(elements)
    return file_path