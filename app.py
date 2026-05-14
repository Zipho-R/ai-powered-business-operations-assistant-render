from flask import Flask, render_template, request, redirect, send_file, url_for
from report_generator import generate_pdf_report
import pandas as pd
from datetime import datetime
import os
import json
import sqlite3
import random
from predictive_insights import build_predictive_insights
from TicketClassifierV2 import process_ticket

app = Flask(__name__)

file_name = "tickets.csv"
DB_FILE = "tickets.db"
TABLE_NAME = "tickets"

ADMIN_PASSWORD = "PBP"
DEPARTMENT_PASSWORD = "PBP"

ALLOWED_DEPARTMENTS = ["HR", "IT", "Finance", "Operations", "Unclassified"]

required_columns = [
    "TicketID",

    # Requester details
    "first_name", "surname", "email",

    # Ticket details
    "request", "category", "priority", "priority_code", "response",
    "date", "urgency", "response_time_hours", "started_at", "closed_at",
    "resolution_comment",

    # Workflow details
    "Status", "review_status", "review_notes", "routed_to", "routed_at",
    "risk_flags", "transparency_note", "approval_required", "approval_status",

    # Existing ML fields
    "ml_confidence", "classification_method",

    # Phase 2 AI fields
    "category_confidence", "priority_confidence", "urgency_confidence",
    "priority_method", "urgency_method"
]

# ==========================================================
# Routing, risk, and workflow helper functions
# ==========================================================

def route_ticket(category):
    mapping = {
        "IT": "IT Department",
        "HR": "HR Department",
        "Finance": "Finance Department",
        "Operations": "Operations Department"
    }
    return mapping.get(category, "Unclassified Queue")

def notify_department(department, ticket_id, message):
    print(f"[AUTO ROUTE] Ticket {ticket_id} sent to {department}")
    print(f"[MESSAGE] {message}")


def get_transparency_note():
    return (
        "This ticket was classified using a hybrid AI system combining machine learning "
        "and rule-based automation. Risk flags, sensitive cases, finance approvals, "
        "and low-confidence tickets may require human review."
    )


def requires_approval(category, ticket=""):
    text = str(ticket).lower()

    if category == "Finance":
        return True

    approval_words = [
        "fraud", "fraudulent", "unauthorized", "unauthorised",
        "breach", "hacked", "legal", "termination",
        "harassment", "discrimination", "injury", "accident"
    ]

    return any(word in text for word in approval_words)


def risk_check(ticket, category):
    text = str(ticket).lower()
    risks = []

    if category == "Unclassified":
        risks.append("Low confidence classification - manual review required")

    if category == "HR" and any(word in text for word in ["harassment", "discrimination", "abuse", "termination", "bullying"]):
        risks.append("Sensitive HR issue - human review recommended")

    if category == "Finance" and any(word in text for word in ["fraud", "fraudulent", "unauthorized", "unauthorised", "missing", "suspicious"]):
        risks.append("Financial risk detected - approval or audit review recommended")

    if category == "IT" and any(word in text for word in ["breach", "virus", "hacked", "malware"]):
        risks.append("Cybersecurity risk detected")

    if category == "Operations" and any(word in text for word in ["accident", "injury", "safety"]):
        risks.append("Health and safety risk detected")

    return ", ".join(risks) if risks else "No major risk detected"


# ==========================================================
# Response-time helper functions
# ==========================================================

def generate_response_time_by_priority(priority):
    """
    Generates realistic demo response times based on priority.
    Returned as string because the database/CSV stores this column safely as text.
    """
    priority = str(priority).strip()

    if priority == "High":
        hours = round(random.uniform(1, 8), 2)
    elif priority == "Medium":
        hours = round(random.uniform(8, 24), 2)
    else:
        hours = round(random.uniform(24, 72), 2)

    return str(hours)


def calculate_response_time(started_at, priority):
    """
    Calculates real elapsed time when started_at exists.
    Falls back to a realistic demo response time if the ticket was closed without being started.
    Always returns a string to prevent pandas dtype errors.
    """
    now = datetime.now()
    started_value = str(started_at).strip()

    try:
        if started_value and started_value.lower() not in ["nan", "none", "", "nat"]:
            started_time = datetime.strptime(started_value, "%Y-%m-%d %H:%M:%S")
            response_hours = round((now - started_time).total_seconds() / 3600, 2)

            if response_hours < 0:
                response_hours = 0

            return str(response_hours)
    except Exception:
        pass

    return generate_response_time_by_priority(priority)


def is_missing_response_time(value):
    if value is None:
        return True

    text = str(value).strip().lower()

    if text in ["", "nan", "none", "nat", "pending", "pending closure"]:
        return True

    try:
        float(value)
        return False
    except Exception:
        return True


# ==========================================================
# Database helpers
# ==========================================================

def get_connection():
    return sqlite3.connect(DB_FILE)


def load_tickets_df():
    """
    Loads tickets safely from SQLite.
    If old database records do not have the new columns, they are added automatically.
    This keeps old tickets from breaking when the schema evolves.
    """
    with get_connection() as conn:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        except Exception:
            df = pd.DataFrame(columns=required_columns)

    for col in required_columns:
        if col not in df.columns:
            if col == "Status":
                df[col] = "Open"
            elif col == "transparency_note":
                df[col] = get_transparency_note()
            elif col == "approval_required":
                df[col] = "No"
            elif col == "approval_status":
                df[col] = "Not Required"
            elif col in ["classification_method", "priority_method", "urgency_method"]:
                df[col] = "Existing record"
            else:
                df[col] = ""

    # Keep these columns text/object to avoid pandas string/float crashes.
    df["response_time_hours"] = df["response_time_hours"].astype("object")
    df["resolution_comment"] = df["resolution_comment"].astype("object")

    # Normalise important text columns.
    df["Status"] = df["Status"].astype(str).str.strip()
    df["TicketID"] = df["TicketID"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    # Repair old closed tickets that have missing response times.
    for idx in df.index:
        status = str(df.at[idx, "Status"]).strip()
        response_time = df.at[idx, "response_time_hours"]

        if status == "Closed" and is_missing_response_time(response_time):
            df.at[idx, "response_time_hours"] = generate_response_time_by_priority(df.at[idx, "priority"])

    return df[required_columns]


def save_tickets_df(df_to_save):
    """
    Saves to both SQLite and CSV backup.
    Keeps the same column order to avoid breaking dashboards/reports.
    """
    for col in required_columns:
        if col not in df_to_save.columns:
            df_to_save[col] = ""

    df_to_save = df_to_save[required_columns].copy()

    # Safe text storage for fields that may be blank or numeric-like.
    for text_col in ["response_time_hours", "resolution_comment", "first_name", "surname", "email"]:
        df_to_save[text_col] = df_to_save[text_col].astype("object")
        df_to_save[text_col] = df_to_save[text_col].apply(
            lambda x: "" if str(x).strip().lower() in ["nan", "none", "nat"] else str(x).strip()
        )

    with get_connection() as conn:
        df_to_save.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    df_to_save.to_csv(file_name, index=False)


if not os.path.exists(file_name):
    pd.DataFrame(columns=required_columns).to_csv(file_name, index=False)


# ==========================================================
# Ticket processing helpers
# ==========================================================

def next_ticket_id(df):
    if df.empty:
        return "T001"

    existing_ids = df["TicketID"].dropna().astype(str)
    numbers = []

    for ticket_id in existing_ids:
        if ticket_id.startswith("T") and ticket_id[1:].isdigit():
            numbers.append(int(ticket_id[1:]))

    if not numbers:
        return "T001"

    return f"T{str(max(numbers) + 1).zfill(3)}"


def process_ticket_rows(ticket):
    data = process_ticket(ticket)

    categories = data["categories"].split(",") if data.get("categories") else ["Unclassified"]
    rows = []

    for category in categories:
        category = category.strip()

        approval_needed = requires_approval(category, ticket)
        is_unclassified = category == "Unclassified"

        if is_unclassified:
            status = "Pending Review"
            approval_required = "No"
            approval_status = "Not Required"
        elif approval_needed:
            status = "Pending Approval"
            approval_required = "Yes"
            approval_status = "Pending"
        else:
            status = "Open"
            approval_required = "No"
            approval_status = "Not Required"

        rows.append({
            "category": category,
            "priority": data["priority"],
            "priority_code": data["priority_code"],
            "response": data["response"],
            "urgency": data["urgency"],
            "response_time_hours": "",
            "started_at": "",
            "closed_at": "",
            "resolution_comment": "",
            "Status": status,
            "review_status": "Not Reviewed" if is_unclassified else "",
            "review_notes": "",
            "routed_to": route_ticket(category),
            "routed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_flags": risk_check(ticket, category),
            "transparency_note": data.get("transparency_note", get_transparency_note()),
            "approval_required": approval_required,
            "approval_status": approval_status,

            "ml_confidence": data.get("ml_confidence", ""),
            "classification_method": data.get("classification_method", ""),

            "category_confidence": data.get("category_confidence", data.get("ml_confidence", "")),
            "priority_confidence": data.get("priority_confidence", ""),
            "urgency_confidence": data.get("urgency_confidence", ""),
            "priority_method": data.get("priority_method", ""),
            "urgency_method": data.get("urgency_method", "")
        })

    return {
        "rows": rows,
        "category_display": data["category_display"],
        "priority": data["priority"],
        "priority_code": data["priority_code"],
        "urgency": data["urgency"],
        "response": data["response"],
        "risk_flags": data.get("risk_flags", ""),
        "transparency_note": data.get("transparency_note", get_transparency_note()),

        "ml_confidence": data.get("ml_confidence", ""),
        "classification_method": data.get("classification_method", ""),

        "category_confidence": data.get("category_confidence", data.get("ml_confidence", "")),
        "priority_confidence": data.get("priority_confidence", ""),
        "urgency_confidence": data.get("urgency_confidence", ""),
        "priority_method": data.get("priority_method", ""),
        "urgency_method": data.get("urgency_method", "")
    }


# ==========================================================
# Pages and routes
# ==========================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/portal", methods=["GET", "POST"])
def portal():
    error = None

    if request.method == "POST":
        first_name = request.form.get("first_name", "").strip()
        surname = request.form.get("surname", "").strip()
        department = request.form.get("department", "").strip()
        password = request.form.get("password", "").strip()

        if not first_name or not surname:
            error = "Please enter your name and surname."
        elif department not in ALLOWED_DEPARTMENTS:
            error = "Please select a valid department."
        elif password != DEPARTMENT_PASSWORD:
            error = "Incorrect department password."
        else:
            user_name = f"{first_name} {surname}"
            return redirect(f"/department/{department}?user_name={user_name}")

    return render_template("portal.html", departments=ALLOWED_DEPARTMENTS, error=error)


@app.route("/log-ticket", methods=["GET", "POST"])
def index():
    result = None
    df = load_tickets_df()

    if request.method == "POST":
        first_name = request.form.get("first_name", "").strip()
        surname = request.form.get("surname", "").strip()
        email = request.form.get("email", "").strip()
        ticket = request.form.get("ticket", "").strip()

        if ticket and first_name and surname:
            data = process_ticket_rows(ticket)
            ticket_id = next_ticket_id(df)
            today = datetime.now().strftime("%Y-%m-%d")

            new_rows = []

            for row in data["rows"]:
                new_rows.append({
                    "TicketID": ticket_id,
                    "first_name": first_name,
                    "surname": surname,
                    "email": email,
                    "request": ticket,
                    "category": row["category"],
                    "priority": row["priority"],
                    "priority_code": row["priority_code"],
                    "response": row["response"],
                    "date": today,
                    "urgency": row["urgency"],
                    "response_time_hours": row["response_time_hours"],
                    "started_at": row["started_at"],
                    "closed_at": row["closed_at"],
                    "resolution_comment": row["resolution_comment"],
                    "Status": row["Status"],
                    "review_status": row["review_status"],
                    "review_notes": row["review_notes"],
                    "routed_to": row["routed_to"],
                    "routed_at": row["routed_at"],
                    "risk_flags": row["risk_flags"],
                    "transparency_note": row["transparency_note"],
                    "approval_required": row["approval_required"],
                    "approval_status": row["approval_status"],

                    "ml_confidence": row["ml_confidence"],
                    "classification_method": row["classification_method"],

                    "category_confidence": row["category_confidence"],
                    "priority_confidence": row["priority_confidence"],
                    "urgency_confidence": row["urgency_confidence"],
                    "priority_method": row["priority_method"],
                    "urgency_method": row["urgency_method"]
                })

            updated_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_tickets_df(updated_df)

            for row in new_rows:
                if row["approval_required"] == "Yes":
                    print(f"[APPROVAL REQUIRED] Ticket {ticket_id} awaiting approval before department notification.")
                else:
                    notify_department(row["routed_to"], ticket_id, row["response"])

            risk_values = sorted(set(row["risk_flags"] for row in new_rows))
            if len(risk_values) > 1 and "No major risk detected" in risk_values:
                risk_values.remove("No major risk detected")

            result = {
                "ticket_text": ticket,
                "requester": f"{first_name} {surname}",
                "email": email,
                "category": data["category_display"],
                "priority": data["priority"],
                "priority_code": data["priority_code"],
                "urgency": data["urgency"],
                "response": data["response"],
                "ticket_id": ticket_id,
                "risk_flags": "; ".join(risk_values),
                "transparency_note": data["transparency_note"],
                "approval_required": "Yes" if any(row["approval_required"] == "Yes" for row in new_rows) else "No",

                "ml_confidence": data["ml_confidence"],
                "classification_method": data["classification_method"],

                "category_confidence": data["category_confidence"],
                "priority_confidence": data["priority_confidence"],
                "urgency_confidence": data["urgency_confidence"],
                "priority_method": data["priority_method"],
                "urgency_method": data["urgency_method"]
            }

    return render_template("index.html", result=result)


@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    error = None
    next_target = request.args.get("next", request.form.get("next", "executive"))

    if request.method == "POST":
        first_name = request.form.get("first_name", "").strip()
        surname = request.form.get("surname", "").strip()
        password = request.form.get("password", "").strip()
        next_target = request.form.get("next", next_target)

        if not first_name or not surname:
            error = "Please enter your name and surname."
        elif password == ADMIN_PASSWORD:
            if next_target == "analytics":
                return redirect(url_for("analytics_dashboard"))
            elif next_target == "reports":
                return redirect(url_for("reports"))
            return redirect(url_for("admin_dashboard"))
        else:
            error = "Incorrect password"

    return render_template("admin_login.html", error=error, next_target=next_target)


@app.route("/admin-dashboard")
def admin_dashboard():
    current_df = load_tickets_df()

    if not current_df.empty:
        current_df["ticket_num"] = current_df["TicketID"].astype(str).str.replace("T", "", regex=False)
        current_df["ticket_num"] = pd.to_numeric(current_df["ticket_num"], errors="coerce").fillna(0).astype(int)
        current_df = current_df.sort_values(by="ticket_num", ascending=False)
        current_df = current_df.drop(columns=["ticket_num"])

    return render_template("admin_dashboard.html", tables=current_df.to_dict(orient="records"))


@app.route("/update-status/<ticket_id>/<new_status>", methods=["POST"])
def update_status(ticket_id, new_status):
    allowed_statuses = ["Open", "In Progress", "Closed", "Pending Review", "Pending Approval"]

    if new_status not in allowed_statuses:
        return redirect("/admin-dashboard")

    source_page = request.form.get("source_page", "admin")
    dept = request.form.get("dept", "")
    user_name = request.form.get("user_name", "Team Member")
    resolution_comment = request.form.get("resolution_comment", "").strip()

    df = load_tickets_df()
    df["response_time_hours"] = df["response_time_hours"].astype("object")
    df["resolution_comment"] = df["resolution_comment"].astype("object")

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mask = df["TicketID"].astype(str) == str(ticket_id)

    if new_status == "In Progress":
        df.loc[mask, "Status"] = "In Progress"

        for idx in df[mask].index:
            if str(df.at[idx, "started_at"]).strip().lower() in ["", "nan", "none", "nat"]:
                df.at[idx, "started_at"] = now_text

    elif new_status == "Closed":
        df.loc[mask, "Status"] = "Closed"
        df.loc[mask, "closed_at"] = now_text

        for idx in df[mask].index:
            response_hours = calculate_response_time(df.at[idx, "started_at"], df.at[idx, "priority"])
            df.at[idx, "response_time_hours"] = str(response_hours)

            # Resolution comment is only saved if the ticket does not already have one.
            # This keeps it read-only after closure.
            existing_comment = str(df.at[idx, "resolution_comment"]).strip()
            if not existing_comment or existing_comment.lower() in ["nan", "none", "nat"]:
                df.at[idx, "resolution_comment"] = resolution_comment

            if not df.at[idx, "resolution_comment"]:
                df.at[idx, "resolution_comment"] = "Closed without additional resolution notes."

    else:
        df.loc[mask, "Status"] = new_status

    save_tickets_df(df)

    if source_page == "department" and dept in ALLOWED_DEPARTMENTS:
        return redirect(f"/department/{dept}?user_name={user_name}")

    return redirect("/admin-dashboard")


@app.route("/approve-ticket/<ticket_id>", methods=["POST"])
def approve_ticket(ticket_id):
    df = load_tickets_df()

    source_page = request.form.get("source_page", "admin")
    dept = request.form.get("dept", "")
    user_name = request.form.get("user_name", "Team Member")

    mask = df["TicketID"].astype(str) == str(ticket_id)

    df.loc[mask, "approval_status"] = "Approved"
    df.loc[mask, "Status"] = "Open"

    for _, row in df[mask].iterrows():
        department = row["routed_to"] if row["routed_to"] else route_ticket(row["category"])
        notify_department(department, row["TicketID"], row["response"])

    save_tickets_df(df)

    if source_page == "department" and dept in ALLOWED_DEPARTMENTS:
        return redirect(f"/department/{dept}?user_name={user_name}")

    return redirect("/admin-dashboard")


@app.route("/route-unclassified/<ticket_id>/<target_department>", methods=["POST"])
def route_unclassified(ticket_id, target_department):
    allowed_targets = ["HR", "IT", "Finance", "Operations"]

    if target_department not in allowed_targets:
        return redirect("/department/Unclassified?user_name=Reviewer")

    df = load_tickets_df()

    user_name = request.form.get("user_name", "Reviewer")
    review_notes = request.form.get("review_notes", "").strip()
    route_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mask = (df["TicketID"].astype(str) == str(ticket_id)) & (df["category"].astype(str) == "Unclassified")
    approval_needed = requires_approval(target_department, " ".join(df.loc[mask, "request"].astype(str).tolist()))

    df.loc[mask, "category"] = target_department
    df.loc[mask, "Status"] = "Pending Approval" if approval_needed else "Open"
    df.loc[mask, "review_status"] = "Routed"
    df.loc[mask, "routed_to"] = route_ticket(target_department)
    df.loc[mask, "review_notes"] = review_notes if review_notes else f"Routed to {target_department} by reviewer"
    df.loc[mask, "routed_at"] = route_time
    df.loc[mask, "approval_required"] = "Yes" if approval_needed else "No"
    df.loc[mask, "approval_status"] = "Pending" if approval_needed else "Not Required"

    for idx in df[mask].index:
        df.at[idx, "risk_flags"] = risk_check(df.at[idx, "request"], target_department)

    save_tickets_df(df)

    return redirect(f"/department/Unclassified?user_name={user_name}")


@app.route("/mark-irrelevant/<ticket_id>", methods=["POST"])
def mark_irrelevant(ticket_id):
    df = load_tickets_df()
    df["response_time_hours"] = df["response_time_hours"].astype("object")
    df["resolution_comment"] = df["resolution_comment"].astype("object")

    user_name = request.form.get("user_name", "Reviewer")
    review_notes = request.form.get("review_notes", "").strip()
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mask = (df["TicketID"].astype(str) == str(ticket_id)) & (df["category"].astype(str) == "Unclassified")

    df.loc[mask, "review_status"] = "Irrelevant"
    df.loc[mask, "Status"] = "Closed"
    df.loc[mask, "closed_at"] = now_text
    df.loc[mask, "review_notes"] = review_notes if review_notes else "Marked as irrelevant during review"
    df.loc[mask, "routed_to"] = ""

    for idx in df[mask].index:
        if is_missing_response_time(df.at[idx, "response_time_hours"]):
            response_hours = calculate_response_time(df.at[idx, "started_at"], df.at[idx, "priority"])
            df.at[idx, "response_time_hours"] = str(response_hours)

        existing_comment = str(df.at[idx, "resolution_comment"]).strip()
        if not existing_comment or existing_comment.lower() in ["nan", "none", "nat"]:
            df.at[idx, "resolution_comment"] = "Ticket marked as irrelevant during manual review."

    save_tickets_df(df)

    return redirect(f"/department/Unclassified?user_name={user_name}")


@app.route("/reports")
def reports():
    current_df = load_tickets_df()

    categories = []
    priorities = ["High", "Medium", "Low"]

    if not current_df.empty and "category" in current_df.columns:
        categories = sorted(current_df["category"].dropna().unique().tolist())

    return render_template("reports.html", categories=categories, priorities=priorities)


@app.route("/generate-report", methods=["POST"])
def generate_report():
    current_df = load_tickets_df()

    report_type = request.form.get("report_type", "executive")
    filter_value = request.form.get("filter_value", "All")

    filtered_df = current_df.copy()

    if report_type == "department" and filter_value != "All":
        filtered_df = filtered_df[filtered_df["category"] == filter_value]
    elif report_type == "priority" and filter_value != "All":
        filtered_df = filtered_df[filtered_df["priority"] == filter_value]

    pdf_path = generate_pdf_report(filtered_df, report_type, filter_value)

    return send_file(pdf_path, as_attachment=True)


@app.route("/analytics-dashboard")
def analytics_dashboard():
    current_df = load_tickets_df()

    if current_df.empty:
        insights = {
            "overall": {"message": "No data available.", "prediction": 0, "trend": "Stable", "avg_tickets_per_day": 0},
            "department": {"message": "No data available.", "top_department": "N/A", "trend": "Stable"},
            "priority": {"message": "No data available."},
            "response_time": {"message": "No data available.", "average_response_time": 0, "trend": "Stable"},
            "forecast_7_days": {"historical_dates": [], "historical_values": [], "forecast_dates": [], "forecast_values": []}
        }

        return render_template(
            "analytics_dashboard.html",
            category_labels=json.dumps([]),
            category_values=json.dumps([]),
            priority_labels=json.dumps([]),
            priority_values=json.dumps([]),
            response_labels=json.dumps([]),
            response_values=json.dumps([]),
            day_labels=json.dumps([]),
            day_values=json.dumps([]),
            grouped_labels=json.dumps([]),
            grouped_high=json.dumps([]),
            grouped_medium=json.dumps([]),
            grouped_low=json.dumps([]),
            forecast_historical_dates=json.dumps([]),
            forecast_historical_values=json.dumps([]),
            forecast_dates=json.dumps([]),
            forecast_values=json.dumps([]),
            insights=insights
        )

    # Analytics needs numeric response time, so convert only here.
    current_df["response_time_hours"] = pd.to_numeric(current_df["response_time_hours"], errors="coerce").fillna(0)

    category_counts = current_df["category"].value_counts()

    priority_order = ["High", "Medium", "Low"]
    priority_counts = current_df["priority"].value_counts().reindex(priority_order, fill_value=0)

    response_time_by_category = current_df.groupby("category")["response_time_hours"].sum()
    tickets_per_day = current_df.groupby("date").size()

    grouped_priority = (
        current_df.groupby(["category", "priority"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=priority_order, fill_value=0)
    )

    insights = build_predictive_insights(current_df)
    forecast_data = insights.get("forecast_7_days", {})

    return render_template(
        "analytics_dashboard.html",
        category_labels=json.dumps(category_counts.index.tolist()),
        category_values=json.dumps(category_counts.values.tolist()),
        priority_labels=json.dumps(priority_counts.index.tolist()),
        priority_values=json.dumps(priority_counts.values.tolist()),
        response_labels=json.dumps(response_time_by_category.index.tolist()),
        response_values=json.dumps(response_time_by_category.values.tolist()),
        day_labels=json.dumps(tickets_per_day.index.tolist()),
        day_values=json.dumps(tickets_per_day.values.tolist()),
        grouped_labels=json.dumps(grouped_priority.index.tolist()),
        grouped_high=json.dumps(grouped_priority["High"].tolist()),
        grouped_medium=json.dumps(grouped_priority["Medium"].tolist()),
        grouped_low=json.dumps(grouped_priority["Low"].tolist()),
        forecast_historical_dates=json.dumps(forecast_data.get("historical_dates", [])),
        forecast_historical_values=json.dumps(forecast_data.get("historical_values", [])),
        forecast_dates=json.dumps(forecast_data.get("forecast_dates", [])),
        forecast_values=json.dumps(forecast_data.get("forecast_values", [])),
        insights=insights
    )


@app.route("/department/<dept>")
def department_view(dept):
    if dept not in ALLOWED_DEPARTMENTS:
        return redirect("/portal")

    current_df = load_tickets_df()
    user_name = request.args.get("user_name", "Team Member")

    filtered_df = current_df[current_df["category"] == dept].copy()

    if not filtered_df.empty:
        filtered_df["sort_time"] = filtered_df["routed_at"].replace("", pd.NA)
        filtered_df["sort_time"] = filtered_df["sort_time"].fillna(filtered_df["date"])
        filtered_df = filtered_df.sort_values(by="sort_time", ascending=False)
        filtered_df = filtered_df.drop(columns=["sort_time"])

    transparency_note = get_transparency_note()

    return render_template(
        "department.html",
        tables=filtered_df.to_dict(orient="records"),
        dept=dept,
        user_name=user_name,
        transparency_note=transparency_note
    )

if __name__ == "__main__":
    app.run(debug=True)
