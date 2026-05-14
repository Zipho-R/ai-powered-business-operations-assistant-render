import pandas as pd
from datetime import timedelta


def get_trend_label(current, previous, tolerance=0.05):
    if previous == 0:
        return "Stable"

    change = (current - previous) / previous

    if change > tolerance:
        return "Increasing"
    elif change < -tolerance:
        return "Decreasing"
    return "Stable"


def overall_ticket_forecast(df):
    if df.empty or "date" not in df.columns:
        return {
            "avg_tickets_per_day": 0,
            "trend": "Stable",
            "prediction": 0,
            "message": "No ticket data is available for forecasting."
        }

    temp_df = df.copy()
    temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce")
    temp_df = temp_df.dropna(subset=["date"])

    if temp_df.empty:
        return {
            "avg_tickets_per_day": 0,
            "trend": "Stable",
            "prediction": 0,
            "message": "No valid ticket data is available for forecasting."
        }

    daily_counts = temp_df.groupby("date")["TicketID"].nunique().sort_index()

    if daily_counts.empty:
        return {
            "avg_tickets_per_day": 0,
            "trend": "Stable",
            "prediction": 0,
            "message": "No ticket data is available for forecasting."
        }

    avg_tickets = round(daily_counts.mean(), 1)

    if len(daily_counts) == 1:
        prediction = round(avg_tickets)
        return {
            "avg_tickets_per_day": avg_tickets,
            "trend": "Stable",
            "prediction": prediction,
            "message": f"Based on limited historical data, the expected ticket volume for the next day is approximately {prediction} ticket(s)."
        }

    last_day = daily_counts.iloc[-1]
    previous_day = daily_counts.iloc[-2]
    trend = get_trend_label(last_day, previous_day)

    if trend == "Increasing":
        prediction = round(last_day + max(1, (last_day - previous_day)))
    elif trend == "Decreasing":
        prediction = round(max(0, last_day - abs(last_day - previous_day)))
    else:
        prediction = round(avg_tickets)

    message = (
        f"Ticket volume is currently {trend.lower()}, with an average of "
        f"{avg_tickets} ticket(s) per day. The projected workload for the next day "
        f"is approximately {prediction} ticket(s)."
    )

    return {
        "avg_tickets_per_day": avg_tickets,
        "trend": trend,
        "prediction": prediction,
        "message": message
    }


def department_forecast(df):
    if df.empty or "category" not in df.columns:
        return {
            "top_department": "N/A",
            "trend": "Stable",
            "message": "No department data is available."
        }

    temp_df = df.copy()
    dept_counts = temp_df["category"].value_counts()

    if dept_counts.empty:
        return {
            "top_department": "N/A",
            "trend": "Stable",
            "message": "No department data is available."
        }

    top_department = dept_counts.idxmax()

    if "date" not in temp_df.columns:
        return {
            "top_department": top_department,
            "trend": "Stable",
            "message": f"{top_department} currently has the highest ticket volume and is expected to remain the busiest department."
        }

    temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce")
    temp_df = temp_df.dropna(subset=["date"])

    dept_daily = (
        temp_df[temp_df["category"] == top_department]
        .groupby("date")
        .size()
        .sort_index()
    )

    if len(dept_daily) < 2:
        trend = "Stable"
    else:
        trend = get_trend_label(dept_daily.iloc[-1], dept_daily.iloc[-2])

    message = (
        f"{top_department} currently has the highest ticket volume and is expected "
        f"to continue receiving the most tickets. The trend for this department is "
        f"{trend.lower()}."
    )

    return {
        "top_department": top_department,
        "trend": trend,
        "message": message
    }


def priority_forecast(df):
    if df.empty or "priority" not in df.columns:
        return {
            "top_priority": "N/A",
            "trend": "Stable",
            "message": "No priority data is available."
        }

    temp_df = df.copy()
    priority_counts = temp_df["priority"].value_counts()

    if priority_counts.empty:
        return {
            "top_priority": "N/A",
            "trend": "Stable",
            "message": "No priority data is available."
        }

    top_priority = priority_counts.idxmax()
    high_df = temp_df[temp_df["priority"] == "High"]

    if "date" in temp_df.columns and not high_df.empty:
        temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce")
        high_df = temp_df[temp_df["priority"] == "High"].dropna(subset=["date"])
        high_daily = high_df.groupby("date").size().sort_index()

        if len(high_daily) >= 2:
            trend = get_trend_label(high_daily.iloc[-1], high_daily.iloc[-2])
        else:
            trend = "Stable"
    else:
        trend = "Stable"

    message = (
        f"{top_priority} priority tickets are currently the most common. "
        f"High-priority ticket demand appears to be {trend.lower()}, which should be monitored closely."
    )

    return {
        "top_priority": top_priority,
        "trend": trend,
        "message": message
    }


def response_time_forecast(df):
    if df.empty or "response_time_hours" not in df.columns:
        return {
            "average_response_time": 0,
            "trend": "Stable",
            "message": "No response-time data is available."
        }

    temp_df = df.copy()
    temp_df["response_time_hours"] = pd.to_numeric(temp_df["response_time_hours"], errors="coerce")
    temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce") if "date" in temp_df.columns else pd.NaT
    temp_df = temp_df.dropna(subset=["response_time_hours"])

    if temp_df.empty:
        return {
            "average_response_time": 0,
            "trend": "Stable",
            "message": "No valid response-time data is available."
        }

    avg_response = round(temp_df["response_time_hours"].mean(), 1)

    if "date" in df.columns:
        daily_response = (
            temp_df.dropna(subset=["date"])
            .groupby("date")["response_time_hours"]
            .mean()
            .sort_index()
        )

        if len(daily_response) >= 2:
            trend = get_trend_label(daily_response.iloc[-1], daily_response.iloc[-2])
        else:
            trend = "Stable"
    else:
        trend = "Stable"

    impact = (
        "This may reduce efficiency and affect user satisfaction."
        if avg_response > 24
        else "This suggests acceptable handling performance for current demand."
    )

    message = f"The average response time is {avg_response} hours and the trend is {trend.lower()}. {impact}"

    return {
        "average_response_time": avg_response,
        "trend": trend,
        "message": message
    }


def forecast_next_7_days(df):
    if df.empty or "date" not in df.columns or "TicketID" not in df.columns:
        return {
            "historical_dates": [],
            "historical_values": [],
            "forecast_dates": [],
            "forecast_values": []
        }

    temp_df = df.copy()
    temp_df["date"] = pd.to_datetime(temp_df["date"], errors="coerce")
    temp_df = temp_df.dropna(subset=["date"])

    if temp_df.empty:
        return {
            "historical_dates": [],
            "historical_values": [],
            "forecast_dates": [],
            "forecast_values": []
        }

    daily_counts = temp_df.groupby("date")["TicketID"].nunique().sort_index()

    if daily_counts.empty:
        return {
            "historical_dates": [],
            "historical_values": [],
            "forecast_dates": [],
            "forecast_values": []
        }

    historical_dates = [d.strftime("%Y-%m-%d") for d in daily_counts.index]
    historical_values = daily_counts.tolist()

    if len(daily_counts) == 1:
        avg_change = 0
    else:
        changes = daily_counts.diff().dropna()
        avg_change = changes.mean() if not changes.empty else 0

    last_date = daily_counts.index[-1]
    last_value = float(daily_counts.iloc[-1])

    forecast_dates = []
    forecast_values = []
    current_value = last_value

    for i in range(1, 8):
        next_date = last_date + timedelta(days=i)
        current_value = max(0, current_value + avg_change)
        forecast_dates.append(next_date.strftime("%Y-%m-%d"))
        forecast_values.append(round(current_value))

    return {
        "historical_dates": historical_dates,
        "historical_values": historical_values,
        "forecast_dates": forecast_dates,
        "forecast_values": forecast_values
    }


def build_predictive_insights(df):
    return {
        "overall": overall_ticket_forecast(df),
        "department": department_forecast(df),
        "priority": priority_forecast(df),
        "response_time": response_time_forecast(df),
        "forecast_7_days": forecast_next_7_days(df)
    }