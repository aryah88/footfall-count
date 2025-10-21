from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

app = FastAPI(title="Footfall Counter Analytics")

LOG_PATH = "logs/footfall_log.csv"
CHART_PATH = "charts/footfall_chart.png"
os.makedirs("charts", exist_ok=True)

# === 1️⃣ /stats — Summary Counts ===
@app.get("/stats")
def get_stats():
    """Return total IN, OUT, and Net count."""
    if not os.path.exists(LOG_PATH):
        return JSONResponse({"error": "No data logged yet."}, status_code=404)

    df = pd.read_csv(LOG_PATH, names=["timestamp", "id", "direction"])
    total_in = (df["direction"] == "IN").sum()
    total_out = (df["direction"] == "OUT").sum()
    net = total_in - total_out

    return {
        "total_in": int(total_in),
        "total_out": int(total_out),
        "net_occupancy": int(net)
    }

# === 2️⃣ /charts — Static Chart Endpoint ===
@app.get("/charts")
def get_chart(type: str = Query("cumulative", enum=["hourly", "daily", "cumulative"])):
    """Generate and return a footfall chart (hourly, daily, or cumulative)."""
    if not os.path.exists(LOG_PATH):
        return JSONResponse({"error": "No data logged yet."}, status_code=404)

    df = pd.read_csv(LOG_PATH, names=["timestamp", "id", "direction"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    plt.figure(figsize=(8, 4))
    if type == "hourly":
        sns.countplot(x="hour", hue="direction", data=df)
        plt.title("Hourly Footfall Distribution")
    elif type == "daily":
        sns.countplot(x="date", hue="direction", data=df)
        plt.title("Daily Footfall Distribution")
    else:
        df["count"] = df["direction"].map({"IN": 1, "OUT": -1}).cumsum()
        plt.plot(df["timestamp"], df["count"], color="blue")
        plt.title("Cumulative Footfall Trend")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Count")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(CHART_PATH)
    plt.close()

    return FileResponse(CHART_PATH, media_type="image/png")

# === 3️⃣ /charts/live — Auto-Refreshing Chart Page ===
@app.get("/charts/live")
def live_chart(type: str = Query("cumulative", enum=["hourly", "daily", "cumulative"])):
    """Serve an auto-refreshing chart page that updates every 5 seconds."""
    chart_url = f"/charts?type={type}"
    html = f"""
    <html>
        <head>
            <meta http-equiv="refresh" content="5">
            <title>Live Footfall Chart</title>
            <style>
                body {{
                    background-color: #f8f9fa;
                    text-align: center;
                    font-family: Arial, sans-serif;
                }}
                img {{
                    width: 90%;
                    margin-top: 20px;
                    border: 2px solid #ccc;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
            </style>
        </head>
        <body>
            <h2>Live Footfall Tracker</h2>
            <p>Auto-refresh every 5 seconds</p>
            <img src="{chart_url}" alt="Live Footfall Chart">
        </body>
    </html>
    """
    return HTMLResponse(content=html)
