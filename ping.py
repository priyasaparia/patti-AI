# ping.py — Called by Render Cron Job every 10 min
import requests
import os

URL = os.environ.get("RENDER_EXTERNAL_URL", "https://your-app.onrender.com")

try:
    response = requests.get(f"{URL}/health", timeout=10)
    print(f"Ping success: {response.status_code}")
except Exception as e:
    print(f"Ping failed: {e}")
