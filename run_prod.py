import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware

# Add current directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the apps
from api.main import app as api_app
from dashboard.app import server as dash_app

# 1. Initialize the main FastAPI container
app = FastAPI(title="Integrated Healthcare System | Clinical Intelligence")

# 2. Mount the API backend at /api
app.mount("/api", api_app)

# 3. Mount the Dash/Flask frontend at the root /
# Using WSGIMiddleware to integrate the Flask-based Dash app
app.mount("/", WSGIMiddleware(dash_app))

if __name__ == "__main__":
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get("PORT", 8000))
    print(f"--- Starting Unified Server on port {port} ---")
    print(f"--- API available at: http://0.0.0.0:{port}/api ---")
    print(f"--- Dashboard available at: http://0.0.0.0:{port}/ ---")
    uvicorn.run(app, host="0.0.0.0", port=port)
