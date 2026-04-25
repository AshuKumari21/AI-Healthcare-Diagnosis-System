import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.staticfiles import StaticFiles

# Add current directory to sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

# Import the apps
from api.main import app as api_app
from dashboard.app import server as dash_app

# 1. Initialize the main FastAPI container
app = FastAPI(title="Integrated Healthcare System | Clinical Intelligence")

# 2. Serve Static & Assets Folders (EXPLICIT SERVING)
# This ensures Render always finds the CSS regardless of middleware routing
# Mount root-level static folder
if os.path.exists(os.path.join(BASE_DIR, "static")):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount Dash assets folder explicitly to fix broken CSS
if os.path.exists(os.path.join(BASE_DIR, "dashboard", "assets")):
    app.mount("/assets", StaticFiles(directory="dashboard/assets"), name="assets")

# 3. Mount the API backend at /api
app.mount("/api", api_app)

# 4. Mount the Dash/Flask frontend at the root /
app.mount("/", WSGIMiddleware(dash_app))

if __name__ == "__main__":
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get("PORT", 8000))
    print(f"--- Starting Unified Server on port {port} ---")
    print(f"--- API available at: http://0.0.0.0:{port}/api ---")
    print(f"--- Dashboard available at: http://0.0.0.0:{port}/ ---")
    uvicorn.run(app, host="0.0.0.0", port=port)
