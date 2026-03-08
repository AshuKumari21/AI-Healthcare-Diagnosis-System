# Aegis AI Healthcare Dashboard

Advanced Multi-Disease Diagnostic Platform using Deep Neural Ensembles.

## 🚀 Google OAuth Configuration

To enable the "Continue with Google" login feature, you must configure your Google Cloud Credentials.

### 1. Create Google Cloud Project
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project.
3. Navigate to **APIs & Services > Credentials**.
4. Click **Create Credentials > OAuth client ID**.
5. Configure the Consent Screen (Internal or External).
6. Select **Web Application**.
7. Add **Authorized redirect URIs**:
   - `http://127.0.0.1:8050/login/google/callback`

### 2. Configure Environment Variables
Create a file named `.env` in the root directory (the same folder as this README) and add your keys:

```env
# Google OAuth
GOOGLE_CLIENT_ID=your-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-secret-key

# Flask Security
FLASK_SECRET_KEY=any-random-long-string
```

### 3. Run the Application
1. Install dependencies: `pip install dash dash-bootstrap-components pandas flask requests python-dotenv`
2. Start the FastAPI backend: `python api/main.py`
3. Start the Dash dashboard: `python dashboard/app.py`
4. Open `http://127.0.0.1:8050` in your browser.

## 🛠 Features
- **Multi-Disease Analysis**: Diabetes, Heart, Stroke, Kidney, Liver, and more.
- **Neural Pulse Engine**: Real-time AI inference powered by specialized ML models.
- **Clinical Reports**: Automated PDF/text report synthesis.
- **Aegis AI Assistant**: Natural language medical guidance.
