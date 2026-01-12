### Local (no docker)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# API
uvicorn src.api.main:app --reload --port 8000

# UI (new terminal)
API_URL=http://localhost:8000 streamlit run ui/app.py
