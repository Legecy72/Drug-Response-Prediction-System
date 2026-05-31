# Drug Response Web App

Organized FastAPI project for GDSC drug response prediction and recommendation.

## Structure

```text
project/
  api/                         FastAPI app, routers, chatbot endpoints
  core/                        Settings, schemas, prediction services, ML engines
  data/                        Prompt/data files used by support modules
  models/                      Saved regression/classification artifacts
  web/                         Static frontend files
  scripts/                     Run helpers
  docs/                        Chatbot and project documentation
  screenshots/                 UI screenshots for project reporting
  tests/                       Chatbot/API tests
  .env.example
  requirements.txt
  README.md
  LICENSE
```

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
.\scripts\run.ps1
```

Open:

```text
http://127.0.0.1:8000
```

If port `8000` is busy:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8001
```

## Main Endpoints

- `GET /api/status`
- `GET /api/options`
- `GET /api/sample-input`
- `POST /api/predict`
- `POST /api/recommend`
- `POST /api/chat`

## Notes

- The frontend is in `web/` and is served by FastAPI.
- Saved ML artifacts are in `models/`.
- `.env.example` is sanitized and does not include a real API key.
