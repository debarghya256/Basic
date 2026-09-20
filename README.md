# 🌸 Iris Classifier: End-to-End ML Project

Train a model → serve it as an API → containerize → deploy.

```
train.py ─► model/iris_model.joblib ─► app/main.py (FastAPI) ─► Docker ─► Cloud
```

## Project layout

```
iris-ml-app/
├── train.py             # trains + evaluates + saves the model
├── app/main.py          # FastAPI: GET /, GET /health, POST /predict
├── tests/test_api.py    # automated tests
├── Dockerfile           # packages everything (trains the model at build time)
├── requirements.txt     # runtime deps
└── requirements-dev.txt # + pytest for testing
```

## 1. Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python train.py                    # creates model/
pytest                             # should show 4 passed
uvicorn app.main:app --reload
```

Open:
- http://localhost:8000       → simple web form
- http://localhost:8000/docs  → auto-generated interactive API docs

Try the API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":6.0,"sepal_width":2.9,"petal_length":4.5,"petal_width":1.5}'
```

## 2. Run with Docker

```bash
docker build -t iris-app .
docker run -p 8000:8000 iris-app
```

## 3. Deploy (free-tier options)

Push the project to GitHub first:

```bash
git init && git add . && git commit -m "Iris ML app"
# create an empty repo on github.com, then:
git remote add origin https://github.com/<you>/iris-ml-app.git
git branch -M main && git push -u origin main
```

### Option A: Render (easiest)
1. Sign up at render.com → New → Web Service → connect your GitHub repo.
2. Environment: Docker (it detects the Dockerfile).
3. Instance type: Free → Create Web Service.
4. After a few minutes you get a public URL like https://iris-app.onrender.com.
   Free instances sleep when idle, so the first request can take 30-60s.

### Option B: Hugging Face Spaces
Create a new Space with the Docker SDK, push these files, and add `app_port: 8000`
to the Space's README header.

### Option C: Google Cloud Run
```bash
gcloud run deploy iris-app --source . --allow-unauthenticated --region us-central1
```

## 4. Verify the deployment

```bash
curl https://<your-url>/health
```

## Ideas to extend it

- Swap the dataset (Wine, Titanic, your own CSV) and retrain
- Add a GitHub Actions workflow that runs `pytest` on every push
- Log predictions and add a `/metrics` endpoint
- Compare models with cross-validation (LogisticRegression, SVM)
- Add a Streamlit front-end that calls the API

## Notes

- The model is trained inside the Docker image so the container is self-contained.
- Keep the scikit-learn version identical for training and serving, otherwise
  loading the saved model can fail or warn.
