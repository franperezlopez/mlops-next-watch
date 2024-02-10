from typing import Optional
from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import Response
from src.services.prometheus import Monitoring
from src.pipelines.inference import Predict
from src.pipelines.training import Train
from src.pipelines.monitoring import Monitor
from src.settings import get_settings
from pathlib import Path

app = FastAPI()

# Instrument the app to collect metrics
Instrumentator().instrument(app).expose(app)
Monitoring.initialize()


@app.get("/custom-metrics")
async def metrics():
    # Generate the latest metrics from the custom registry
    return Response(
        generate_latest(Monitoring.REGISTRY), media_type=CONTENT_TYPE_LATEST
    )


@app.get("/predict")
async def predict(prediction_file_path: Path):
    Predict(get_settings()).run(prediction_file_path)


@app.get("/train")
async def train(training_file_path: Optional[Path] = None):
    Train(get_settings()).run(training_file_path)


@app.get("/monitor")
async def monitor():
    Monitor(get_settings()).run()
