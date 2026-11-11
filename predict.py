"""Serve the trained demand forecasting model over HTTP.

Usage:
    uv run python predict.py

Endpoints:
    GET  /health   Service and model status.
    POST /predict  Forecast demand for one or more instances.
    GET  /         Usage summary.

Example request:
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"instances": [{"demand_lag_1": 5.0, ...}]}'
"""

import os

from perishable_demand_forecasting.api.app import app
from perishable_demand_forecasting.api.model import load_model

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


if __name__ == "__main__":
    host = os.environ.get("HOST", DEFAULT_HOST)
    port = int(os.environ.get("PORT", DEFAULT_PORT))

    load_model()

    print(f"Serving demand forecasts on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
