"""Flask application exposing the trained demand forecasting model."""

from flask import Flask, jsonify, request

from perishable_demand_forecasting.api.model import load_model, predict_instances


def create_app() -> Flask:
    """Create the Flask application."""

    app = Flask(__name__)

    @app.get("/")
    def index():
        bundle = load_model()
        return jsonify(
            {
                "service": "perishable-demand-forecasting",
                "model": bundle.get("family", "ridge_regression"),
                "params": bundle.get("params"),
                "trained_at": bundle["trained_at"],
                "features": bundle["features"],
                "endpoints": {
                    "GET /health": "service status",
                    "POST /predict": 'JSON body: {"instances": [{...}]}',
                },
            }
        )

    @app.get("/health")
    def health():
        try:
            bundle = load_model()
        except FileNotFoundError as error:
            return jsonify({"status": "error", "detail": str(error)}), 503

        return jsonify(
            {
                "status": "ok",
                "model": bundle.get("family", "ridge_regression"),
                "params": bundle.get("params"),
                "trained_at": bundle["trained_at"],
            }
        )

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)

        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        if isinstance(payload, dict) and "instances" in payload:
            instances = payload["instances"]
        elif isinstance(payload, dict) and "features" in payload:
            instances = [payload["features"]]
        else:
            return (
                jsonify(
                    {
                        "error": (
                            'Expected {"instances": [features, ...]} '
                            'or {"features": {...}}.'
                        )
                    }
                ),
                400,
            )

        if not isinstance(instances, list) or not instances:
            return jsonify({"error": "'instances' must be a non-empty list."}), 400

        try:
            predictions = predict_instances(instances)
        except (ValueError, FileNotFoundError) as error:
            return jsonify({"error": str(error)}), 400

        target = load_model()["target"]

        if len(predictions) == 1:
            return jsonify({"prediction": predictions[0], "target": target})

        return jsonify(
            {
                "predictions": predictions,
                "count": len(predictions),
                "target": target,
            }
        )

    return app


app = create_app()
