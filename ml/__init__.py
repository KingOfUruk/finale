from pathlib import Path

# Ensure a default models directory exists for persisted estimators
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

__all__ = ["MODELS_DIR"]
