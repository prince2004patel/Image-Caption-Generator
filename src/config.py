import os

ARTIFACTS_DIR = "artifacts"

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "img_caption_model.keras")
PARAMS_PATH = os.path.join(ARTIFACTS_DIR, "model_params.json")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer.pkl")