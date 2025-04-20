from tensorflow.keras.models import load_model
from .utils import load_pickle, load_json
from .config import MODEL_PATH, TOKENIZER_PATH, PARAMS_PATH

def load_caption_model():
    return load_model(MODEL_PATH)

def load_tokenizer():
    return load_pickle(TOKENIZER_PATH)

def load_max_length():
    params = load_json(PARAMS_PATH)
    return params["max_sequence_length"]
