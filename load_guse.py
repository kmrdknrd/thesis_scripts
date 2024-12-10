import tensorflow_hub as hub
import tempfile
import shutil
from pathlib import Path

def load_model():
    """Loads the GUSE model from TF Hub.
    
    Returns:
        tensorflow_hub.keras_layer.KerasLayer: GUSE model.
    """

    # Load GUSE model
    try:
        model(input)
    except NameError as ne:
        print(ne)
        model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
        try:
            print("Loading model...")
            model = hub.load(model_url)
        except OSError as ose:
            print(ose)
            print("Directory already exists, but model not found. Deleting and re-downloading model...")
            temp_dir = Path(tempfile.gettempdir()) / "tfhub_modules"
            shutil.rmtree(temp_dir)
            model = hub.load(model_url)
        finally:
            print("Model loaded.")
    except ValueError as ve:
        pass

    return model