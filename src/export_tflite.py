from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]


def convert_to_tflite(keras_model_path: str | Path, output_tflite_path: str | Path):
    keras_model_path = Path(keras_model_path)
    output_tflite_path = Path(output_tflite_path)
    if not keras_model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {keras_model_path}. Run `python src/train.py` first."
        )

    model = tf.keras.models.load_model(keras_model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    output_tflite_path.parent.mkdir(parents=True, exist_ok=True)
    output_tflite_path.write_bytes(converter.convert())
    print(f"Quantized TFLite model exported to {output_tflite_path}")


if __name__ == "__main__":
    convert_to_tflite(ROOT / "saved_har_model.h5", ROOT / "har_model_quantized.tflite")
