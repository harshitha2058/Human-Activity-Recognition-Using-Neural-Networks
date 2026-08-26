from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from preprocess import load_and_preprocess_data
from model import build_mlp_model

ROOT = Path(__file__).resolve().parents[1]

def run_pipeline(train_path, test_path=None, assets_dir=ROOT / 'assets', model_path=ROOT / 'saved_har_model.h5', epochs: int = 22):
    assets_dir = Path(assets_dir)
    model_path = Path(model_path)
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading and preprocessing dataset...")
    X_train, y_train, X_test, y_test, encoder = load_and_preprocess_data(train_path, test_path)

    print("[2/4] Initializing Deep Neural Network...")
    model = build_mlp_model(input_dim=X_train.shape[1], num_classes=y_train.shape[1])

    print("[3/4] Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=256,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )

    print("[4/4] Evaluating performance and saving assets...")
    y_pred = model.predict(X_test)
    y_test_class = np.argmax(y_test, axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)

    print(f"Overall Accuracy: {accuracy_score(y_test_class, y_pred_class)*100:.2f}%")
    print(classification_report(y_test_class, y_pred_class, target_names=encoder.classes_))

    # Save Accuracy & Loss plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy Curves')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss Curves')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(assets_dir / 'accuracy_loss_curves.png')
    plt.close()

    # Save Confusion Matrix plot
    cm = confusion_matrix(y_test_class, y_pred_class)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(assets_dir / 'confusion_matrix.png')
    plt.close()

    model.save(model_path)
    print(f"Saved trained model to {model_path}")
    return model

if __name__ == '__main__':
    run_pipeline(ROOT / 'data' / 'train.csv', ROOT / 'data' / 'test.csv')
