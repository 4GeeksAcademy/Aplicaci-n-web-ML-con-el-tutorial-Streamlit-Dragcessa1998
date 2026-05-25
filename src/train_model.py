from __future__ import annotations

from utils import MODEL_PATH, train_and_save_model


def main() -> None:
    results = train_and_save_model()
    print("Streamlit ML project training complete")
    print(f"Dataset rows: {results['dataset']['rows']}")
    print(
        "Optimized model - "
        f"accuracy: {results['optimized_model']['accuracy']:.3f}, "
        f"macro F1: {results['optimized_model']['macro_f1']:.3f}"
    )
    print(f"Best params: {results['optimized_model']['best_params']}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
