from model_pipeline import FakeNewsPipeline

pipeline = FakeNewsPipeline()

if __name__ == "__main__":
    text = input("Enter article:\n")

    result = pipeline.predict(text)

    print("\nPrediction:", result["label"])
    print(f"Meta Model: {result['confidence']:.4f}")

    print("\n--- Model Breakdown ---")
    print(f"Text Model:  {result['model_scores']['text_model']:.4f}")
    print(f"Style Model: {result['model_scores']['style_model']:.4f}")
    print(f"TF-IDF Model:{result['model_scores']['tfidf_model']:.4f}")

from explainability import full_explain, plot_lime, plot_style_shap, plot_meta_shap

explain = input("\nRun explainability? (y/n): ").strip().lower()
if explain == "y":
    print("\nRunning LIME + SHAP explainers (this may take ~10–20s)...")
    exp = full_explain(text, pipeline, verbose=True)

    save = input("Save plots? (y/n): ").strip().lower()
    if save == "y":
        plot_lime(exp,       save_path="lime_words.png")
        plot_style_shap(exp, save_path="style_shap.png")
        plot_meta_shap(exp,  save_path="meta_shap.png")
        print("Plots saved: lime_words.png, style_shap.png, meta_shap.png")
    else:
        plot_lime(exp)
        plot_style_shap(exp)
        plot_meta_shap(exp)