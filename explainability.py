# EXPLAINABILITY MODULE — LIME + SHAP
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Try to set a Hindi-supporting font if available
plt.rcParams['font.family'] = 'Nirmala UI'

STYLE_FEATURE_NAMES = [
    "word_count", "sentence_count", "avg_sentence_len",
    "exclamations", "questions", "punctuation",
    "uppercase_ratio", "type_token_ratio", "stopword_ratio",
]
META_FEATURE_NAMES = ["text_model_score", "style_model_score", "tfidf_model_score"]


# 1. LIME — TF-IDF Model
def explain_tfidf_lime(text, tfidf_vectorizer, tfidf_clf, num_features=15, num_samples=500):
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        raise ImportError("Run: pip install lime")

    def tfidf_predict_proba(texts):
        return tfidf_clf.predict_proba(tfidf_vectorizer.transform(texts))

    explainer = LimeTextExplainer(class_names=["REAL", "FAKE"], split_expression=r'\s+', bow=True, random_state=42)
    exp = explainer.explain_instance(text, tfidf_predict_proba, num_features=num_features, num_samples=num_samples, top_labels=2)
    # Use whichever label LIME actually computed — prefer FAKE (1) if available
    top_label = 1 if 1 in exp.local_exp else exp.top_labels[0]
    top_words = sorted(exp.as_list(label=top_label), key=lambda x: abs(x[1]), reverse=True)
    return exp, top_words


# 1.5 LIME — Semantic Model (SentenceTransformer + RandomForest)
def explain_semantic_lime(text, embedder, text_clf, num_features=15, num_samples=200):
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        raise ImportError("Run: pip install lime")

    def semantic_predict_proba(texts):
        # show_progress_bar=False to keep console clean during LIME iterations
        embeddings = embedder.encode(texts, show_progress_bar=False)
        return text_clf.predict_proba(embeddings)

    explainer = LimeTextExplainer(class_names=["REAL", "FAKE"], split_expression=r'\s+', bow=True, random_state=42)
    # num_samples is lower because Transformer encoding is computationally expensive
    exp = explainer.explain_instance(text, semantic_predict_proba, num_features=num_features, num_samples=num_samples, top_labels=2)
    
    top_label = 1 if 1 in exp.local_exp else exp.top_labels[0]
    top_words = sorted(exp.as_list(label=top_label), key=lambda x: abs(x[1]), reverse=True)
    return exp, top_words


# 2. SHAP — Style Model (RandomForest → TreeExplainer)
def explain_style_shap(text, pipeline):
    try:
        import shap
    except ImportError:
        raise ImportError("Run: pip install shap")

    style_vec = np.array(pipeline.extract_style_features(text))  # shape (1, 9)

    explainer = shap.TreeExplainer(pipeline.style_clf)
    shap_values = explainer.shap_values(style_vec)

    # shap_values is list [class0(1,9), class1(1,9)] — take FAKE class, flatten to (9,)
    if isinstance(shap_values, list):
        shap_vals = np.array(shap_values[1]).flatten()
    else:
        shap_vals = np.array(shap_values).flatten()

    return shap_vals, STYLE_FEATURE_NAMES, style_vec.flatten()


# 3. SHAP — Meta Model (LogisticRegression)
def explain_meta_shap(text_prob, style_prob, tfidf_prob, meta_model, scaler):
    meta_input_raw = np.array([[text_prob, style_prob, tfidf_prob]])
    meta_input_scaled = scaler.transform(meta_input_raw).flatten()

    # For logistic regression: shap ≈ coef * feature_value (exact for linear models)
    coef = meta_model.coef_[0]                  # shape (3,)
    shap_vals = coef * meta_input_scaled         # element-wise, shape (3,)

    return shap_vals, META_FEATURE_NAMES, meta_input_raw.flatten()


# 4. FULL EXPLAIN
def full_explain(text, pipeline, lime_num_features=15, verbose=True):
    text_lower = text.lower()
    result = pipeline.predict(text)
    tp = result["model_scores"]["text_model"]
    sp = result["model_scores"]["style_model"]
    fp = result["model_scores"]["tfidf_model"]

    # 1. TF-IDF LIME
    tfidf_lime_exp, tfidf_top_words = explain_tfidf_lime(text_lower, pipeline.tfidf_vectorizer, pipeline.tfidf_clf, lime_num_features)
    
    # 2. Semantic LIME
    semantic_lime_exp, semantic_top_words = explain_semantic_lime(text_lower, pipeline.embedder, pipeline.text_clf, lime_num_features)
    
    # 3. Style SHAP
    style_shap, style_names, style_vals = explain_style_shap(text_lower, pipeline)
    
    # 4. Meta SHAP
    meta_shap, meta_names, meta_vals = explain_meta_shap(tp, sp, fp, pipeline.meta_model, pipeline.scaler)

    explanation = {
        "prediction": result,
        "tfidf_lime": {"exp": tfidf_lime_exp, "top_words": tfidf_top_words},
        "semantic_lime": {"exp": semantic_lime_exp, "top_words": semantic_top_words},
        "style": {"shap_values": style_shap, "feature_names": style_names, "feature_values": style_vals},
        "meta":  {"shap_values": meta_shap,  "feature_names": meta_names,  "feature_values": meta_vals},
    }

    if verbose:
        _print_explanation(explanation)
    return explanation


# ══════════════════════════════════════════════════════════════
# 5. CONSOLE PRINTER
# ══════════════════════════════════════════════════════════════
def _print_explanation(exp):
    r = exp["prediction"]
    print("\n" + "═" * 60)
    print(f"  VERDICT : {r['label']}  (confidence: {r['confidence']:.4f})")
    print("═" * 60)

    print("\n📊 META MODEL — Sub-model Contributions (SHAP)")
    for name, val, sv in zip(exp["meta"]["feature_names"], exp["meta"]["feature_values"], exp["meta"]["shap_values"]):
        print(f"  {name:<22} score={float(val):.4f}  shap={float(sv):+.4f}  {'→FAKE' if sv > 0 else '→REAL'}")

    print("\n🧠 SEMANTIC MODEL — Deep Meaning Clues (LIME)")
    for word, weight in exp["semantic_lime"]["top_words"][:10]:
        print(f"  {word:<25} {weight:+.4f}  {'→FAKE' if weight > 0 else '→REAL'}")

    print("\n✍️  STYLE MODEL — Writing Feature Contributions (SHAP)")
    pairs = sorted(
        zip(exp["style"]["feature_names"], exp["style"]["feature_values"], exp["style"]["shap_values"]),
        key=lambda x: abs(float(x[2])), reverse=True
    )
    for name, val, sv in pairs:
        print(f"  {name:<22} val={float(val):.4f}  shap={float(sv):+.4f}  {'→FAKE' if sv > 0 else '→REAL'}")

    print("\n🔤 TF-IDF MODEL — Top Influential Words (LIME)")
    for word, weight in exp["tfidf_lime"]["top_words"][:10]:
        print(f"  {word:<25} {weight:+.4f}  {'→FAKE' if weight > 0 else '→REAL'}")
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════
# 6. PLOT HELPERS
# ══════════════════════════════════════════════════════════════
def plot_tfidf_lime(explanation, save_path=None):
    import matplotlib.pyplot as plt
    words  = [w for w, _ in explanation["tfidf_lime"]["top_words"]]
    scores = [s for _, s in explanation["tfidf_lime"]["top_words"]]
    colors = ["#e74c3c" if s > 0 else "#2ecc71" for s in scores]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(words[::-1], scores[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME Weight  (positive → FAKE, negative → REAL)")
    ax.set_title("TF-IDF Model — Top Influential Words (LIME)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150) if save_path else plt.show()
    plt.close()

def plot_semantic_lime(explanation, save_path=None):
    import matplotlib.pyplot as plt
    words  = [w for w, _ in explanation["semantic_lime"]["top_words"]]
    scores = [s for _, s in explanation["semantic_lime"]["top_words"]]
    colors = ["#e74c3c" if s > 0 else "#2ecc71" for s in scores]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(words[::-1], scores[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME Weight  (positive → FAKE, negative → REAL)")
    ax.set_title("Semantic Model — Deep Meaning Clues (LIME)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150) if save_path else plt.show()
    plt.close()

def plot_style_shap(explanation, save_path=None):
    import matplotlib.pyplot as plt
    pairs = sorted(
        zip(explanation["style"]["feature_names"], explanation["style"]["shap_values"], explanation["style"]["feature_values"]),
        key=lambda x: abs(float(x[1])), reverse=True
    )
    names  = [f"{n}\n(val={float(v):.2f})" for n, _, v in pairs]
    shaps  = [float(s) for _, s, _ in pairs]
    colors = ["#e74c3c" if s > 0 else "#2ecc71" for s in shaps]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names[::-1], shaps[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value  (positive → FAKE, negative → REAL)")
    ax.set_title("Style Model — Feature Contributions (SHAP TreeExplainer)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150) if save_path else plt.show()
    plt.close()

def plot_meta_shap(explanation, save_path=None):
    import matplotlib.pyplot as plt
    names  = explanation["meta"]["feature_names"]
    shaps  = [float(s) for s in explanation["meta"]["shap_values"]]
    vals   = [float(v) for v in explanation["meta"]["feature_values"]]
    colors = ["#e74c3c" if s > 0 else "#2ecc71" for s in shaps]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh([f"{n}\n(score={v:.3f})" for n, v in zip(names, vals)], shaps, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value  (positive → FAKE, negative → REAL)")
    ax.set_title("Meta Model — Sub-model Contributions (SHAP)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150) if save_path else plt.show()
    plt.close()
