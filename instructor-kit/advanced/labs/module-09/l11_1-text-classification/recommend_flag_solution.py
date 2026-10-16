# --- lab bootstrap: find labs_paths.py (the same three lines open every lab; see labs/README.md) ---
import pathlib, sys
sys.path[:0] = [str(p) for p in pathlib.Path(globals().get("__file__", "lab.py")).resolve().parents if (p / "labs_paths.py").is_file()]
from labs_paths import DATA

"""Lab 11.1 A (SOLUTION): would this customer recommend the product?

Every review form ends with a checkbox, "I would recommend this product" (the `recommended` column).
The marketing team wants to read that intent from text that arrives WITHOUT the checkbox (emails,
chat). A new target on the same reviews, including the mixed 3-star ones the lesson left out: build
the TF-IDF pipeline, compare three vectorizer settings with cross-validation, read the words, probe it.
"""
# %% Step 1: target, split, baseline
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

reviews = pd.read_csv(DATA / "reviews.csv")
print(reviews.shape, f"recommended share {reviews['recommended'].mean():.3f}")
print("recommended share by star rating:", reviews.groupby("rating")["recommended"].mean().round(3).to_dict())

train, test = train_test_split(reviews, test_size=0.25, stratify=reviews["recommended"], random_state=42)
baseline = DummyClassifier(strategy="most_frequent").fit(train["text"], train["recommended"])
print(len(train), "train,", len(test), "test;", f"baseline accuracy {baseline.score(test['text'], test['recommended']):.3f}")

# %% Step 2: three vectorizer settings, compared with cross-validation on the training reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
settings = {
    "unigrams": TfidfVectorizer(),
    "unigrams, stop words removed": TfidfVectorizer(stop_words="english"),
    "unigrams + bigrams, min_df=2": TfidfVectorizer(ngram_range=(1, 2), min_df=2),
}
for name, vectorizer in settings.items():
    scores = cross_val_score(make_pipeline(vectorizer, LogisticRegression()), train["text"], train["recommended"], cv=cv)
    print(f"{name:30s} CV accuracy {scores.mean():.3f} ± {scores.std():.3f}")

# %% Step 3: fit the bigram model, test once, read the "not recommended" row
from sklearn.metrics import classification_report

model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression())
model.fit(train["text"], train["recommended"])
pred = model.predict(test["text"])
print(classification_report(test["recommended"], pred, target_names=["not recommended", "recommended"], digits=3))

# %% Step 4: the words with the strongest pull
vec = model.named_steps["tfidfvectorizer"]
weights = pd.Series(model.named_steps["logisticregression"].coef_[0], index=vec.get_feature_names_out()).sort_values()
print("toward NOT recommended:", weights.head(6).round(2).to_dict())
print("toward recommended    :", weights.tail(6).round(2).to_dict())

# %% Step 5: where does the model struggle? Accuracy by star rating
test = test.assign(pred=pred)
by_star = test.groupby("rating").apply(lambda g: (g["pred"] == g["recommended"]).mean(), include_groups=False)
print("test accuracy by star rating:", by_star.round(3).to_dict())

# %% Step 6: probe it with sentences of your own
probes = ["Would buy again.", "Would not buy again.", "Not bad at all, but it runs small.",
          "It leaked on the first night.", "Great value. Though the zipper broke."]
for text, p in zip(probes, model.predict_proba(probes)[:, 1]):
    print(f"{text:38s} P(recommended) {p:.2f}")
