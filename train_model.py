#!/usr/bin/env python3
"""Reproduit l'évaluation et l'export du modèle EVON AFRO."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "data.csv")
X = df.drop(columns=["ID", "Segmentation"])
y = df["Segmentation"]
classes = ["A", "B", "C", "D"]
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
pre = ColumnTransformer([
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ("num", SimpleImputer(strategy="median"), num_cols),
])
pipe = Pipeline([("preprocessor", pre), ("classifier", GradientBoostingClassifier(random_state=42))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pred = cross_val_predict(pipe, X, y, cv=cv)
metrics = {
    "accuracy": float(accuracy_score(y, pred)),
    "macro_f1": float(f1_score(y, pred, average="macro")),
    "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    "baseline_majority": float(y.value_counts(normalize=True).max()),
    "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
    "classification_report": classification_report(y, pred, labels=classes, output_dict=True, zero_division=0),
}
pipe.fit(X, y)
pre = pipe.named_steps["preprocessor"]
clf = pipe.named_steps["classifier"]
cat_pipe = pre.named_transformers_["cat"]
encoder = cat_pipe.named_steps["onehot"]
cat_imputer = cat_pipe.named_steps["imputer"]
num_imputer = pre.named_transformers_["num"]
log_prior = np.log(clf.init_.class_prior_)
trees = []
for stage in clf.estimators_:
    row = []
    for estimator in stage:
        tree = estimator.tree_
        row.append({"f": tree.feature.astype(int).tolist(), "th": tree.threshold.tolist(),
                    "l": tree.children_left.astype(int).tolist(), "r": tree.children_right.astype(int).tolist(),
                    "v": tree.value[:, 0, 0].tolist()})
    trees.append(row)
payload = {
    "classes": clf.classes_.tolist(), "catColumns": cat_cols, "numColumns": num_cols,
    "categories": [a.tolist() for a in encoder.categories_], "catImpute": cat_imputer.statistics_.tolist(),
    "numImpute": num_imputer.statistics_.tolist(), "initial": (log_prior-log_prior.mean()).tolist(),
    "learningRate": clf.learning_rate, "trees": trees, "cv": metrics
}
model_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
predictor = r"""
window.EVON_MODEL.predict = function(input) {
  const m = window.EVON_MODEL;
  const vector = [];
  m.catColumns.forEach((col, i) => {
    const value = input[col] == null || input[col] === '' ? m.catImpute[i] : String(input[col]);
    m.categories[i].forEach(category => vector.push(value === category ? 1 : 0));
  });
  m.numColumns.forEach((col, i) => {
    const raw = input[col];
    const value = raw == null || raw === '' || !Number.isFinite(Number(raw)) ? m.numImpute[i] : Number(raw);
    vector.push(value);
  });
  const scores = m.initial.slice();
  m.trees.forEach(stage => stage.forEach((tree, classIndex) => {
    let node = 0;
    while (tree.l[node] !== -1) {
      const feature = tree.f[node];
      node = vector[feature] <= tree.th[node] ? tree.l[node] : tree.r[node];
    }
    scores[classIndex] += m.learningRate * tree.v[node];
  }));
  const maxScore = Math.max(...scores);
  const exps = scores.map(value => Math.exp(value - maxScore));
  const total = exps.reduce((sum, value) => sum + value, 0);
  const probabilities = {};
  m.classes.forEach((label, i) => probabilities[label] = exps[i] / total);
  const best = m.classes.reduce((a, b) => probabilities[b] > probabilities[a] ? b : a, m.classes[0]);
  return { segment: best, probabilities, confidence: probabilities[best] };
};
"""
with open(BASE_DIR / "model.js", "w", encoding="utf-8") as f:
    f.write("window.EVON_MODEL = " + model_json + ";\n" + predictor)
print(json.dumps({k: metrics[k] for k in ["accuracy", "macro_f1", "balanced_accuracy", "baseline_majority"]}, indent=2))
