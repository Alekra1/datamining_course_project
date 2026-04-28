# Trains a Decision Tree to predict whether a product is vegan from its nutrient profile.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("datasets/cleaned.csv", dtype={"code": str})
df["is_vegan"] = df["is_vegan"].astype(bool)

FEATURES = [
    "salt_100g",
    "sugars_100g",
    "saturated-fat_100g",
    "proteins_100g",
    "fiber_100g",
    "additives_n",
]

X = df[FEATURES]
y = df["is_vegan"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(
    ascending=False
)
print("\nFeature importances:")
print(importances.round(4).to_string())

# Tree visualisation
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    clf,
    feature_names=FEATURES,
    class_names=["non-vegan", "vegan"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
)
plt.tight_layout()
plt.savefig("datasets/tree.png", dpi=150)
print("\nSaved tree visualisation to datasets/tree.png")
