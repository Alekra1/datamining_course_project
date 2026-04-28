# Association rules: vegan status vs ultra-processing and nutrient flags

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("datasets/cleaned.csv", dtype={"code": str})
df = df.dropna(subset=["nova_group"])

salt_75 = df["salt_100g"].quantile(0.75)
sugar_75 = df["sugars_100g"].quantile(0.75)
fat_75 = df["saturated-fat_100g"].quantile(0.75)

binary = pd.DataFrame(
    {
        "is_vegan": df["is_vegan"].astype(bool),
        "nova_4": df["nova_group"] == 4,
        "high_salt": df["salt_100g"] > salt_75,
        "high_sugar": df["sugars_100g"] > sugar_75,
        "high_sat_fat": df["saturated-fat_100g"] > fat_75,
        "low_nutriscore": df["nutriscore_grade"].isin(["d", "e"]),
    }
)

frequent = apriori(binary, min_support=0.05, use_colnames=True)
rules = association_rules(
    frequent, num_itemsets=len(binary), metric="lift", min_threshold=0.5
)

# Keep only rules where is_vegan is the antecedent
rules_from_vegan = rules[
    rules["antecedents"].apply(lambda x: x == frozenset({"is_vegan"}))
]
rules_from_vegan = rules_from_vegan.sort_values("lift", ascending=False)

print("is_vegan → X rules:")
for _, row in rules_from_vegan.iterrows():
    con = list(row["consequents"])[0]
    pct = row["confidence"] * 100
    print(
        f"  is_vegan → {con:20s}  lift={row['lift']:.3f}  {pct:.1f}% of vegan products have this trait"
    )

# Bar chart of lift values
labels = [list(r["consequents"])[0] for _, r in rules_from_vegan.iterrows()]
lifts = rules_from_vegan["lift"].tolist()

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(labels, lifts, color="#e07b7b")
ax.axvline(
    1.0, color="black", linewidth=1, linestyle="--", label="lift = 1 (no association)"
)
ax.set_xlabel("Lift")
ax.set_title("Association of vegan status with health risk factors")
ax.legend()
plt.tight_layout()
plt.savefig("datasets/vegan_lift.png", dpi=150)
print("\nSaved chart to datasets/vegan_lift.png")
