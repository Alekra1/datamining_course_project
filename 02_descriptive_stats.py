# Computes mean, median, std for chosen nutrients split by vegan status.
# Saves per-category stats to datasets/desc_stats_by_category.csv.
# Saves a bar chart of nutriscore difference to datasets/nutriscore_diff.png.

import matplotlib.pyplot as plt
import pandas as pd

NUTRIENTS = [
    "nutriscore_score",
    "sugars_100g",
    "salt_100g",
    "saturated-fat_100g",
    "additives_n",
]

df = pd.read_csv("datasets/cleaned.csv")
df["is_vegan"] = df["is_vegan"].astype(bool)

# --- Per-category counts ---
counts = (
    df.groupby(["pnns_groups_2", "is_vegan"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={False: "n_non_vegan", True: "n_vegan"})
)

# Per-category mean, median, std for each nutrient
rows = []
for nutrient in NUTRIENTS:
    grp = df.groupby(["pnns_groups_2", "is_vegan"])[nutrient].agg(
        ["mean", "median", "std"]
    )
    grp.columns = [f"{nutrient}_{c}" for c in grp.columns]
    rows.append(grp)

per_cat = pd.concat(rows, axis=1).round(3)
per_cat = counts.join(per_cat)

per_cat.to_csv("datasets/desc_stats_by_category.csv")
print("Saved per-category stats to datasets/desc_stats_by_category.csv")

# Bar chart for nutriscore difference per category
vegan_rows = per_cat.xs(True, level="is_vegan")
nonvegan_rows = per_cat.xs(False, level="is_vegan")

diff = vegan_rows["nutriscore_score_mean"] - nonvegan_rows["nutriscore_score_mean"]
diff = diff.sort_values()

bar_colors = ["green" if v < 0 else "red" for v in diff]

fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(diff.index, diff.values, color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(
    "Nutriscore difference (vegan minus non-vegan)\nNegative means vegan scores better"
)
ax.set_title("Nutriscore difference by food category")
plt.tight_layout()
plt.savefig("datasets/nutriscore_diff.png", dpi=150)
print("Saved chart to datasets/nutriscore_diff.png")

# Table for meat and dairy categories
Q2_CATS = ["Meat", "Processed meat", "Cheese", "Dairy desserts"]
Q2_NUTRIENTS = [
    ("salt_100g_mean", "Salt mean"),
    ("salt_100g_median", "Salt median"),
    ("sugars_100g_mean", "Sugars mean"),
    ("sugars_100g_median", "Sugars median"),
    ("saturated-fat_100g_mean", "Sat. fat mean"),
    ("saturated-fat_100g_median", "Sat. fat median"),
]

print("\nTable: meat and dairy categories (per 100g)\n")
print(f"{'Category':<20} {'Nutrient':<18} {'Vegan':>8} {'Non-vegan':>12}")
print("-" * 60)

for cat in Q2_CATS:
    first_row = True
    for col, label in Q2_NUTRIENTS:
        cat_label = cat if first_row else ""
        v = vegan_rows.loc[cat, col]
        nv = nonvegan_rows.loc[cat, col]
        print(f"{cat_label:<20} {label:<18} {v:>8.3f} {nv:>12.3f}")
        first_row = False
    print()
