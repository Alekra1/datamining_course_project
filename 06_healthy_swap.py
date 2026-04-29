# Compares all vegan products against only "healthy" non-vegan products,
# defined by UK FSA traffic-light thresholds and NOVA group <= 2.

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("datasets/cleaned.csv")
df["is_vegan"] = df["is_vegan"].astype(bool)
df = df.dropna(subset=["nova_group"])

# UK FSA amber thresholds (per 100g) + NOVA filter
df["is_healthy"] = (
    (df["saturated-fat_100g"] <= 5)
    & (df["sugars_100g"] <= 22.5)
    & (df["salt_100g"] <= 1.5)
    & (df["nova_group"] <= 2)
)

vegan = df[df["is_vegan"]]
nonvegan_all = df[~df["is_vegan"]]
nonvegan_healthy = df[(~df["is_vegan"]) & df["is_healthy"]]

print(f"All vegan products:                 {len(vegan):>7,}")
print(f"All non-vegan products:             {len(nonvegan_all):>7,}")
print(f"Healthy non-vegan products:         {len(nonvegan_healthy):>7,}")
print(
    f"Share of non-vegan that is healthy: {len(nonvegan_healthy) / len(nonvegan_all):>7.1%}"
)
print(
    f"Share of vegan that is healthy:     {df[df['is_vegan'] & df['is_healthy']].shape[0] / len(vegan):>7.1%}"
)
print()

subset = pd.concat([vegan, nonvegan_healthy])

counts = subset.groupby(["pnns_groups_2", "is_vegan"]).size().unstack(fill_value=0)
valid_cats = counts[(counts.get(True, 0) >= 30) & (counts.get(False, 0) >= 30)].index
subset = subset[subset["pnns_groups_2"].isin(valid_cats)]

print(f"Categories with >=30 products on each side: {len(valid_cats)}")
print()

metrics = ["nutriscore_score", "saturated-fat_100g", "sugars_100g", "salt_100g"]

# Bar chart: nutriscore difference per category
nutri = (
    subset.groupby(["pnns_groups_2", "is_vegan"])["nutriscore_score"].mean().unstack()
)
diff = (nutri[True] - nutri[False]).sort_values()

bar_colors = ["green" if v < 0 else "red" for v in diff]

fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(diff.index, diff.values, color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(
    "Nutriscore difference (vegan minus healthy non-vegan)\nNegative means vegan scores better"
)
ax.set_title("Vegan vs healthy non-vegan: nutriscore difference by category")
plt.tight_layout()
plt.savefig("datasets/healthy_swap_diff.png", dpi=150)
print("Saved chart to datasets/healthy_swap_diff.png")

# Comparison table
print("\nPer-category mean nutriscore (lower = healthier):\n")
print(f"{'Category':<25} {'Vegan':>8} {'Healthy NV':>12} {'Diff':>8}")
print("-" * 55)
for cat in diff.index:
    v = nutri.loc[cat, True]
    nv = nutri.loc[cat, False]
    d = v - nv
    print(f"{cat:<25} {v:>8.2f} {nv:>12.2f} {d:>+8.2f}")
