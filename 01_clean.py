# Loads only the needed columns from the raw OpenFoodFacts CSV, derives a
# vegan flag from ingredients_analysis_tags, removes unusable and biologically
# impossible rows, and keeps only categories that have both vegan and non-vegan
# products. Saves a compact cleaned CSV ready for analysis.

import pandas as pd

RAW = "datasets/food_dataset.csv"
OUT = "datasets/cleaned.csv"

COLUMNS = [
    "code",
    "product_name",
    "ingredients_analysis_tags",
    # Standardised food category at a useful level of detail
    # (e.g. "Processed meat", "Dairy desserts", "Plant-based foods").
    # Used to make fair like-for-like comparisons (Q1) and to isolate
    # meat/dairy alternatives for Q2.
    "pnns_groups_2",
    # Numeric nutri-score (-15 to +40). Needed for statistical comparisons
    # (means, distributions) between vegan and non-vegan products (Q1).
    "nutriscore_score",
    # Letter grade version of the nutri-score (a–e). Easier to read in charts
    # and tables alongside the numeric score.
    "nutriscore_grade",
    # NOVA processing group (1–4). 4 = ultra-processed. Central to Q3:
    # are vegan products more likely to be ultra-processed?
    "nova_group",
    # Count of food additives (emulsifiers, preservatives, colourings, etc.).
    # A proxy for processing level that complements NOVA in Q3.
    "additives_n",
    # Energy density. A basic nutritional baseline useful for Q1 comparisons
    # and for contextualising the sugar/fat numbers in Q2.
    "energy-kcal_100g",
    # Total fat — one of the four nutri-score negative nutrients.
    # Relevant to Q1 (overall nutrition) and Q2 (fat in alternatives).
    "fat_100g",
    # Saturated fat — the nutri-score penalises this separately from total fat
    # because it is linked to cardiovascular risk.
    "saturated-fat_100g",
    # Sugar — central to Q2: do vegan alternatives add sugar to compensate
    # for missing animal-derived flavour?
    "sugars_100g",
    # Dietary fibre — a key positive nutrient in the nutri-score and a marker
    # of diet quality. Vegan diets are often associated with higher fibre;
    # including it lets us check whether processed vegan products keep that advantage.
    "fiber_100g",
    # Protein — important for Q2 because one common criticism of vegan
    # alternatives is lower or lower-quality protein.
    "proteins_100g",
    # Salt — the form in which sodium is reported on European packaging.
    # Included alongside sodium because some rows populate one but not the other.
    "salt_100g",
    # Sodium — the nutrient directly used in the nutri-score calculation and
    # central to Q2: do vegan alternatives add sodium for flavour?
    "sodium_100g",
]

# Nutrients that must be present and physically plausible (0–100 g per 100 g).
# Energy is checked separately with a wider bound (0–900 kcal per 100 g).
NUTRIENT_COLS = [
    "fat_100g",
    "saturated-fat_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
]

MIN_CATEGORY_SIZE = 30  # minimum products per vegan-status group per category

print("Loading selected columns...")
df = pd.read_csv(RAW, sep="\t", usecols=COLUMNS, low_memory=False)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# ── 1. Derive vegan flag ──────────────────────────────────────────────────────
# ingredients_analysis_tags contains comma-separated tags like
# "en:vegan,en:palm-oil-free" or "en:non-vegan".
tags = df["ingredients_analysis_tags"].fillna("")
df["is_vegan"] = tags.str.contains("en:vegan", na=False) & ~tags.str.contains(
    "en:non-vegan", na=False
)
df["is_non_vegan"] = tags.str.contains("en:non-vegan", na=False)

df = df[df["is_vegan"] | df["is_non_vegan"]].copy()
print(
    f"After vegan filter:          {len(df):,} rows  (vegan {df['is_vegan'].sum():,} | non-vegan {df['is_non_vegan'].sum():,})"
)

# ── 2. Require nutriscore_grade (Q1 needs it) ─────────────────────────────────
# Drop nulls, "unknown" (score couldn't be computed), and "not-applicable"
# (water, baby food, etc. that are exempt from nutriscore by regulation).
df["nutriscore_grade"] = df["nutriscore_grade"].str.strip().str.lower()
df = df[
    df["nutriscore_grade"].notna()
    & ~df["nutriscore_grade"].isin(["unknown", "not-applicable"])
].copy()
print(f"After nutriscore filter:     {len(df):,} rows")

# ── 3. Require pnns_groups_2 (fair category comparisons need it) ──────────────
# Drop nulls, empty strings, AND the literal value "unknown" — OpenFoodFacts
# uses "unknown" as a placeholder when the category could not be determined.
df = df[
    df["pnns_groups_2"].notna()
    & (df["pnns_groups_2"].str.strip() != "")
    & (df["pnns_groups_2"].str.lower() != "unknown")
].copy()
print(f"After category filter:       {len(df):,} rows")

# ── 4. Coerce nutrient columns to numeric (CSV stores everything as strings) ──
for col in NUTRIENT_COLS + ["energy-kcal_100g"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── 5. Require key nutrients (Q2 analysis needs them) ────────────────────────
df = df[df[NUTRIENT_COLS].notna().all(axis=1)].copy()
print(f"After nutrient filter:       {len(df):,} rows")

# ── 6. Remove biologically impossible values ──────────────────────────────────
# Each nutrient is per 100 g of food, so it cannot exceed 100 g or go below 0.
# Energy cannot exceed 900 kcal/100 g (pure fat ≈ 900 kcal).
nutrient_mask = (df[NUTRIENT_COLS] >= 0).all(axis=1) & (df[NUTRIENT_COLS] <= 100).all(
    axis=1
)
df = df[nutrient_mask].copy()
print(f"After nutrient range filter: {len(df):,} rows")

energy_mask = df["energy-kcal_100g"].isna() | df["energy-kcal_100g"].between(0, 900)
df = df[energy_mask].copy()
print(f"After energy range filter:   {len(df):,} rows")

# ── 7. Normalise remaining columns ────────────────────────────────────────────
df["nova_group"] = pd.to_numeric(df["nova_group"], errors="coerce")

# ── 8. Keep only categories with both groups (≥ MIN_CATEGORY_SIZE each) ───────
# Q1 compares vegan vs non-vegan within the same category. Categories that only
# have one group make that comparison impossible; tiny groups produce noisy stats.
counts = df.groupby(["pnns_groups_2", "is_vegan"]).size().unstack(fill_value=0)
# unstack gives columns True and False (vegan / non-vegan)
valid_cats = counts[
    (counts.get(True, 0) >= MIN_CATEGORY_SIZE)
    & (counts.get(False, 0) >= MIN_CATEGORY_SIZE)
].index
df = df[df["pnns_groups_2"].isin(valid_cats)].copy()
print(
    f"After category balance filter:{len(df):,} rows  ({len(valid_cats)} categories kept)"
)

print(f"\nFinal dataset: {len(df):,} rows")
print(
    df[["is_vegan", "nutriscore_grade", "nova_group"]]
    .describe(include="all")
    .to_string()
)

df.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")
