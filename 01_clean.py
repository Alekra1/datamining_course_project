# Loads relevant columns from dataset, cleans it,
# and keeps only categories that have both vegan and non-vegan
# products. Saves a clean CSV.

import pandas as pd

RAW = "datasets/food_dataset.csv"
OUT = "datasets/cleaned.csv"

COLUMNS = [
    "code",
    "product_name",
    "ingredients_analysis_tags",
    "pnns_groups_2",
    "nutriscore_score",
    "nutriscore_grade",
    "nova_group",
    "additives_n",
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
]

NUTRIENT_COLS = [
    "fat_100g",
    "saturated-fat_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
]

INVALID_GRADES = {"unknown", "not-applicable"}

MIN_CATEGORY_SIZE = 30

print("Loading selected columns...")
df = pd.read_csv(RAW, sep="\t", usecols=COLUMNS, low_memory=False)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")


tags = df["ingredients_analysis_tags"].fillna("")
df["is_vegan"] = tags.str.contains("en:vegan", na=False) & ~tags.str.contains(
    "en:non-vegan", na=False
)
is_non_vegan = tags.str.contains("en:non-vegan", na=False)

df["nutriscore_grade"] = df["nutriscore_grade"].str.strip().str.lower()

df["pnns_groups_2"] = df["pnns_groups_2"].str.strip()

numeric_cols = NUTRIENT_COLS + ["energy-kcal_100g", "nova_group"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Filter

# Keeping only rows where vegan status is known
df = df[df["is_vegan"] | is_non_vegan]
print(
    f"After vegan filter:          {len(df):,} rows  (vegan {df['is_vegan'].sum():,} | non-vegan {(~df['is_vegan']).sum():,})"
)

# Require a real nutriscore_grade.
df = df[df["nutriscore_grade"].notna() & ~df["nutriscore_grade"].isin(INVALID_GRADES)]
print(f"After nutriscore filter:     {len(df):,} rows")

# Require a known food category
df = df[
    df["pnns_groups_2"].notna() & ~df["pnns_groups_2"].str.lower().isin({"", "unknown"})
]
print(f"After category filter:       {len(df):,} rows")

# Require all key nutrients to be present
df = df[df[NUTRIENT_COLS].notna().all(axis=1)]
print(f"After nutrient filter:       {len(df):,} rows")

# Remove biologically impossible values.
nutrient_ok = (df[NUTRIENT_COLS] >= 0).all(axis=1) & (df[NUTRIENT_COLS] <= 100).all(
    axis=1
)
energy_ok = df["energy-kcal_100g"].isna() | df["energy-kcal_100g"].between(0, 900)
df = df[nutrient_ok & energy_ok]
print(f"After impossible value filter:{len(df):,} rows")

# Keep only categories with both groups
counts = df.groupby(["pnns_groups_2", "is_vegan"]).size().unstack(fill_value=0)
valid_cats = counts[
    (counts.get(True, 0) >= MIN_CATEGORY_SIZE)
    & (counts.get(False, 0) >= MIN_CATEGORY_SIZE)
].index
df = df[df["pnns_groups_2"].isin(valid_cats)]
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
