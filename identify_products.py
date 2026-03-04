"""
I'm using the instacart dataset.
Use product/aisle/department names to identify the following types of products:
- diabetes-management
- low-sodium/hypertension-friendly
- GI sensitivity / gluten-free 
- Lactose-free / lactase enzyme 
- Alcohol purchasing intensity 
- Ultra-processed / sugary beverage heavy

Ultra-processed foods include carbonated soft drinks; sweet or savoury packaged snacks; chocolate, candies (confectionery); ice cream; mass-produced packaged breads and buns; margarines and other spreads; cookies (biscuits), pastries, cakes and cake mixes; breakfast ‘cereals’; pre-prepared pies and pasta and pizza dishes; poultry and fish ‘nuggets’ and ‘sticks’, sausages, burgers, hot dogs and other reconstituted meat products; powdered and packaged ‘instant’ soups, noodles and desserts; and many other products.

Create a boolean column for each product type. 
"""

#!/usr/bin/env python3
"""
Instacart product-type flags (boolean columns) based on product/aisle/department names.

Inputs (standard Instacart schema):
  - ./data/raw/products.csv:        product_id, product_name, aisle_id, department_id
  - ./data/raw/aisles.csv:          aisle_id, aisle
  - ./data/raw/departments.csv:     department_id, department

Output:
  - ./data/processed/products_flagged.csv: product_id + joined names + boolean flags

Notes:
- This is a *heuristic keyword* approach. Expect false positives/negatives.
- You can tighten/expand keyword lists per your needs.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def compile_patterns(keywords):
    """Compile keywords (strings or regex fragments) into a single case-insensitive regex."""
    # Escape plain words unless they already look like regex (contain regex metacharacters)
    escaped = []
    for k in keywords:
        if any(ch in k for ch in r".^$*+?{}[]\|()"):
            escaped.append(k)
        else:
            escaped.append(re.escape(k))
    return re.compile(r"(?i)\b(?:%s)\b" % "|".join(escaped))


def normalize_text(s: pd.Series) -> pd.Series:
    # Lowercase, remove punctuation-ish clutter, collapse whitespace
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^a-z0-9\s/&+-]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def make_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must include:
      product_id, product_name, aisle, department
    """
    # Create a combined text field for matching
    text = normalize_text(
        df["product_name"].astype(str) + " " + df["aisle"].astype(str) + " " + df["department"].astype(str)
    )

    # ---------------------------
    # Keyword sets (edit freely)
    # ---------------------------

    # 1) Diabetes-management (non-tautological-ish: supplies, glucose tabs, “diabetic” labeled, sugar-free sweeteners)
    diabetes_kw = [
        # direct labeling / common terms
        "diabetic", "diabetes", "blood sugar", "glucose",
        # diabetes supplies / treatment-adjacent products present in grocery catalogs
        "glucose tablets", "glucose gel", "glucose shot", "dextrose", "meter", "test strips",
        # sugar alternatives often used by diabetics
        "sugar free", "no sugar", "zero sugar", "unsweetened",
        "stevia", "erythritol", "xylitol", "monk fruit", "sucralose", "aspartame",
        # low-carb / keto labeling sometimes used for glucose management
        "low carb", "keto", "ketogenic", "net carb", "carb smart",
    ]

    # 2) Low-sodium / hypertension-friendly
    lowsodium_kw = [
        "low sodium", "reduced sodium", "no salt", "salt free", "unsalted",
        "lightly salted", "heart healthy", "cardio", "blood pressure",
        # common substitutes / ingredients
        "potassium salt", "salt substitute",
    ]

    # 3) GI sensitivity / gluten-free
    glutenfree_kw = [
        "gluten free", "gluten-free", "gf ",
        "celiac", "coeliac", "wheat free", "wheat-free",
        "grain free", "grain-free",
        # low-FODMAP / IBS-adjacent labels sometimes appear
        "low fodmap", "fodmap", "ibs",
    ]

    # 4) Lactose-free / lactase enzyme
    lactosefree_kw = [
        "lactose free", "lactose-free",
        "lactaid", "lactase",
        "dairy free", "dairy-free",
        # common lactose-free dairy variants
        "ultra filtered", "ultrafiltered",
        # enzyme / pills (some catalogs include)
        "digestive enzyme", "enzyme",
    ]
    # We'll try to avoid flagging every non-dairy milk as lactose-free if you prefer:
    # You can comment out these if too broad, but many lactose-intolerant users buy these.
    lactosefree_broad_addons = [
        "almond milk", "oat milk", "soy milk", "coconut milk", "rice milk",
        "non dairy", "nondairy",
        "plant based", "plant-based", "vegan",
    ]

    # 5) Alcohol purchasing intensity
    alcohol_kw = [
        "beer", "ale", "lager", "ipa", "stout", "porter",
        "wine", "champagne", "prosecco", "sparkling wine", "rose",
        "vodka", "whiskey", "whisky", "bourbon", "scotch", "rum", "gin", "tequila", "brandy",
        "hard seltzer", "cider", "mead",
        "liqueur", "cordial", "vermouth",
        "cocktail", "margarita", "mojito",
        # mixers can be alcohol-adjacent; include only if you want broader intensity
        # "bitters", "simple syrup", "tonic", "club soda",
    ]

    # 6) Ultra-processed / sugary beverage heavy (as per your list; plus common sugary drinks)
    ultraprocessed_kw = [
        # carbonated soft drinks & sugary beverages
        "soda", "soft drink", "cola", "pop", "root beer", "ginger ale",
        "lemonade", "fruit punch", "sweet tea", "iced tea", "energy drink", "sports drink",
        "juice drink", "nectar", "sweetened", "sugar added",
        # snacks, sweets, confectionery, ice cream
        "chips", "crisps", "pretzels", "snack", "snack mix",
        "candy", "candies", "chocolate", "confectionery", "gummies", "gum", "lollipop",
        "ice cream", "frozen dessert", "sherbet", "gelato",
        # packaged breads/buns/cookies/cakes/pastries/cereals
        "white bread", "buns", "rolls", "white tortillas",
        "cookies", "biscuits", "crackers",
        "pastry", "pastries", "donut", "doughnut", "cake", "brownie", "cupcake", "muffin", "cake mix",
        "cereal", "breakfast cereal", "granola bar", "snack bar",
        # spreads
        "margarine", "frosting", "mayonnaise", "ketchup", "bbq sauce", "ranch dressing", "salad dressing",
        # pre-prepared pies/pasta/pizza dishes; instant soups/noodles/desserts
        "pizza", "frozen pizza", "pasta meal", "microwave meal", "ready meal", "prepared meal",
        "frozen pie", "pot pie", "instant soup", "cup soup", "ramen", "instant noodles",
        "instant dessert", "pudding mix",
        # nuggets/sticks/sausages/hot dogs/burgers (reconstituted meat products)
        "nuggets", "fish sticks", "chicken sticks",
        "hot dog", "hotdogs", # "sausage", "burger",  # too broad if you want to avoid flagging all sausages/burgers
    ]

    # ---------------------------
    # Compile regex patterns
    # ---------------------------
    diabetes_re = compile_patterns(diabetes_kw)
    lowsodium_re = compile_patterns(lowsodium_kw)
    glutenfree_re = compile_patterns(glutenfree_kw)
    lactose_re = compile_patterns(lactosefree_kw + lactosefree_broad_addons)
    alcohol_re = compile_patterns(alcohol_kw)
    ultraprocessed_re = compile_patterns(ultraprocessed_kw)

    # ---------------------------
    # Apply flags
    # ---------------------------
    out = df.copy()

    out["is_diabetes_management"] = text.str.contains(diabetes_re)
    out["is_low_sodium_hypertension_friendly"] = text.str.contains(lowsodium_re)
    out["is_gi_sensitivity_gluten_free"] = text.str.contains(glutenfree_re)
    out["is_lactose_free_or_lactase"] = text.str.contains(lactose_re)
    out["is_alcohol"] = text.str.contains(alcohol_re)
    out["is_ultra_processed_or_sugary_beverage_heavy"] = text.str.contains(ultraprocessed_re)

    # Optional: make alcohol detection more conservative by requiring department/aisle hints
    # Uncomment if your catalog has too many false positives from e.g., "ginger ale" non-alcoholic.
    dept_aisle = normalize_text(out["aisle"].astype(str) + " " + out["department"].astype(str))
    alcohol_context = dept_aisle.str.contains(compile_patterns(["alcohol", "beer", "wine", "spirits", "liquor"]))
    out["is_alcohol"] = out["is_alcohol"] & (alcohol_context | out["product_name"].str.lower().str.contains(r"\b(vodka|whisk(e)?y|tequila|gin|rum)\b"))

    # Hard exclusions: departments that can never contain ultra-processed food
    NON_FOOD_DEPTS = {"household", "personal care", "pets", "babies", "missing"}
    out.loc[out["department"].isin(NON_FOOD_DEPTS), "is_ultra_processed_or_sugary_beverage_heavy"] = False

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data/raw", help="Directory containing products.csv, aisles.csv, departments.csv")
    ap.add_argument("--out", type=str, default="./data/processed/products_flagged.csv", help="Output CSV path")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    products = pd.read_csv(data_dir / "products.csv")
    aisles = pd.read_csv(data_dir / "aisles.csv")
    depts = pd.read_csv(data_dir / "departments.csv")

    # Standardize column names expected by Instacart
    # products: product_id, product_name, aisle_id, department_id
    # aisles: aisle_id, aisle
    # departments: department_id, department
    df = (
        products.merge(aisles, on="aisle_id", how="left")
                .merge(depts, on="department_id", how="left")
    )

    flagged = make_flags(df)

    # Keep a tidy set of columns (you can keep all if you want)
    keep_cols = [
        "product_id", "product_name", "aisle_id", "aisle", "department_id", "department",
        "is_diabetes_management",
        "is_low_sodium_hypertension_friendly",
        "is_gi_sensitivity_gluten_free",
        "is_lactose_free_or_lactase",
        "is_alcohol",
        "is_ultra_processed_or_sugary_beverage_heavy",
    ]
    for c in keep_cols:
        if c not in flagged.columns:
            raise ValueError(f"Missing expected column: {c}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flagged[keep_cols].to_csv(out_path, index=False)
    print(f"Wrote: {out_path}  (rows={len(flagged):,})")
    # Report some flagging stats
    print("Flagging summary:")
    for col in keep_cols[6:]:
        count = flagged[col].sum()
        print(f"  {col}: {count} products ({count/len(flagged):.2%})")

"""
49,688 products total in Instacart catalog (after merging names)
Flagging summary:
  is_diabetes_management: 548 products (1.10%)
  is_low_sodium_hypertension_friendly: 391 products (0.79%)
  is_gi_sensitivity_gluten_free: 893 products (1.80%)
  is_lactose_free_or_lactase: 811 products (1.63%)
  is_alcohol: 643 products (1.29%)
  is_ultra_processed_or_sugary_beverage_heavy: 10652 products (21.44%)
"""

if __name__ == "__main__":
    main()