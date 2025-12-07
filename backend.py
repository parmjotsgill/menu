import re
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# 1. Load and clean menu data
# -------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "Menus.xlsx"

dfs_menus: Dict[str, pd.DataFrame] = {}
all_menus_df = pd.DataFrame()


def _load_real_menus() -> pd.DataFrame:
    """
    Try to load the real Menus.xlsx file.
    Returns an empty DataFrame if anything goes wrong.
    """
    if not DATA_PATH.exists():
        print(f"[backend] Menus.xlsx not found at {DATA_PATH}")
        return pd.DataFrame()

    try:
        # Explicitly use openpyxl so it works better across environments
        excel_file = pd.ExcelFile(DATA_PATH, engine="openpyxl")
        sheet_names = excel_file.sheet_names
        print(f"[backend] Loaded Menus.xlsx with sheets: {sheet_names}")

        dfs: Dict[str, pd.DataFrame] = {}
        for sheet_name in sheet_names:
            df = pd.read_excel(DATA_PATH, sheet_name=sheet_name, engine="openpyxl")

            # --- Cleaning and Standardization ---
            column_mapping = {
                "Name of item": "Name",
                "Total calories": "Calories",
                "Total Calories": "Calories",
                "Protein (grams)": "Protein (g)",
                "Carbs (grams)": "Carbs (g)",
                "Fat (grams)": "Fat (g)",
                "Sugar (grams)": "Sugar (g)",
            }
            df = df.rename(columns=column_mapping)

            # Clean 'Price' column
            if "Price" in df.columns:
                if df["Price"].dtype == "object":
                    df["Price"] = df["Price"].astype(str)
                    df["Price"] = df["Price"].str.replace("$", "", regex=False).str.strip()
                    df["Price"] = df["Price"].apply(
                        lambda x: re.match(r"\d+\.?\d*", x).group(0)
                        if re.match(r"\d+\.?\d*", x)
                        else None
                    )
                    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

            # Handle missing 'Sugar (g)'
            if "Sugar (g)" in df.columns:
                df["Sugar (g)"] = df["Sugar (g)"].fillna(0)

            # Fill numeric columns with mean
            numeric_cols_to_fill = ["Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]
            for col in numeric_cols_to_fill:
                if col in df.columns and df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)

            dfs[sheet_name] = df

        combined = []
        for restaurant_name, df in dfs.items():
            df = df.copy()
            df["Restaurant"] = restaurant_name
            combined.append(df)

        if combined:
            full_df = pd.concat(combined, ignore_index=True)
            print(f"[backend] Combined menu rows: {len(full_df)}")
            return full_df

        print("[backend] Combined menus empty after loading.")
        return pd.DataFrame()

    except Exception as e:
        # This is where LFS / engine / weird cloud stuff will show
        print(f"[backend] Error loading Menus.xlsx: {e}")
        return pd.DataFrame()


def _build_fallback_menus() -> pd.DataFrame:
    """
    Build a tiny in-memory menu so the app still works
    even if Menus.xlsx cannot be used on Streamlit Cloud.
    """
    print("[backend] Using fallback in-memory menu data.")

    data = [
        {
            "Restaurant": "Sweetgreen Menu",
            "Name": "Kale Caesar Salad",
            "Price": 12.50,
            "Calories": 450,
            "Protein (g)": 20,
            "Carbs (g)": 30,
            "Fat (g)": 25,
            "Sugar (g)": 5,
        },
        {
            "Restaurant": "Frank Pepe Menu",
            "Name": "Margherita Pizza (Slice)",
            "Price": 5.00,
            "Calories": 350,
            "Protein (g)": 15,
            "Carbs (g)": 40,
            "Fat (g)": 14,
            "Sugar (g)": 4,
        },
        {
            "Restaurant": "BAR Pizza",
            "Name": "Veggie Pizza (Slice)",
            "Price": 5.50,
            "Calories": 320,
            "Protein (g)": 12,
            "Carbs (g)": 38,
            "Fat (g)": 11,
            "Sugar (g)": 3,
        },
        {
            "Restaurant": "Mecha Noodle Bar",
            "Name": "Spicy Ramen Bowl",
            "Price": 16.00,
            "Calories": 700,
            "Protein (g)": 30,
            "Carbs (g)": 65,
            "Fat (g)": 28,
            "Sugar (g)": 6,
        },
    ]
    return pd.DataFrame(data)


# Try real data first
all_menus_df = _load_real_menus()

# If real data fails (cloud issues, LFS, etc.), use fallback
if all_menus_df.empty:
    all_menus_df = _build_fallback_menus()


# -------------------------------------------------------------------
# 2. Dietary tagging and recommendation logic (from your notebook)
# -------------------------------------------------------------------

def tag_dietary_preferences(df: pd.DataFrame) -> pd.DataFrame:
    df["is_vegetarian"] = False
    df["is_vegan"] = False
    df["is_gluten_free"] = False
    df["is_dairy_free"] = False
    df["is_nut_free"] = False

    # Sweetgreen vegetarian tagging
    df.loc[df["Restaurant"] == "Sweetgreen Menu", "is_vegetarian"] = ~df["Name"].str.contains(
        "Steak|Chicken|Tuna|Shrimp|Salmon|Pork|Bacon", case=False, na=False
    )

    # Pizza / Frank Pepe / BAR vegetarian tagging
    df.loc[
        (
            df["Restaurant"].str.contains("Pizza|Frank Pepe|BAR", case=False, na=False)
        )
        & (
            df["Name"].str.contains(
                "Vegetable|Margherita|Mushroom|Spinach|Plain|Cheese|Tomato Pie",
                case=False,
                na=False,
            )
        )
        & (
            ~df["Name"].str.contains(
                "Meatball|Pepperoni|Sausage|Clam|Bacon", case=False, na=False
            )
        ),
        "is_vegetarian",
    ] = True

    # Vegan / gluten-free / dairy-free / nut-free tagging
    df.loc[df["Name"].str.contains("Vegan", case=False, na=False), "is_vegan"] = True
    df.loc[
        df["Name"].str.contains("Gluten Free|Gluten-Free", case=False, na=False),
        "is_gluten_free",
    ] = True
    df.loc[
        df["Name"].str.contains("Dairy Free|Dairy-Free|No Cheese", case=False, na=False),
        "is_dairy_free",
    ] = True
    df.loc[
        ~df["Name"].str.contains(
            "Nut|Almond|Peanut|Cashew|Walnut|Pecan", case=False, na=False
        ),
        "is_nut_free",
    ] = True

    return df


def parse_user_query(query: str) -> Dict[str, Any]:
    """Parses a natural language query into structured parameters."""
    parsed_data: Dict[str, Any] = {
        "cuisine": None,
        "max_calories": None,
        "excluded_ingredients": [],
        "dietary_preferences": [],
    }

    query_lower = query.lower()

    cuisines = [
        "japanese",
        "italian",
        "mexican",
        "american",
        "indian",
        "chinese",
        "thai",
        "mediterranean",
        "pizza",
        "noodle",
        "salad",
        "burger",
    ]
    for c in cuisines:
        if c in query_lower:
            parsed_data["cuisine"] = c.capitalize()
            break

    calorie_match = re.search(r"(?:under|within)\s*(\d+)\s*calories?", query_lower)
    if calorie_match:
        parsed_data["max_calories"] = int(calorie_match.group(1))

    exclusion_patterns = [r"no (\w+)", r"(\w+)-free"]
    for pattern in exclusion_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            parsed_data["excluded_ingredients"].append(match)

    specific_exclusions_match = re.search(
        r"no\s+(.+?)(?:or\s+(.+?))?(?:\s+allerg(?:y|ies)|\s+exclusion)?(?:\s|$)",
        query_lower,
    )
    if specific_exclusions_match:
        items = specific_exclusions_match.group(1)
        if specific_exclusions_match.group(2):
            items += " or " + specific_exclusions_match.group(2)

        for item in items.split(" or "):
            clean_item = item.strip()
            if clean_item and clean_item not in parsed_data["excluded_ingredients"]:
                parsed_data["excluded_ingredients"].append(clean_item)

    preferences = [
        "vegetarian",
        "vegan",
        "pescatarian",
        "keto",
        "paleo",
        "low-carb",
        "low-fat",
        "high-protein",
    ]
    for p in preferences:
        if p in query_lower:
            parsed_data["dietary_preferences"].append(p)

    return parsed_data


def retrieve_menu_items(all_menus_df: pd.DataFrame, parsed_query: Dict[str, Any]) -> pd.DataFrame:
    """Retrieves menu items based on parsed user query parameters, leveraging dietary tags."""
    filtered_df = all_menus_df.copy()

    # Cuisine-based restaurant filtering
    if parsed_query["cuisine"]:
        cuisine_lower = parsed_query["cuisine"].lower()
        if cuisine_lower == "japanese":
            filtered_df = filtered_df[
                filtered_df["Restaurant"].str.contains(
                    "Mecha Noodle Bar", case=False, na=False
                )
            ]
        elif cuisine_lower == "pizza":
            filtered_df = filtered_df[
                filtered_df["Restaurant"].str.contains(
                    "Pizza|Frank Pepe|BAR", case=False, na=False
                )
            ]
        elif cuisine_lower == "american":
            filtered_df = filtered_df[
                filtered_df["Restaurant"].str.contains("BAR", case=False, na=False)
            ]

    # Calorie filtering
    if parsed_query["max_calories"] is not None and "Calories" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Calories"] <= parsed_query["max_calories"]]

    excluded_keywords = [item.lower() for item in parsed_query["excluded_ingredients"]]

    # Allergy / restriction filters using tags or name-based heuristics
    if "nuts" in excluded_keywords and "is_nut_free" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["is_nut_free"]]

    if "gluten" in excluded_keywords and "is_gluten_free" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["is_gluten_free"]]

    if "dairy" in excluded_keywords and "is_dairy_free" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["is_dairy_free"]]

    if "seafood" in excluded_keywords:
        filtered_df = filtered_df[
            ~filtered_df["Name"].str.contains(
                "seafood|shrimp|fish|crab|lobster", case=False, na=False
            )
        ]
    if "soy" in excluded_keywords:
        filtered_df = filtered_df[
            ~filtered_df["Name"].str.contains("soy", case=False, na=False)
        ]

    # Dietary preferences
    if "vegetarian" in parsed_query["dietary_preferences"] and "is_vegetarian" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["is_vegetarian"]]

    if "vegan" in parsed_query["dietary_preferences"] and "is_vegan" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["is_vegan"]]

    if "high-protein" in parsed_query["dietary_preferences"] and "Protein (g)" in filtered_df.columns:
        protein_threshold = all_menus_df["Protein (g)"].quantile(0.75)
        filtered_df = filtered_df[filtered_df["Protein (g)"] >= protein_threshold]

    return filtered_df


def generate_recommendations(all_menus_df: pd.DataFrame, parsed_query: Dict[str, Any]) -> Union[str, List[Dict[str, Any]]]:
    """Generates restaurant and menu item recommendations based on a parsed user query."""
    if all_menus_df.empty:
        return "Menu data is not available, even after fallback."

    filtered_items_df = retrieve_menu_items(all_menus_df, parsed_query)

    if filtered_items_df.empty:
        return "No recommendations found matching your criteria."

    filtered_items_df = filtered_items_df.copy()

    # Value metric: calories per dollar (for items with valid prices)
    if "Calories" in filtered_items_df.columns and "Price" in filtered_items_df.columns:
        filtered_items_df["value_metric"] = filtered_items_df.apply(
            lambda row: row["Calories"] / row["Price"]
            if pd.notna(row["Price"]) and row["Price"] > 0
            else 0,
            axis=1,
        )
    else:
        filtered_items_df["value_metric"] = 0

    recommendations: List[Dict[str, Any]] = []

    for restaurant_name, restaurant_df in filtered_items_df.groupby("Restaurant"):
        restaurant_df = restaurant_df.sort_values(by="value_metric", ascending=False)
        top_item = restaurant_df.iloc[0]

        recommended_dish = {
            "Name": top_item.get("Name"),
            "Price": top_item.get("Price"),
            "Calories": top_item.get("Calories"),
            "Protein (g)": top_item.get("Protein (g)"),
            "Carbs (g)": top_item.get("Carbs (g)"),
            "Fat (g)": top_item.get("Fat (g)"),
            "Sugar (g)": top_item.get("Sugar (g)"),
        }

        dietary_notes: List[str] = []
        if "vegetarian" in parsed_query["dietary_preferences"] and top_item.get("is_vegetarian"):
            dietary_notes.append("Vegetarian compliant")
        if "vegan" in parsed_query["dietary_preferences"] and top_item.get("is_vegan"):
            dietary_notes.append("Vegan compliant")
        if "gluten" in parsed_query["excluded_ingredients"] and top_item.get("is_gluten_free"):
            dietary_notes.append("Gluten-free compliant")
        if "dairy" in parsed_query["excluded_ingredients"] and top_item.get("is_dairy_free"):
            dietary_notes.append("Dairy-free compliant")
        if "nuts" in parsed_query["excluded_ingredients"] and top_item.get("is_nut_free"):
            dietary_notes.append("Nut-free compliant")
        if "high-protein" in parsed_query["dietary_preferences"] and "Protein (g)" in all_menus_df.columns:
            protein_threshold = all_menus_df["Protein (g)"].quantile(0.75)
            if top_item.get("Protein (g)", 0) >= protein_threshold:
                dietary_notes.append("High-protein compliant")

        for excluded_item in parsed_query["excluded_ingredients"]:
            if excluded_item.lower() in ["seafood", "soy"]:
                name_val = top_item.get("Name")
                if pd.notna(name_val) and excluded_item.lower() not in str(name_val).lower():
                    dietary_notes.append(
                        f"Excluded {excluded_item} (by name check) compliant"
                    )

        if not dietary_notes:
            dietary_notes = ["No specific dietary compliance noted for this item."]

        recommendations.append(
            {
                "Restaurant Name": restaurant_name,
                "Recommended Dish(es)": [recommended_dish],
                "Total Price": recommended_dish["Price"],
                "Nutritional Summary": recommended_dish,
                "Dietary Compliance Notes": dietary_notes,
                "Availability": "Available",
            }
        )

    return recommendations


# Apply dietary tagging once data is loaded
if not all_menus_df.empty:
    all_menus_df = tag_dietary_preferences(all_menus_df.copy())
    print(f"[backend] Tagged dietary preferences on {len(all_menus_df)} rows.")


# -------------------------------------------------------------------
# 3. Single entry point for the Streamlit front end
# -------------------------------------------------------------------

def get_recommendation(cuisine: str, max_calories: int, restrictions: str) -> Dict[str, Any]:
    """
    Entry point for app.py.

    Takes structured inputs from the UI, builds a natural language query,
    runs the pipeline, and returns a single dish/restaurant in a simple dict.
    """

    parts: List[str] = []
    if cuisine:
        parts.append(f"{cuisine} meal")
    if max_calories:
        parts.append(f"under {max_calories} calories")
    if restrictions:
        parts.append(restrictions)

    user_query = ", ".join(parts) if parts else "meal recommendation"

    parsed_output = parse_user_query(user_query)
    # If user didn't mention calories in text, inject from slider
    if max_calories and parsed_output.get("max_calories") is None:
        parsed_output["max_calories"] = max_calories

    recommendations = generate_recommendations(all_menus_df, parsed_output)

    if isinstance(recommendations, str):
        return {
            "restaurant_name": "No match found",
            "dish_name": "N/A",
            "total_calories": max_calories,
            "nutritional_breakdown": recommendations,
            "notes": "Try adjusting your cuisine, calorie target, or restrictions.",
        }

    if not recommendations:
        return {
            "restaurant_name": "No match found",
            "dish_name": "N/A",
            "total_calories": max_calories,
            "nutritional_breakdown": "No recommendations matched your criteria.",
            "notes": "Try adjusting your query.",
        }

    first_rec = recommendations[0]
    restaurant_name = first_rec.get("Restaurant Name", "Unknown restaurant")

    dishes = first_rec.get("Recommended Dish(es)", [])
    first_dish = dishes[0] if dishes else {}

    dish_name = first_dish.get("Name", "Unknown dish")
    calories = first_dish.get("Calories", "N/A")
    protein = first_dish.get("Protein (g)", "N/A")
    carbs = first_dish.get("Carbs (g)", "N/A")
    fat = first_dish.get("Fat (g)", "N/A")
    sugar = first_dish.get("Sugar (g)", "N/A")

    breakdown_lines = [
        f"Calories: {calories} kcal",
        f"Protein: {protein} g",
        f"Carbs: {carbs} g",
        f"Fat: {fat} g",
        f"Sugar: {sugar} g",
    ]
    nutritional_breakdown = "\n".join(breakdown_lines)

    dietary_notes = first_rec.get("Dietary Compliance Notes", [])
    notes_text = ", ".join(dietary_notes) if dietary_notes else ""

    return {
        "restaurant_name": restaurant_name,
        "dish_name": dish_name,
        "total_calories": calories,
        "nutritional_breakdown": nutritional_breakdown,
        "notes": notes_text,
    }