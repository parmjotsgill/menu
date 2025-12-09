import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# --- Helper function for dietary tagging (from initial data loading) ---
def tag_dietary_preferences(df):
    df['is_vegetarian'] = False
    df['is_vegan'] = False
    df['is_gluten_free'] = False
    df['is_dairy_free'] = False
    df['is_nut_free'] = False

    df.loc[df['Restaurant'] == 'Sweetgreen Menu', 'is_vegetarian'] = \
        ~df['Name'].str.contains('Steak|Chicken|Tuna|Shrimp|Salmon|Pork|Bacon', case=False, na=False)

    df.loc[(df['Restaurant'].str.contains('Pizza|Frank Pepe|BAR', case=False, na=False)) & \
           (df['Name'].str.contains('Vegetable|Margherita|Mushroom|Spinach|Plain|Cheese|Tomato Pie', case=False, na=False)) & \
           (~df['Name'].str.contains('Meatball|Pepperoni|Sausage|Clam|Bacon', case=False, na=False)), 'is_vegetarian'] = True

    df.loc[df['Name'].str.contains('Vegan', case=False, na=False), 'is_vegan'] = True
    df.loc[df['Name'].str.contains('Gluten Free|Gluten-Free', case=False, na=False), 'is_gluten_free'] = True
    df.loc[df['Name'].str.contains('Dairy Free|Dairy-Free|No Cheese', case=False, na=False), 'is_dairy_free'] = True
    df.loc[~df['Name'].str.contains('Nut|Almond|Peanut|Cashew|Walnut|Pecan', case=False, na=False), 'is_nut_free'] = True

    return df

def categorize_item_type(row):
    """Categorizes a menu item as 'Entree' or 'Side' based on calories and name keywords."""
    calorie_threshold = 600 # Calorie threshold for a side item
    side_keywords = ['side', 'soup', 'fries', 'bread', 'appetizer', 'small', 'salad', 'starter']

    item_name_lower = str(row['Name']).lower()

    # Check for calorie criteria and side keywords
    if row['Calories'] < calorie_threshold and any(keyword in item_name_lower for keyword in side_keywords):
        return 'Side'
    else:
        return 'Entree'

# --- Data Loading and Preprocessing (Global scope for `backend.py`) ---

# Define the actual file path in Google Drive
file_path = '/content/drive/MyDrive/Menus.xlsx' # Primary path to user's data

dfs_menus = {}
try:
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names

    for sheet_name in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        dfs_menus[sheet_name] = df
    print(f"[backend] Successfully loaded menu data from '{file_path}'.")
except FileNotFoundError:
    print(f"[backend] Error: The file '{file_path}' was not found. Please ensure the file exists and the path is correct.")
    # Fallback for demonstration if file not found, creating in-memory data
    print("[backend] Menus.xlsx not found at Google Drive path, using in-memory dummy data for demonstration.")
    dfs_menus = {
        'Sweetgreen Menu': pd.DataFrame({
            'Name of item': ['Kale Salad', 'Chicken & Rice', 'Tuna Bowl'],
            'Total Calories': [350, 500, 420],
            'Protein (grams)': [15, 35, 30],
            'Carbs (grams)': [30, 45, 25],
            'Fat (grams)': [20, 20, 25],
            'Sugar (grams)': [5, 8, 4],
            'Price': [12.50, 15.00, 14.75]
        }),
        'Frank Pepe Menu': pd.DataFrame({
            'Name of item': ['Large Cheese Pizza', 'Pepperoni Pizza', 'Vegetable Pizza'],
            'Total Calories': [1800, 2200, 1950],
            'Protein (grams)': [80, 100, 90],
            'Carbs (grams)': [200, 250, 220],
            'Fat (grams)': [70, 90, 80],
            'Sugar (grams)': [15, 20, 18],
            'Price': [25.00, 28.00, 27.50]
        })
    }
except Exception as e:
    print(f"[backend] An error occurred during file loading from '{file_path}': {e}")

standardized_dfs_menus = {}
if dfs_menus:
    for sheet_name, df_menu in dfs_menus.items(): # Renamed df to df_menu
        column_mapping = {
            'Name of item': 'Name',
            'Total calories': 'Calories',
            'Total Calories': 'Calories',
            'Protein (grams)': 'Protein (g)',
            'Carbs (grams)': 'Carbs (g)',
            'Fat (grams)': 'Fat (g)',
            'Sugar (grams)': 'Sugar (g)'
        }
        df_menu = df_menu.rename(columns=column_mapping)

        if 'Price' in df_menu.columns:
            if df_menu['Price'].dtype == 'object':
                df_menu['Price'] = df_menu['Price'].astype(str)
                df_menu['Price'] = df_menu['Price'].str.replace('$', '', regex=False).str.strip()
                df_menu['Price'] = df_menu['Price'].apply(lambda x: re.match(r'\d+\.?\d*', x).group(0) if re.match(r'\d+\.?\d*', x) else None)
                df_menu['Price'] = pd.to_numeric(df_menu['Price'], errors='coerce')

        if 'Sugar (g)' in df_menu.columns:
            df_menu['Sugar (g)'] = df_menu['Sugar (g)'].fillna(0)

        numeric_cols_to_fill = ['Calories', 'Protein (g)', 'Carbs (g)', 'Fat (g)']
        for col in numeric_cols_to_fill:
            if col in df_menu.columns and df_menu[col].isnull().any():
                mean_val = df_menu[col].mean()
                df_menu[col] = df_menu[col].fillna(mean_val)

        standardized_dfs_menus[sheet_name] = df_menu

dfs_menus = standardized_dfs_menus

all_menus_df = pd.DataFrame()
if dfs_menus:
    for restaurant_name, df_menu in dfs_menus.items():
        df_menu['Restaurant'] = restaurant_name
        all_menus_df = pd.concat([all_menus_df, df_menu], ignore_index=True)

    # Apply dietary preferences tagging
    all_menus_df = tag_dietary_preferences(all_menus_df.copy())
    print("[backend] Tagged dietary preferences on " + str(len(all_menus_df)) + " rows.")

    # Apply item type categorization
    all_menus_df['Item_Type'] = all_menus_df.apply(categorize_item_type, axis=1)
    print("[backend] Categorized item types on " + str(len(all_menus_df)) + " rows.")

    # Create 'text_for_embedding' column for RAG
    all_menus_df['text_for_embedding'] = all_menus_df['Name'].fillna('').astype(str) + ' (Restaurant: ' + all_menus_df['Restaurant'].fillna('').astype(str) + ')'
    dietary_cols_map = {
        'is_vegetarian': 'vegetarian',
        'is_vegan': 'vegan',
        'is_gluten_free': 'gluten-free',
        'is_dairy_free': 'dairy-free',
        'is_nut_free': 'nut-free'
    }
    for col, tag in dietary_cols_map.items():
        if col in all_menus_df.columns:
            all_menus_df.loc[all_menus_df[col] == True, 'text_for_embedding'] = \
                all_menus_df['text_for_embedding'] + f', {tag}'
    print("[backend] Created text_for_embedding column.")

# --- SentenceTransformer Model and FAISS Index (Global scope) ---
model = None
index = None
if not all_menus_df.empty:
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        menu_texts = all_menus_df['text_for_embedding'].tolist()
        item_embeddings = model.encode(menu_texts, convert_to_numpy=True)
        item_embeddings = item_embeddings.astype('float32') # FAISS requires float32

        d = item_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(item_embeddings)
        print("[backend] SentenceTransformer model and FAISS index initialized.")
    except Exception as e:
        print(f"[backend] Error initializing RAG components: {e}")
        model = None
        index = None
else:
    print("[backend] all_menus_df is empty, RAG components not initialized.")


def parse_user_query(query):
    """Parses a natural language query into structured parameters."""
    parsed_data = {
        'cuisine': None,
        'max_calories': None,
        'excluded_ingredients': [],
        'dietary_preferences': [],
        'query_text': query,
        'multi_item_request': [],
        'data_query': None
    }

    query_lower = query.lower()

    cuisines = ['japanese', 'italian', 'mexican', 'american', 'indian', 'chinese', 'thai', 'mediterranean', 'pizza', 'noodle', 'salad', 'burger']
    for c in cuisines:
        if c in query_lower:
            parsed_data['cuisine'] = c.capitalize()
            break

    # Overall max calories for the entire meal (if specified)
    calorie_match = re.search(r'(?:under|within)\s*(\d+)\s*calories?', query_lower)
    if calorie_match:
        parsed_data['max_calories'] = int(calorie_match.group(1))

    # Detect multi-item requests and their individual constraints
    item_type_keywords = {
        'entree': ['entree', 'entrees'],
        'side': ['side', 'sides'],
        'dish': ['dish', 'dishes']
    }

    # Split the query into potential item request clauses, handling various separators
    item_clauses = re.split(r'\s+and\s+|\s*,\s*', query_lower)

    for clause in item_clauses:
        found_item_type = None
        for item_type_key, keywords in item_type_keywords.items():
            for keyword in keywords:
                if keyword in clause:
                    found_item_type = item_type_key
                    break
            if found_item_type:
                break

        if found_item_type:
            max_calories_per_item = None
            calories_match_clause = re.search(r'under\s*(\d+)\s*calories?\s+each', clause)
            if calories_match_clause:
                max_calories_per_item = int(calories_match_clause.group(1))

            num_items = 1 # Default count
            if 'two' in clause:
                num_items = 2
            elif 'multiple' in clause:
                num_items = None # Indicates unspecified multiple items

            # Prevent adding duplicate item types with the same calorie constraint
            item_already_added = False
            for existing_item in parsed_data['multi_item_request']:
                if existing_item['item_type'] == found_item_type and \
                   existing_item['max_calories_per_item'] == max_calories_per_item:
                    item_already_added = True
                    break

            if not item_already_added:
                parsed_data['multi_item_request'].append({
                    'item_type': found_item_type,
                    'max_calories_per_item': max_calories_per_item,
                    'count': num_items
                })

    # Implement regex patterns to detect data-oriented questions
    # Keywords for data queries: highest, lowest, count, average, total
    # Target attributes: calories, protein, carbs, fat, sugar, price
    data_query_patterns = {
        'highest': r'(highest|most)\s*(calorie|protein|carb|fat|sugar|price)s?',
        'lowest': r'(lowest|least)\s*(calorie|protein|carb|fat|sugar|price)s?',
        'count': r'count\s*(.+?)?(?:items|dishes)?(?:with)?\s*(calories|protein|carb|fat|sugar|price)?',
        'average': r'average\s*(calorie|protein|carb|fat|sugar|price)s?',
        'total': r'total\s*(calorie|protein|carb|fat|sugar|price)s?'
    }

    for query_type, pattern in data_query_patterns.items():
        match = re.search(pattern, query_lower)
        if match:
            target_attribute = None
            if query_type in ['highest', 'lowest', 'average', 'total']:
                target_attribute = match.group(2) if query_type in ['highest', 'lowest'] else match.group(1)
            elif query_type == 'count':
                # For count, the target could be implied or part of a broader filter
                # For simplicity, if a specific attribute is mentioned, use that
                if len(match.groups()) > 1 and match.group(2):
                    target_attribute = match.group(2)
                else:
                    target_attribute = 'item' # Default for count

            parsed_data['data_query'] = {
                'type': query_type,
                'target': target_attribute.replace('s', '') if target_attribute else None # Remove plural for standardization
            }
            break # Found a data query, stop checking further patterns

    # Refined exclusion patterns for better precision and to avoid duplicates
    potential_exclusions = set() # Use a set to automatically handle duplicates

    # Pattern for "no X" and "X-free"
    for match in re.finditer(r'(?:no\s+|([\w-]+)-free)\s*([\w\s-]+?)(?=\s+or\s+|\s+and\s+|\s+allerg(?:y|ies)|\s+exclusion|\s|$|,|.)', query_lower):
        if match.group(1): # X-free pattern
            parts = re.split(r'\s+or\s+|\s+and\s+', match.group(1))
        else: # no X pattern
            parts = re.split(r'\s+or\s+|\s+and\s+', match.group(2))

        for part in parts:
            clean_part = part.strip().replace('.', '') # Clean up any trailing punctuation
            if clean_part:
                potential_exclusions.add(clean_part)

    # Convert set back to list and filter for common dietary restrictions first
    common_excluded_ingredients = ['gluten', 'dairy', 'nut', 'shellfish', 'soy', 'egg']
    for item in potential_exclusions:
        if item in common_excluded_ingredients or item not in parsed_data['excluded_ingredients']:
            parsed_data['excluded_ingredients'].append(item)


    preferences = ['vegetarian', 'vegan', 'pescatarian', 'keto', 'paleo', 'low-carb', 'low-fat', 'high-protein']
    for p in preferences:
        if p in query_lower:
            parsed_data['dietary_preferences'].append(p)

    return parsed_data

def retrieve_menu_items(all_menus_df, parsed_query, model, index, k=10, target_item_type=None, target_max_calories=None):
    """Retrieves menu items based on parsed user query parameters, leveraging dietary tags and RAG."""

    # Handle cases where model or index might not be initialized due to empty data
    if model is None or index is None or all_menus_df.empty:
        return pd.DataFrame() # Return empty if RAG components are not ready

    # 1. Embed the user's query
    user_query_embedding = model.encode([parsed_query['query_text']], convert_to_numpy=True)
    user_query_embedding = user_query_embedding.astype('float32')

    # 2. Perform semantic search on the FAISS index
    # D: distances, I: indices of the nearest neighbors
    distances, item_indices = index.search(user_query_embedding, k)

    # Ensure indices are valid and flatten them
    valid_item_indices = item_indices[0][item_indices[0] != -1] # Filter out -1 if k is larger than available items

    # 3. Filter all_menus_df to include only semantically retrieved items
    if len(valid_item_indices) > 0:
        filtered_df = all_menus_df.iloc[valid_item_indices].copy() # Use .copy() to avoid SettingWithCopyWarning
    else:
        return pd.DataFrame() # Return empty if no semantic matches

    # --- New Filtering based on target_item_type ---
    if target_item_type:
        filtered_df = filtered_df[filtered_df['Item_Type'].str.lower() == target_item_type.lower()]
        if filtered_df.empty: # If no items match the type, return empty
            return pd.DataFrame() # Return empty if no items match the target type

    # Now apply existing filters to this semantically pre-filtered DataFrame

    if parsed_query['cuisine']:
        cuisine_lower = parsed_query['cuisine'].lower()
        if cuisine_lower == 'japanese':
            filtered_df = filtered_df[filtered_df['Restaurant'].str.contains('Mecha Noodle Bar', case=False, na=False)]
        elif cuisine_lower == 'pizza':
            filtered_df = filtered_df[filtered_df['Restaurant'].str.contains('Pizza|Frank Pepe|BAR', case=False, na=False)]
        elif cuisine_lower == 'american':
             filtered_df = filtered_df[filtered_df['Restaurant'].str.contains('BAR', case=False, na=False)]

    # --- Modified Calorie Filtering Logic ---
    if target_max_calories is not None:
        filtered_df = filtered_df[filtered_df['Calories'] <= target_max_calories]
    elif parsed_query['max_calories'] is not None: # Fallback to overall max_calories if no specific item calorie limit
        filtered_df = filtered_df[filtered_df['Calories'] <= parsed_query['max_calories']]

    excluded_keywords = [item.lower() for item in parsed_query['excluded_ingredients']]

    if 'nuts' in excluded_keywords and 'is_nut_free' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_nut_free']]
    elif 'nuts' in excluded_keywords:
        pass # Cannot accurately filter without explicit tag

    if 'gluten' in excluded_keywords and 'is_gluten_free' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_gluten_free']]
    elif 'gluten' in excluded_keywords:
        pass # Cannot accurately filter without explicit tag

    if 'dairy' in excluded_keywords and 'is_dairy_free' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_dairy_free']]
    elif 'dairy' in excluded_keywords:
        pass # Cannot accurately filter without explicit tag

    if 'seafood' in excluded_keywords:
        filtered_df = filtered_df[~filtered_df['Name'].str.contains('seafood|shrimp|fish|crab|lobster', case=False, na=False)]
    if 'soy' in excluded_keywords:
        filtered_df = filtered_df[~filtered_df['Name'].str.contains('soy', case=False, na=False)]

    if 'vegetarian' in parsed_query['dietary_preferences'] and 'is_vegetarian' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_vegetarian']]
    elif 'vegetarian' in parsed_query['dietary_preferences']:
        pass # Cannot accurately filter without explicit tag

    if 'vegan' in parsed_query['dietary_preferences'] and 'is_vegan' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_vegan']]
    elif 'vegan' in parsed_query['dietary_preferences']:
        pass # Cannot accurately filter without explicit tag

    if 'high-protein' in parsed_query['dietary_preferences']:
        # Ensure all_menus_df is used for overall quantile calculation if filtered_df might be too small
        protein_threshold = all_menus_df['Protein (g)'].quantile(0.75)
        filtered_df = filtered_df[filtered_df['Protein (g)'] >= protein_threshold]

    return filtered_df

def handle_data_query(df, parsed_query):
    """Executes data-oriented queries against the menu DataFrame."""
    data_query = parsed_query.get('data_query')

    if not data_query:
        return "No data query found in parsed request."

    query_type = data_query['type']
    target_attribute = data_query['target']

    # Apply filters based on other parsed query parameters before executing data query
    filtered_df = df.copy()

    if parsed_query['cuisine']:
        cuisine_lower = parsed_query['cuisine'].lower()
        if cuisine_lower == 'japanese': # Assuming 'Mecha Noodle Bar' is Japanese
            filtered_df = filtered_df[filtered_df['Restaurant'].str.contains('Mecha Noodle Bar', case=False, na=False)]
        elif cuisine_lower == 'pizza': # Assuming 'Frank Pepe' and 'BAR Pizza' are pizza places
            filtered_df = filtered_df[filtered_df['Restaurant'].str.contains('Pizza|Frank Pepe|BAR', case=False, na=False)]
        elif cuisine_lower == 'american': # Assuming '80 Proof American Kitchen & Bar' is American
            filtered_df = filtered_df[filtered_df['Restaurant'].str.contains('BAR', case=False, na=False)]

    if parsed_query['max_calories'] is not None:
        filtered_df = filtered_df[filtered_df['Calories'] <= parsed_query['max_calories']]

    excluded_keywords = [item.lower() for item in parsed_query['excluded_ingredients']]

    if 'nuts' in excluded_keywords and 'is_nut_free' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_nut_free']]
    if 'gluten' in excluded_keywords and 'is_gluten_free' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_gluten_free']]
    if 'dairy' in excluded_keywords and 'is_dairy_free' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_dairy_free']]

    # Specific keyword exclusions from item name
    if 'seafood' in excluded_keywords:
        filtered_df = filtered_df[~filtered_df['Name'].str.contains('seafood|shrimp|fish|crab|lobster', case=False, na=False)]
    if 'soy' in excluded_keywords:
        filtered_df = filtered_df[~filtered_df['Name'].str.contains('soy', case=False, na=False)]

    if 'vegetarian' in parsed_query['dietary_preferences'] and 'is_vegetarian' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_vegetarian']]
    if 'vegan' in parsed_query['dietary_preferences'] and 'is_vegan' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_vegan']]
    if 'high-protein' in parsed_query['dietary_preferences']:
        protein_threshold = df['Protein (g)'].quantile(0.75)
        filtered_df = filtered_df[filtered_df['Protein (g)'] >= protein_threshold]


    if filtered_df.empty:
        return "No items found after applying filters for the data query."

    # Map target attribute to DataFrame column name
    col_map = {
        'calorie': 'Calories',
        'protein': 'Protein (g)',
        'carb': 'Carbs (g)',
        'fat': 'Fat (g)',
        'sugar': 'Sugar (g)',
        'price': 'Price'
    }
    df_col = col_map.get(target_attribute)

    if query_type == 'highest':
        if df_col and df_col in filtered_df.columns:
            # Ensure the column is numeric before idxmax()
            if pd.api.types.is_numeric_dtype(filtered_df[df_col]):
                max_val_row = filtered_df.loc[filtered_df[df_col].idxmax()]
                return f"Highest {target_attribute}: {max_val_row[df_col]} ({max_val_row['Name']} at {max_val_row['Restaurant']})"
            else:
                return f"Cannot find highest {target_attribute}. Column '{df_col}' is not numeric."
        else:
            return f"Cannot find highest {target_attribute}. Invalid column or not available."
    elif query_type == 'lowest':
        if df_col and df_col in filtered_df.columns:
            # Ensure the column is numeric before idxmin()
            if pd.api.types.is_numeric_dtype(filtered_df[df_col]):
                min_val_row = filtered_df.loc[filtered_df[df_col].idxmin()]
                return f"Lowest {target_attribute}: {min_val_row[df_col]} ({min_val_row['Name']} at {min_val_row['Restaurant']})"
            else:
                return f"Cannot find lowest {target_attribute}. Column '{df_col}' is not numeric."
        else:
            return f"Cannot find lowest {target_attribute}. Invalid column or not available."
    elif query_type == 'count':
        # For count, the target_attribute 'item' implies counting all filtered items
        if target_attribute == 'item' or not df_col:
            return f"Count of items: {len(filtered_df)}."
        elif df_col and df_col in filtered_df.columns:
            # If a specific attribute is requested for count, e.g., 'count items with high protein'
            # For now, it will count all items that meet the initial filters.
            return f"Count of items with specified {target_attribute} criteria: {len(filtered_df)}."
        else:
             return f"Count of items: {len(filtered_df)}."
    elif query_type == 'average':
        if df_col and df_col in filtered_df.columns:
            if pd.api.types.is_numeric_dtype(filtered_df[df_col]):
                avg_val = filtered_df[df_col].mean()
                return f"Average {target_attribute}: {avg_val:.2f}"
            else:
                return f"Cannot calculate average {target_attribute}. Column '{df_col}' is not numeric."
        else:
            return f"Cannot calculate average {target_attribute}. Invalid column or not available."
    elif query_type == 'total':
        if df_col and df_col in filtered_df.columns:
            if pd.api.types.is_numeric_dtype(filtered_df[df_col]):
                total_val = filtered_df[df_col].sum()
                return f"Total {target_attribute}: {total_val:.2f}"
            else:
                return f"Cannot calculate total {target_attribute}. Column '{df_col}' is not numeric."
        else:
            return f"Cannot calculate total {target_attribute}. Invalid column or not available."
    else:
        return "Unsupported data query type."

def generate_recommendations(all_menus_df, parsed_query, model, index):
    """Generates restaurant and menu item recommendations based on a parsed user query."""

    recommendations = []

    # --- Multi-item request handling ---
    if parsed_query['multi_item_request']:
        multi_item_recommendations = []

        for restaurant_name in all_menus_df['Restaurant'].unique():
            current_order_dishes = []
            total_price = 0.0
            total_calories = 0.0
            total_protein = 0.0
            total_carbs = 0.0
            total_fat = 0.0
            total_sugar = 0.0
            all_items_found_for_restaurant = True

            for item_request in parsed_query['multi_item_request']:
                # Retrieve items for the specific item type and calorie constraint
                # Pass target_item_type and target_max_calories
                item_type_filtered_df = retrieve_menu_items(
                    all_menus_df,
                    parsed_query,
                    model,
                    index,
                    target_item_type=item_request['item_type'],
                    target_max_calories=item_request['max_calories_per_item']
                )

                # Filter further to include only items from the current restaurant
                restaurant_specific_items = item_type_filtered_df[
                    item_type_filtered_df['Restaurant'] == restaurant_name
                ].copy()

                if not restaurant_specific_items.empty:
                    # Recalculate value_metric for selection within this context
                    restaurant_specific_items['value_metric'] = restaurant_specific_items.apply(
                        lambda row: row['Calories'] / row['Price'] if pd.notna(row['Price']) and row['Price'] > 0 else 0,
                        axis=1
                    )
                    best_item = restaurant_specific_items.sort_values(by='value_metric', ascending=False).iloc[0]

                    current_order_dishes.append({
                        'Name': best_item['Name'],
                        'Price': best_item['Price'],
                        'Calories': best_item['Calories'],
                        'Protein (g)': best_item['Protein (g)'],
                        'Carbs (g)': best_item['Carbs (g)'],
                        'Fat (g)': best_item['Fat (g)'],
                        'Sugar (g)': best_item['Sugar (g)']
                    })
                    total_price += best_item['Price'] if pd.notna(best_item['Price']) else 0
                    total_calories += best_item['Calories']
                    total_protein += best_item['Protein (g)']
                    total_carbs += best_item['Carbs (g)']
                    total_fat += best_item['Fat (g)']
                    total_sugar += best_item['Sugar (g)']
                else:
                    all_items_found_for_restaurant = False
                    break # Cannot fulfill all item requests for this restaurant

            if all_items_found_for_restaurant and current_order_dishes:
                # Check overall meal calorie constraint if specified
                if parsed_query['max_calories'] is not None and total_calories > parsed_query['max_calories']:
                    continue # Skip this order if it exceeds overall max calories

                # Aggregate dietary notes for the combined order
                dietary_notes = []

                # Re-evaluate dietary compliance for the *selected* dishes
                current_order_df = pd.DataFrame(current_order_dishes)
                # Merge with all_menus_df to get back dietary tags
                merged_dishes = pd.merge(current_order_df, all_menus_df[['Name', 'is_vegetarian', 'is_vegan', 'is_gluten_free', 'is_dairy_free', 'is_nut_free']], on='Name', how='left')

                if 'vegetarian' in parsed_query['dietary_preferences']:
                    if all(merged_dishes['is_vegetarian']):
                         dietary_notes.append('Vegetarian compliant order')
                    else:
                         dietary_notes.append('Some items may not be Vegetarian')

                if 'vegan' in parsed_query['dietary_preferences']:
                    if all(merged_dishes['is_vegan']):
                         dietary_notes.append('Vegan compliant order')
                    else:
                         dietary_notes.append('Some items may not be Vegan')

                if 'gluten' in parsed_query['excluded_ingredients']:
                    if all(merged_dishes['is_gluten_free']): # Check if all selected items are gluten-free
                        dietary_notes.append('Gluten-free compliant order')
                    else:
                        dietary_notes.append('Some items may not be Gluten-free')

                if 'dairy' in parsed_query['excluded_ingredients']:
                    if all(merged_dishes['is_dairy_free']): # Check if all selected items are dairy-free
                        dietary_notes.append('Dairy-free compliant order')
                    else:
                        dietary_notes.append('Some items may not be Dairy-free')

                if 'nuts' in parsed_query['excluded_ingredients']:
                    if all(merged_dishes['is_nut_free']): # Check if all selected items are nut-free
                        dietary_notes.append('Nut-free compliant order')
                    else:
                        dietary_notes.append('Some items may not be Nut-free')

                if 'high-protein' in parsed_query['dietary_preferences']:
                    protein_threshold = all_menus_df['Protein (g)'].quantile(0.75) # Use global threshold
                    if all(merged_dishes['Protein (g)'] >= protein_threshold):
                        dietary_notes.append('High-protein compliant order')
                    else:
                        dietary_notes.append('Some items may not be High-protein')

                for excluded_item in parsed_query['excluded_ingredients']:
                    if excluded_item.lower() in ['seafood', 'soy']:
                        # Check if any selected item name contains the excluded ingredient
                        if not any(excluded_item.lower() in dish['Name'].lower() for dish in current_order_dishes):
                            dietary_notes.append(f'Excluded {excluded_item} (by name check) compliant')
                        else:
                            dietary_notes.append(f'Some items may contain {excluded_item}')

                if not dietary_notes: # Fallback if no specific notes are added
                    dietary_notes.append('No specific dietary compliance noted for this combined order.')

                multi_item_recommendations.append({
                    'Restaurant Name': restaurant_name,
                    'Recommended Dish(es)': current_order_dishes,
                    'Total Price': total_price,
                    'Nutritional Summary': {
                        'Calories': total_calories,
                        'Protein (g)': total_protein,
                        'Carbs (g)': total_carbs,
                        'Fat (g)': total_fat,
                        'Sugar (g)': total_sugar
                    },
                    'Dietary Compliance Notes': dietary_notes,
                    'Availability': 'Available'
                })
        return multi_item_recommendations if multi_item_recommendations else "No recommendations found matching your criteria."

    # --- Single-item request handling (existing logic) ---
    else:
        filtered_items_df = retrieve_menu_items(all_menus_df, parsed_query, model, index)

        if filtered_items_df.empty:
            return "No recommendations found matching your criteria."

        filtered_items_df['value_metric'] = filtered_items_df.apply(
            lambda row: row['Calories'] / row['Price'] if pd.notna(row['Price']) and row['Price'] > 0 else 0,
            axis=1
        )

        for restaurant_name, restaurant_df in filtered_items_df.groupby('Restaurant'):
            restaurant_df = restaurant_df.sort_values(by='value_metric', ascending=False)
            if not restaurant_df.empty:
                top_item = restaurant_df.iloc[0]

                recommended_dish = {
                    'Name': top_item['Name'],
                    'Price': top_item['Price'],
                    'Calories': top_item['Calories'],
                    'Protein (g)': top_item['Protein (g)'],
                    'Carbs (g)': top_item['Carbs (g)'],
                    'Fat (g)': top_item['Fat (g)'],
                    'Sugar (g)': top_item['Sugar (g)']
                }

                dietary_notes = []
                if 'vegetarian' in parsed_query['dietary_preferences'] and top_item.get('is_vegetarian', False):
                    dietary_notes.append('Vegetarian compliant')
                if 'vegan' in parsed_query['dietary_preferences'] and top_item.get('is_vegan', False):
                    dietary_notes.append('Vegan compliant')
                if 'gluten' in parsed_query['excluded_ingredients'] and top_item.get('is_gluten_free', False):
                    dietary_notes.append('Gluten-free compliant')
                if 'dairy' in parsed_query['excluded_ingredients'] and top_item.get('is_dairy_free', False):
                    dietary_notes.append('Dairy-free compliant')
                if 'nuts' in parsed_query['excluded_ingredients'] and top_item.get('is_nut_free', False):
                    dietary_notes.append('Nut-free compliant')
                if 'high-protein' in parsed_query['dietary_preferences']:
                    protein_threshold = all_menus_df['Protein (g)'].quantile(0.75)
                    if top_item['Protein (g)'] >= protein_threshold:
                        dietary_notes.append('High-protein compliant')

                for excluded_item in parsed_query['excluded_ingredients']:
                    if excluded_item.lower() in ['seafood', 'soy'] and not pd.isna(top_item['Name']) and excluded_item.lower() not in top_item['Name'].lower():
                         dietary_notes.append(f'Excluded {excluded_item} (by name check) compliant')

                recommendations.append({
                    'Restaurant Name': restaurant_name,
                    'Recommended Dish(es)': [recommended_dish],
                    'Total Price': recommended_dish['Price'],
                    'Nutritional Summary': recommended_dish,
                    'Dietary Compliance Notes': dietary_notes if dietary_notes else ['No specific dietary compliance noted for this item.'],
                    'Availability': 'Available'
                })
        return recommendations
