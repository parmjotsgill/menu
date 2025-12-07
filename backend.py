# backend.py

# TODO: add your real imports from the notebook here
# e.g.
# import openai
# from langchain_xxx import YYY

def get_recommendation(cuisine: str, max_calories: int, restrictions: str) -> dict:
    """
    Main function the front-end will call.

    Inputs:
      cuisine: e.g. "Japanese"
      max_calories: e.g. 750
      restrictions: e.g. "no shellfish, gluten-free"

    Output:
      a dict with keys:
        - restaurant_name
        - dish_name
        - total_calories
        - nutritional_breakdown
        - notes
    """

    # 1. Build a prompt or input string
    user_prompt = (
        f"I want {cuisine} food under {max_calories} calories. "
        f"My dietary restrictions are: {restrictions}."
    )

    # 2. TODO: Replace this fake result with your REAL model call
    # For now this is just a placeholder so the app runs.
    # later we will paste in your notebook logic.
    result = {
        "restaurant_name": "Demo Sushi Place",
        "dish_name": "Salmon Avocado Roll + Miso Soup",
        "total_calories": 540,
        "nutritional_breakdown": "Protein 32g, Carbs 48g, Fat 18g",
        "notes": "Try swapping tempura for steamed veggies to save ~120 calories.",
    }

    return result