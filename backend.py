# backend.py

def get_recommendation(cuisine: str, max_calories: int, restrictions: str) -> dict:
    """
    Temporary fake backend so we can test the front end.

    Later, we will replace this with your real notebook logic.
    """

    # You can change this text to anything you want for demo vibes.
    return {
        "restaurant_name": "Demo Sushi Place",
        "dish_name": f"{cuisine} tasting platter (demo)",
        "total_calories": min(max_calories, 650),
        "nutritional_breakdown": "Protein 32g, Carbs 48g, Fat 18g (demo data)",
        "notes": "This is placeholder output from backend.py. Next, we’ll wire your real model.",
    }