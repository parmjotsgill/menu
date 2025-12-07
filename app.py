import streamlit as st
from backend import get_recommendation  # uses the function you just created

st.set_page_config(
    page_title="Calorie-Constrained Dining Assistant",
    page_icon="🍽️",
    layout="centered",
)

# ---- HEADER ----
st.title("🍽️ Calorie-Constrained Dining Assistant")
st.write(
    "Tell me what you're craving, your calorie budget, and any dietary restrictions — "
    "I’ll suggest a restaurant order that fits your goals."
)

st.markdown("---")

# ---- INPUT FORM ----
with st.form("meal_form"):
    col1, col2 = st.columns(2)

    with col1:
        cuisine = st.text_input(
            "What are you in the mood for?",
            placeholder="e.g., Japanese, Italian, Mexican"
        )

    with col2:
        max_calories = st.number_input(
            "Max calories for this meal",
            min_value=100,
            max_value=3000,
            value=750,
            step=50
        )

    restrictions = st.text_area(
        "Dietary needs or preferences",
        placeholder="e.g., no seafood, vegan, gluten-free, no soy"
    )

    submitted = st.form_submit_button("✨ Find my meal")

# ---- RUN BACKEND ----
if submitted:
    if not cuisine:
        st.error("Please tell me at least a cuisine or type of food.")
    else:
        with st.spinner("Thinking about the tastiest options that fit your goals..."):
            try:
                result = get_recommendation(
                    cuisine=cuisine,
                    max_calories=int(max_calories),
                    restrictions=restrictions,
                )

                st.markdown("## ✅ Recommendation")

                st.markdown(
                    f"**Restaurant:** {result.get('restaurant_name', 'Unknown')}"
                )
                st.markdown(
                    f"**Order:** {result.get('dish_name', 'No dish name provided')}"
                )
                st.markdown(
                    f"**Estimated Calories:** {result.get('total_calories', 'N/A')}"
                )

                st.markdown("### 🧾 Nutritional Breakdown")
                st.write(result.get("nutritional_breakdown", "No breakdown available."))

                notes = result.get("notes")
                if notes:
                    st.markdown("### 💡 Suggested Tweaks")
                    st.write(notes)

            except Exception as e:
                st.error("Something went wrong. Show this error to your teammate or instructor.")
                st.exception(e)