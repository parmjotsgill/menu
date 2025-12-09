import streamlit as st
import pandas as pd
import importlib

# Import backend and force reload to ensure latest changes are picked up
import backend
importlib.reload(backend)

from backend import parse_user_query, generate_recommendations, handle_data_query, all_menus_df, model, index

# --- Streamlit UI ---
st.set_page_config(page_title="Menu Recommendation App", layout="wide")
st.title("🍔 Menu Recommendation App")

st.markdown("Enter your dietary preferences, calorie limits, or ask data-oriented questions about the menu.")

user_input_query = st.text_input(
    "Your Query:",
    placeholder="e.g., 'Japanese meal under 750 calories, no seafood or soy', 'I want an entree under 600 calories each and a side, vegetarian', or 'What is the highest calorie item?'"
)

if st.button("Get Suggestions"):
    if user_input_query:
        st.write(f"\nProcessing your query: '{user_input_query}'")
        parsed_output = parse_user_query(user_input_query)

        if parsed_output['data_query']:
            # Handle data-oriented queries
            st.subheader("📊 Data Query Result")
            result = handle_data_query(all_menus_df, parsed_output)
            st.write(result)
        else:
            # Handle recommendation queries (single or multi-item)
            st.subheader("🍽️ Recommendations")

            # Ensure model and index are available for recommendations
            if model is None or index is None:
                st.error("Recommendation engine not initialized. Please ensure menu data is loaded correctly.")
            else:
                final_recommendations = generate_recommendations(all_menus_df, parsed_output, model, index)

                if isinstance(final_recommendations, str):
                    st.warning(final_recommendations)
                elif final_recommendations:
                    for rec in final_recommendations:
                        st.markdown(f"### Restaurant: {rec['Restaurant Name']}")
                        st.write("**Recommended Dish(es):**")
                        for dish in rec['Recommended Dish(es)']:
                            st.markdown(f"- **{dish['Name']}**")
                            st.write(f"  Price: ${dish['Price']:.2f}" if pd.notna(dish['Price']) else "  Price: N/A")
                            st.write(f"  Calories: {dish['Calories']} kcal")
                            st.write(f"  Protein: {dish['Protein (g)']} g")
                            st.write(f"  Carbs: {dish['Carbs (g)']} g")
                            st.write(f"  Fat: {dish['Fat (g)']} g")
                            st.write(f"  Sugar: {dish['Sugar (g)']} g")

                        # Display aggregated nutritional summary for multi-item orders
                        if parsed_output['multi_item_request']:
                            st.write("**Combined Nutritional Summary:**")
                            summary = rec['Nutritional Summary']
                            st.write(f"  Total Calories: {summary['Calories']:.0f} kcal")
                            st.write(f"  Total Protein: {summary['Protein (g)']:.1f} g")
                            st.write(f"  Total Carbs: {summary['Carbs (g)']:.1f} g")
                            st.write(f"  Total Fat: {summary['Fat (g)']:.1f} g")
                            st.write(f"  Total Sugar: {summary['Sugar (g)']:.1f} g")
                            st.write(f"**Total Price: ${rec['Total Price']:.2f}**")

                        st.write(f"**Dietary Notes:** {', '.join(rec['Dietary Compliance Notes'])}")
                        st.markdown("--- ")
                else:
                    st.info("No recommendations found matching your criteria. Try adjusting your query!")
    else:
        st.warning("Please enter a query to get suggestions.")
