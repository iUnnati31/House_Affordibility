import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import folium
from streamlit_folium import folium_static

model = load('best_classification_model.joblib')

def create_features(longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                   population, households, median_income, ocean_proximity):
    rooms_per_household = total_rooms / households if households != 0 else 0
    population_per_household = population / households if households != 0 else 0
    bedrooms_per_room = total_bedrooms / total_rooms if total_rooms != 0 else 0
    
    ocean_categories = ['INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    ocean_encoding = {cat: 1 if cat == ocean_proximity else 0 for cat in ocean_categories}
    
    features = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'rooms_per_household': rooms_per_household,
        'population_per_household': population_per_household,
        'bedrooms_per_room': bedrooms_per_room,
        'ocean_proximity_INLAND': ocean_encoding['INLAND'],
        'ocean_proximity_ISLAND': ocean_encoding['ISLAND'],
        'ocean_proximity_NEAR BAY': ocean_encoding['NEAR BAY'],
        'ocean_proximity_NEAR OCEAN': ocean_encoding['NEAR OCEAN']
    }
    
    return pd.DataFrame([features])

def main():
    st.set_page_config(page_title="House Affordability Genie", layout="wide")
    
    st.title("HOUSE AFFORDABILITY GENIE 🧞")
    st.markdown("""
    Your magical guide to California house affordability! Make a wish by entering the details below.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Location Details")
        longitude = st.slider("Longitude", -124.3, -114.3, -119.5)
        latitude = st.slider("Latitude", 32.5, 42.5, 37.5)
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            options=['INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
        )
        
        st.write("Selected Location:")
        m = folium.Map(location=[latitude, longitude], zoom_start=8)
        folium.Marker([latitude, longitude]).add_to(m)
        folium_static(m, width=400, height=300)
    
    with col2:
        st.subheader("Property Details")
        housing_median_age = st.slider("Housing Median Age", 1, 52, 25)
        total_rooms = st.number_input("Total Rooms", min_value=1, value=100)
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=40)
        population = st.number_input("Population", min_value=1, value=300)
        households = st.number_input("Households", min_value=1, value=100)
        median_income = st.slider("Median Income (in tens of thousands)", 0.0, 15.0, 5.0)
    
    if st.button("✨ Make Your Wish (Predict)", type="primary"):
        features_df = create_features(
            longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
            population, households, median_income, ocean_proximity
        )
        
        proba = model.predict_proba(features_df)[0]
        less_affordable_prob = proba[1]  # Probability of being less affordable
        
        st.markdown("---")
        st.subheader("Your Wish Results 🎯")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if less_affordable_prob >= 0.5:
                st.error("🔴 The Genie Says: Less Affordable 💰")
                confidence = less_affordable_prob
            else:
                st.success("🟢 The Genie Says: More Affordable ✨")
                confidence = 1 - less_affordable_prob
        
        with col4:
            st.write("Genie's Confidence:", f"{confidence * 100:.2f}%")
            
        st.markdown("---")
        st.write("Probability of being less affordable:", f"{less_affordable_prob * 100:.2f}%")
        st.write("Probability of being more affordable:", f"{(1 - less_affordable_prob) * 100:.2f}%")
        
        if hasattr(model, 'feature_importances_'):
            st.subheader("✨ Magic Behind the Prediction")
            importance_df = pd.DataFrame({
                'Feature': features_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(importance_df.set_index('Feature')['Importance'])

if __name__ == "__main__":
    main()