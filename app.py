import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="🏠 California Housing Price Predictor", layout="wide")

# ----------------------------
# 💛 Custom CSS Styling
# ----------------------------
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1516663713099-37eb6d60c825?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Overlay for text readability */
    .main::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.65);
        z-index: -1;
    }

    /* White text for labels/titles */
    .stMarkdown, .stTitle, .stSlider label {
        color: white !important;
    }

    /* Slider: yellow track (line), thumb (circle), number values */
    input[type="range"]::-webkit-slider-runnable-track {
        background: #f1c40f !important;
        height: 6px;
        border-radius: 3px;
    }

    input[type="range"]::-webkit-slider-thumb {
        background-color: #f1c40f !important;
        border: 2px solid white;
        height: 20px;
        width: 20px;
        margin-top: -7px;
        border-radius: 50%;
    }

    input[type="range"]::-moz-range-track {
        background: #f1c40f !important;
        height: 6px;
    }

    input[type="range"]::-moz-range-thumb {
        background-color: #f1c40f !important;
        border: 2px solid white;
        border-radius: 50%;
    }

    .stSlider .css-1y4p8pa, .stSlider .css-1n76uvr, .stSlider .css-1xtw4zc {
        color: #f1c40f !important;
        font-weight: bold;
    }

    /* Yellow button */
    .stButton button {
        background-color: #f1c40f;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 26px;
        font-size: 16px;
    }

    .stButton button:hover {
        background-color: #e1b90e;
    }

    /* Result box */
    .prediction-box {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        color: #2d3436;
        box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
        margin-top: 20px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# 🎯 Load Model
# ----------------------------
model = joblib.load("best_model.pkl")

# ----------------------------
# 📘 Sidebar
# ----------------------------
st.sidebar.title("📘 About Project")
st.sidebar.markdown("""
### 🏠 California Housing Price Predictor  
This ML app predicts **median house prices** based on:
- Median income
- House age
- Avg rooms & bedrooms
- Population density
- Location (Lat/Long)

### 🔍 How It Works  
1. Adjust sliders to input values  
2. Click **Predict**  
3. Get estimated house value!

### 🛠️ Tech Stack  
- `scikit-learn`  
- `Streamlit`  
- `pandas`, `numpy`
""")

# ----------------------------
# 🧠 Title
# ----------------------------
st.title("🏡 California Housing Price Predictor")
st.markdown("Use the sliders below to estimate the **median value of a house** in California based on real data.")

# ----------------------------
# 🎛️ Input Sliders
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    MedInc = st.slider("Median Income", 0.0, 20.0, 5.0, step=0.1)
    HouseAge = st.slider("House Age", 1, 50, 20)

with col2:
    AveRooms = st.slider("Avg Rooms", 1.0, 10.0, 5.0)
    AveBedrms = st.slider("Avg Bedrooms", 0.5, 5.0, 1.0)

with col3:
    Population = st.slider("Population", 100.0, 3000.0, 1000.0)
    AveOccup = st.slider("Avg Occupants", 0.5, 5.0, 3.0)

Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

# ----------------------------
# 🧮 Feature Engineering
# ----------------------------
bedrooms_per_room = AveBedrms / AveRooms
income_per_room = MedInc / AveRooms
occupancy_per_room = AveOccup / AveRooms

input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population,
                        AveOccup, Latitude, Longitude,
                        bedrooms_per_room, income_per_room, occupancy_per_room]])

# ----------------------------
# 🚀 Predict Button
# ----------------------------
if st.button("💡 Predict House Price"):
    prediction = model.predict(input_data)
    st.markdown(
        f"""
        <div class="prediction-box">
            🎯 Estimated Median House Price: <br>
            <span style='font-size: 26px;'>${prediction[0] * 100000:,.2f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# 🔻 Footer
# ----------------------------
st.markdown("---")
st.markdown("Made with ❤️ by **Muhammad Talha**", unsafe_allow_html=True)
