# ============================================
# AI HOUSE PRICE PREDICTOR - PREMIUM VERSION
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import sys

if not st.runtime.exists():
    sys.exit(
        "This script must be run with Streamlit.\n"
        "Use:\n    streamlit run d:/Python/SY_25/1st.py"
    )

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>

/* FULL APP BACKGROUND */

html, body, [class*="css"]  {
    background: linear-gradient(135deg,#ffe0c3,#ffd6e8,#fff5f5);
    color: black;
}

/* STREAMLIT MAIN CONTAINER */

[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#ffe0c3,#ffd6e8,#fff5f5);
}

/* MAIN CONTENT */

.main .block-container{
    padding-top:2rem;
    background: transparent;
}

/* SIDEBAR */

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#fff1f2,#ffe4e6);
}

/* HERO SECTION */

.hero{
    padding:45px;
    border-radius:25px;
    background: linear-gradient(135deg,#ff6a00,#ee0979);
    box-shadow:0 10px 30px rgba(0,0,0,0.15);
    margin-bottom:30px;
}

/* TITLE */

.main-title{
    text-align:center;
    font-size:60px;
    font-weight:bold;
    color:white;
}

/* SUBTITLE */

.subtitle{
    text-align:center;
    font-size:22px;
    color:white;
}

/* GLASS CARDS */

.card{
    background: rgba(255,255,255,0.6);
    padding:20px;
    border-radius:20px;
    backdrop-filter: blur(10px);
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

/* METRICS */

[data-testid="stMetric"]{
    background: rgba(255,255,255,0.7);
    padding:20px;
    border-radius:18px;
    border:1px solid rgba(0,0,0,0.05);
}

/* BUTTON */

div.stButton > button{
    width:100%;
    background: linear-gradient(90deg,#ff6a00,#ff006e);
    color:white;
    border:none;
    border-radius:14px;
    height:55px;
    font-size:18px;
    font-weight:bold;
}

/* BUTTON HOVER */

div.stButton > button:hover{
    transform:scale(1.02);
}

/* REMOVE STREAMLIT FOOTER */

footer{
    visibility:hidden;
}

/* MOBILE RESPONSIVE */

@media (max-width: 768px){

    .main-title{
        font-size:38px;
    }

    .subtitle{
        font-size:16px;
    }

    .hero{
        padding:25px;
    }

}

</style>
""", unsafe_allow_html=True)



# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🏠 Maharashtra Real Estate AI")

st.sidebar.markdown("""
### Smart Property Prediction System

✅ AI Powered  
✅ Interactive Dashboard  
✅ Real Estate Analytics  
✅ Machine Learning Model  

---
""")

page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "📊 Dashboard", "💰 Prediction", "📍 Top Districts", "ℹ About"]
)

# ============================================
# DATA GENERATION
# ============================================

@st.cache_data
def generate_data():

    districts = [
        "Mumbai",
        "Pune",
        "Nagpur",
        "Nashik",
        "Solapur",
        "Thane",
        "Kolhapur",
        "Aurangabad",
        "Satara",
        "Sangli",
        "Amravati",
        "Jalgaon",
        "Latur",
        "Ahmednagar",
        "Nanded"
    ]

    base_price = {
        "Mumbai":28000,
        "Pune":12000,
        "Nagpur":7000,
        "Nashik":7500,
        "Solapur":5000,
        "Thane":16000,
        "Kolhapur":6000,
        "Aurangabad":6500,
        "Satara":5500,
        "Sangli":5800,
        "Amravati":5200,
        "Jalgaon":5000,
        "Latur":4700,
        "Ahmednagar":6200,
        "Nanded":4800
    }

    np.random.seed(42)

    n = 1000

    data = {
        "District": np.random.choice(districts,n),
        "BHK": np.random.randint(1,6,n),
        "SquareFeet": np.random.randint(400,4000,n),
        "Bathrooms": np.random.randint(1,5,n),
        "Age": np.random.randint(0,30,n)
    }

    df = pd.DataFrame(data)

    prices = []

    for i in range(n):

        district = df["District"][i]
        sqft = df["SquareFeet"][i]
        bhk = df["BHK"][i]
        age = df["Age"][i]

        price = sqft * base_price[district]

        if bhk >= 3:
            price *= 1.18

        if age > 15:
            price *= 0.90

        price *= np.random.uniform(0.9,1.1)

        prices.append(price)

    df["Price"] = prices

    return df

df = generate_data()

# ============================================
# MODEL
# ============================================

X = pd.get_dummies(df.drop("Price", axis=1))
y = df["Price"]

@st.cache_resource
def train_model():

    model = RandomForestRegressor(
        n_estimators=25,
        max_depth=12,
        random_state=42
    )

    model.fit(X, y)

    return model

model = train_model()

predictions = model.predict(X)

accuracy = r2_score(y, predictions)
error = mean_absolute_error(y, predictions)

# ============================================
# HOME PAGE
# ============================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1 class="main-title">🏠 AI House Price Predictor</h1>
        <p class="subtitle">
        Smart Maharashtra Real Estate Prediction using Artificial Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🚀 Platform Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🏘 Listings", f"{len(df)}")
    c2.metric("📍 Districts", df["District"].nunique())
    c3.metric("🤖 Accuracy", f"{accuracy*100:.1f}%")
    c4.metric("💰 Avg Price", f"₹{int(df['Price'].mean()):,}")

    st.markdown("## 📈 Market Insights")

    st.success("""
    ✅ Mumbai has highest real estate prices  
    ✅ Pune & Thane are fast growing markets  
    ✅ Bigger homes increase value significantly  
    ✅ Older properties reduce pricing  
    """)

# ============================================
# DASHBOARD PAGE
# ============================================

elif page == "📊 Dashboard":

    st.title("📊 Maharashtra Property Analytics")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            df.groupby("District")["Price"].mean().reset_index(),
            x="District",
            y="Price",
            color="District",
            title="Average Price by District"
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(fig, width='stretch')

    with col2:

        fig2 = px.pie(
            df,
            names="District",
            title="Property Distribution"
        )

        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )

        st.plotly_chart(fig2, width='stretch')

# ============================================
# PREDICTION PAGE
# ============================================

elif page == "💰 Prediction":

    st.title("💰 Predict Property Price")

    col1, col2 = st.columns(2)

    with col1:

        district = st.selectbox(
            "📍 District",
            sorted(df["District"].unique())
        )

        bhk = st.slider("🛏 BHK",1,5,2)

        sqft = st.number_input(
            "📐 Square Feet",
            400,
            4000,
            1200
        )

    with col2:

        bathrooms = st.slider("🚿 Bathrooms",1,5,2)

        age = st.slider("🏢 Property Age",0,30,5)

    if st.button("🚀 Predict Price"):

        with st.spinner("🤖 AI analyzing property..."):
            time.sleep(1)

        input_data = {
            "BHK": bhk,
            "SquareFeet": sqft,
            "Bathrooms": bathrooms,
            "Age": age
        }

        for col in X.columns:

            if col.startswith("District_"):

                input_data[col] = 1 if col == f"District_{district}" else 0

        input_df = pd.DataFrame([input_data])

        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(input_df)[0]

        st.markdown(f"""
        <div style="
            background:linear-gradient(90deg,#06b6d4,#3b82f6);
            padding:35px;
            border-radius:20px;
            text-align:center;
            color:white;
            font-size:38px;
            font-weight:bold;
            margin-top:20px;
        ">
            🏠 Estimated Property Price <br>
            ₹ {prediction:,.0f}
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Predicted Price"},
            gauge={
                'axis': {'range': [None, 100000000]}
            }
        ))

        st.plotly_chart(fig, width='stretch')

# ============================================
# TOP DISTRICTS PAGE
# ============================================

elif page == "📍 Top Districts":

    st.title("📍 Top Maharashtra Districts")

    top_df = (
        df.groupby("District")["Price"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.dataframe(top_df, width='stretch')

# ============================================
# ABOUT PAGE
# ============================================

elif page == "ℹ About":

    st.title("ℹ About Project")

    st.markdown("""
    ## 🏠 AI House Price Predictor

    This project predicts Maharashtra property prices using Machine Learning.

    ### 🔧 Technologies
    - Python
    - Streamlit
    - Plotly
    - Scikit-Learn

    ### 🤖 Machine Learning
    Random Forest Regressor

    ### 📊 Features
    - Modern Premium UI
    - Fast Predictions
    - Maharashtra District Analytics
    - Interactive Charts
    - Mobile Friendly Dashboard

    ---
    Made with ❤️ by Mehjabeen
    """)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<hr>
<center>
🏠 AI House Price Predictor <br>
Made using Python, Streamlit & Machine Learning
</center>
""", unsafe_allow_html=True)
