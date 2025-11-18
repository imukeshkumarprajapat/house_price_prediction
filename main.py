# from flask import Flask , render_template, request
# import pandas as pd
# import pickle as pickle
# import numpy as np

# app=Flask(__name__)
# data=pd.read_csv("notebook\clean_data.csv")
# pipe=pickle.load(open("notebook\RidgeModel.pkl","rb"))

# @app.route('/')
# def index():
#     locations=sorted(data["location"].unique())

#     return render_template("index.html", locations=locations,prediction=None)



# @app.route('/predict', methods=["POST"])
# def predict():
#     location = request.form.get('location')
#     bhk = request.form.get("bhk")
#     bath = request.form.get("bath")
#     sqft = request.form.get("sqft")  # ✅ match HTML field name

#     print(location, bhk, bath, sqft)

#     try:
#         bhk = float(bhk)
#         bath = float(bath)
#         sqft = float(sqft)
#     except:
#         return "Error: Invalid input values"

#     # ✅ Use correct column names expected by model
#     input_df = pd.DataFrame([[location, sqft, bath, bhk]],
#                             columns=["location", "total_sqft", "bath", "bhk"])

#     print("Any NaN present:", input_df.isnull().values.any())
#     print("NaN count per column:\n", input_df.isnull().sum())

#     if input_df.isnull().values.any():
#         return "Error: Missing values in input"

#     prediction = pipe.predict(input_df)[0] * 1e5
#     prediction = np.round(prediction, 2)

#     locations = sorted(data["location"].unique())
#     return render_template("index.html", locations=locations, prediction=prediction)


# if __name__=="__main__":
#     app.run(debug=True, port=5001)



# import streamlit as st # type: ignore
# import pandas as pd
# import numpy as np
# import pickle

# # Load model and data
# pipe = pickle.load(open("notebook/RidgeModel.pkl", "rb"))
# data = pd.read_csv("notebook/clean_data.csv")
# locations = sorted(data["location"].unique())

# # UI
# st.title("🏡 DreamHome Price Estimator")

# with st.form("prediction_form"):
#     location = st.selectbox("Select Location", locations)
#     bhk = st.number_input("BHK", min_value=1, step=1)
#     bath = st.number_input("Bathrooms", min_value=1, step=1)
#     sqft = st.number_input("Total Square Feet", min_value=100)

#     submitted = st.form_submit_button("Estimate Price")

# if submitted:
#     input_df = pd.DataFrame([[location, sqft, bath, bhk]],
#                             columns=["location", "total_sqft", "bath", "bhk"])

#     if input_df.isnull().values.any():
#         st.error("Missing values in input")
#     else:
#         prediction = pipe.predict(input_df)[0] * 1e5
#         st.success(f"Estimated Price: ₹{np.round(prediction, 2):,.2f}")



######------------------------------------------------#######
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. Streamlit Page Configuration ---
# Use a wide layout and a professional-looking icon
st.set_page_config(
    page_title="DreamHome Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Load Model and Data ---
try:
    # Load model and data
    pipe = pickle.load(open("notebook/RidgeModel.pkl", "rb"))
    data = pd.read_csv("notebook/clean_data.csv")
    locations = sorted(data["location"].unique())
except FileNotFoundError:
    st.error("🚨 Error: Model file (RidgeModel.pkl) or data file (clean_data.csv) not found. Please check the paths.")
    st.stop()
except Exception as e:
    st.error(f"🚨 An unexpected error occurred during loading: {e}")
    st.stop()

# --- 3. Sidebar (For a professional touch) ---
with st.sidebar:
    st.title("🏡 DreamHome Realty")
    st.write("Welcome to the **most trusted** property valuation tool.")
    st.subheader("Navigation")
    st.info("Currently viewing: **Price Estimation**")
    st.write("---")
    st.caption("Powered by Machine Learning")

# --- 4. Main Title and Description ---
st.title("💰 Property Price Estimator - Get Your Instant Valuation")
st.markdown(
    """
    Enter the details of the property below to get an **accurate, data-driven** price estimate. 
    Our model is trained on properties across key areas.
    """
)

st.write("---")

# --- 5. Prediction Form (Using Columns for better layout) ---

# Use columns to make the form look less stacked and more like a real website
col1, col2, col3 = st.columns([1, 2, 1])

with col2: # Center the form inputs in the middle column
    with st.container(border=True):
        st.subheader("Property Details")
        with st.form("prediction_form"):
            # Use st.select_box for location - it's already good
            location = st.selectbox("📍 Select Location", locations, help="Choose the neighborhood of the property.")
            
            # Use columns inside the form for a compact layout
            c1, c2, c3 = st.columns(3)
            with c1:
                bhk = st.number_input("🛏️ BHK (Bedrooms)", min_value=1, step=1, value=2)
            with c2:
                bath = st.number_input("🛁 Bathrooms", min_value=1, step=1, value=2)
            with c3:
                # Add a suffix for clarity (Sq. Ft.)
                sqft = st.number_input("📏 Total Square Feet", min_value=100, step=50, value=1200)

            st.write("") # Add some vertical space
            
            # Use 'primary' type for the submit button to make it stand out
            submitted = st.form_submit_button("✨ Estimate Price Now", type="primary", use_container_width=True)

st.write("---")

# --- 6. Prediction Output ---
if submitted:
    # Prepare input data
    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=["location", "total_sqft", "bath", "bhk"])

    # Basic input validation (already present, but good to keep)
    if input_df.isnull().values.any():
        st.error("🚨 Missing values in input. Please fill all fields.")
    else:
        try:
            # Perform prediction
            prediction = pipe.predict(input_df)[0]
            
            # Ensure prediction is positive (a common check for real estate)
            if prediction < 0:
                 st.warning("⚠️ Prediction resulted in a value that is too low. The details might be outside the model's training range.")
                 st.write("---")
            else:
                # Final price in Lakhs/Crores for better readability (using * 1e5 as per your original code)
                estimated_price = prediction * 1e5 
                
                # --- Display Result ---
                st.subheader("✅ Valuation Complete!")
                st.balloons() # Add a celebratory effect
                
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                    # Use a metric card for high visibility
                    st.metric(
                        label=f"Estimated Price for {location}",
                        value=f"₹{np.round(estimated_price / 1e7, 2):,.2f} Crore",
                        delta="Data-Driven ML Estimate"
                    )
                    st.write(f"*(Approx. ₹{np.round(estimated_price / 1e5, 2):,.2f} Lakhs)*")
                
                with result_col2:
                    st.success(f"""
                        **Details of the Estimate:**
                        * **Location:** {location}
                        * **Area:** {sqft} Sq. Ft.
                        * **BHK/Bath:** {bhk} / {bath}
                        
                        This estimate is provided by a **Ridge Regression Model** trained on historical market data.
                    """)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- 7. Footer (Professional touch) ---
st.markdown("---")
st.caption("© 2025 DreamHome Price Estimator. Disclaimer: This is an estimated price and actual transaction value may vary.")
