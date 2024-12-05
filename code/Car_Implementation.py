import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset (replace with actual dataset path)
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(
            r"C:\Users\HP\Documents\Car Price Prediction Project\Car Price Dataset.csv",
            on_bad_lines="skip",
            sep=",",
            encoding="utf-8"
        )
        # Drop rows with missing values
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return pd.DataFrame()

# Load data
data = load_data()

# Check if the dataset is loaded successfully
if data.empty:
    st.error("The dataset could not be loaded. Please check the file path or data format.")
    st.stop()

# Separate features and target
try:
    X = data.drop(columns=['selling_price'])
    Y = data['selling_price']
except KeyError as e:
    st.error(f"Column missing in dataset: {e}")
    st.stop()

# Streamlit app
st.title("Car Price Prediction")
st.write("This app predicts car prices using Linear Regression and Lasso Regression.")

# Sidebar for user inputs
st.sidebar.header("Car Features")
name = st.sidebar.selectbox("Name", options=X['name'].unique())
year = st.sidebar.number_input("Year of Manufacture", min_value=2000, max_value=2024, step=1, value=2010)
fuel = st.sidebar.selectbox("Fuel Type", options=X['fuel'].unique())
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=1000, value=50000)
model_type = st.sidebar.selectbox("Model", options=['Linear Regression', 'Lasso Regression'])

# Prepare data for model
categorical_columns = ['name', 'fuel']
numeric_columns = ['year', 'km_driven']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numeric_columns)
    ]
)

# Split data
try:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
except ValueError as e:
    st.error(f"Data splitting failed: {e}")
    st.stop()

# Initialize models
lin_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

lasso_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1))
])

# Train models
try:
    lin_reg.fit(X_train, Y_train)
    lasso_reg.fit(X_train, Y_train)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

# Predict based on user input
user_input = pd.DataFrame({
    'name': [name],
    'year': [year],
    'fuel': [fuel],
    'km_driven': [km_driven]
})

try:
    # Preprocess user input to match training data format
    if st.button("Predict Price"):
        if model_type == 'Linear Regression':
            prediction = lin_reg.predict(user_input)
        elif model_type == 'Lasso Regression':
            prediction = lasso_reg.predict(user_input)
        
        st.write(f"### Predicted Price: ₹{int(prediction[0]):,}")
except Exception as e:
    st.error(f"Prediction failed: {e}")
