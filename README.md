# House Price Predictor

This is a **house price prediction web application** built with Python and Streamlit.  
It allows users to input house details and get an estimated sale price using a trained XGBoost regression model.

## Features

- Interactive Streamlit UI for entering house features
- Handles both **numerical** and **categorical** features
- Preprocessing pipeline includes: imputation, scaling, encoding, and optional PCA
- Predicts house sale price instantly based on user inputs

## Project Structure
house-price-predictor/
app.py # Streamlit app
house_price_pipeline.pkl # Pickled trained pipeline
price.csv # Original house price dataset
train.csv # Training dataset
test.csv # Testing dataset
new_house_data.csv # Example input for predictions
README.md # Project documentation
requirements.txt # Required Python packages
screenshots/

## How to Run

1. Make sure `house_price_pipeline.pkl` is in the project folder.
2. Install required packages:
3. Run the Streamlit app:
4. Input house details and click **Predict Sale Price**.

## Model Info

- **Model:** XGBoost Regressor
- **Hyperparameters optimized** with `RandomizedSearchCV`
- **Performance on test set:**
  - MAE ≈ 6,600
  - MSE ≈ 74,000,000
  - RMSE ≈ 8,600
  - R² ≈ 0.73

## App Link

You can try the live app here: [House Price Predictor]
<"https://housepredictionmode.streamlit.app/">
