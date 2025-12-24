# ===================================================================
# Melbourne Housing Price Prediction Web App
#
# To run this application:
# 1. Make sure 'final_data_sorted.csv' is in the same folder.
# 2. Install necessary libraries: pip install gradio pandas scikit-learn xgboost
# 3. Run from your terminal: python app.py
# ===================================================================

# --- Core Libraries ---
import pandas as pd
import numpy as np
import gradio as gr
import sklearn

# --- Scikit-learn for Preprocessing and Modeling ---
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

print("Starting app setup...")

# --- 1. Load Data and Perform Feature Engineering ---
# This function handles all data preparation steps. It is run only once when the app starts.
def load_and_engineer_features(csv_path: str):
    """Loads the dataset and applies all the feature engineering steps from the notebook."""
    df = pd.read_csv(csv_path)
    
    # Ensure key columns are numeric, converting any non-numeric values to NaN
    for col in ['price', 'latitude', 'longitude', 'postcode']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop any rows where the conversion failed
    df.dropna(subset=['price', 'latitude', 'longitude', 'postcode'], inplace=True)
    
    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    # --- Feature Engineering ---
    # Create date-based features
    df['sale_year'] = df['date'].dt.year
    df['sale_month'] = df['date'].dt.month
    
    # Create location-based feature (distance to CBD)
    cbd_lat, cbd_lon = -37.8136, 144.9631
    df['distance_to_cbd'] = np.sqrt((df['latitude'] - cbd_lat)**2 + (df['longitude'] - cbd_lon)**2)
    
    # Create density feature (rooms per square meter)
    landsize_no_zero = df['landsize'].replace(0, np.nan)
    df['rooms_per_100sqm'] = (df['rooms'] / landsize_no_zero) * 100
    
    # Create advanced target-encoded feature (Leave-One-Out)
    df['yr_qtr'] = df['date'].dt.to_period('Q').astype(str)
    group_key = ['suburb', 'yr_qtr']
    grouped = df.groupby(group_key)['price'].agg(['sum', 'count'])
    df = pd.merge(df, grouped, on=group_key, how='left')
    df['suburb_qtr_mean_price_loo'] = (df['sum'] - df['price']) / (df['count'] - 1)
    yearly_mean_price = df.groupby('sale_year')['price'].transform('mean')
    df['suburb_qtr_mean_price_loo'] = df['suburb_qtr_mean_price_loo'].fillna(yearly_mean_price)
    df = df.drop(columns=['sum', 'count'])
    
    return df

# Load and process the data when the script starts
print("Loading and engineering data...")
df = load_and_engineer_features("final_data_sorted.csv")

# --- 2. Build and Train the Model Pipeline ---
# The model is trained once on the entire dataset to be as accurate as possible.

# Use the log-transformed price as the target variable for better model performance
y = np.log1p(df['price'])
# Define the features (X) by dropping the target and high-cardinality identifiers
X = df.drop(columns=['price', 'seller_g', 'date', 'council_area', 'region_name'])

# Identify categorical and numerical column names for the preprocessor
cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Create a preprocessing pipeline for categorical features (impute then one-hot encode)
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
# Create a preprocessing pipeline for numerical features (impute then standardize)
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

# Combine both pipelines into a single preprocessor object
preprocessor = ColumnTransformer(transformers=[("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)], remainder="drop")

# Define the final model pipeline, chaining the preprocessor and the Gradient Boosting model
final_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", GradientBoostingRegressor(n_estimators=700, learning_rate=0.03, max_depth=5, subsample=0.7, random_state=42))
])

# Train the entire pipeline on the full dataset
print("Training the final model on all data...")
final_pipeline.fit(X, y)
print("Model training complete.")

# --- 3. Prepare Data and Functions for Gradio Interface ---

# Create a dictionary mapping each suburb to its list of valid postcodes (as strings for the UI)
suburb_postcode_map = df.groupby('suburb')['postcode'].unique().apply(lambda x: [str(int(p)) for p in x]).to_dict()

# Get unique values from the dataframe to populate the dropdown menus
suburb_choices = sorted(df['suburb'].unique())
type_choices = sorted(df['type'].unique())
method_choices = sorted(df['method'].unique())

# Pre-calculate mean prices for use in single predictions
loo_means = df.groupby(['suburb', 'yr_qtr'])['price'].mean()
global_mean = df['price'].mean()

# This is the main function that runs every time the user clicks "Predict"
def predict_price(suburb, p_type, method, postcode, rooms, bathroom, car, landsize, distance, bedroom2, latitude, longitude, property_count):
    """Takes all user inputs, applies feature engineering, and returns a formatted price prediction."""
    
    # Convert the postcode from the UI (string) back to a number for the model
    postcode = float(postcode)

    # Create a single-row DataFrame from all the user inputs
    input_data = {
        'suburb': suburb, 'rooms': rooms, 'type': p_type, 'method': method,
        'distance': distance, 'postcode': postcode, 'bedroom2': bedroom2,
        'bathroom': bathroom, 'car': car, 'landsize': landsize,
        'latitude': latitude, 'longitude': longitude, 'property_count': property_count,
        'date': pd.to_datetime('today') # Use today's date to generate year/month features
    }
    input_df = pd.DataFrame([input_data])
    
    # Apply the same feature engineering steps to the single input row
    input_df['sale_year'] = input_df['date'].dt.year
    input_df['sale_month'] = input_df['date'].dt.month
    input_df['distance_to_cbd'] = np.sqrt((input_df['latitude'] - (-37.8136))**2 + (input_df['longitude'] - (144.9631))**2)
    landsize_no_zero = input_df['landsize'].replace(0, np.nan)
    input_df['rooms_per_100sqm'] = (input_df['rooms'] / landsize_no_zero) * 100
    input_df['yr_qtr'] = input_df['date'].dt.to_period('Q').astype(str)
    
    # For the LOO feature, look up the pre-calculated mean for the suburb/quarter
    try:
        loo_val = loo_means.loc[(suburb, input_df['yr_qtr'].iloc[0])]
    except KeyError:
        loo_val = global_mean # If not found, use the global average price as a fallback
    input_df['suburb_qtr_mean_price_loo'] = loo_val
    
    # Use the trained pipeline to predict the price (on the log scale)
    prediction_log = final_pipeline.predict(input_df)[0]
    # Convert the prediction from the log scale back to actual dollars
    prediction_dollar = np.expm1(prediction_log)
    
    # Return the final prediction, formatted as a currency string
    return f"${prediction_dollar:,.0f}"

# This function is triggered whenever the 'Suburb' dropdown changes
def update_postcode_choices(suburb):
    """Updates the postcode radio buttons based on the selected suburb."""
    postcodes_for_suburb = suburb_postcode_map.get(suburb, [])
    # Return a new Radio component with the updated choices
    return gr.Radio(choices=postcodes_for_suburb, value=postcodes_for_suburb[0] if postcodes_for_suburb else None)

# --- 4. Create and Launch the Gradio Interface ---
# gr.Blocks provides a flexible way to design the UI layout
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏡 Melbourne Housing Price Demo")
    gr.Markdown("Enter the details of a property to get a price prediction from our Gradient Boosting model.")
    
    # Use rows and columns to organize the interface
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Property & Sale Details")
            suburb = gr.Dropdown(choices=suburb_choices, label="Suburb", value='Reservoir')
            p_type = gr.Dropdown(choices=type_choices, label="Property Type", value='h')
            method = gr.Dropdown(choices=method_choices, label="Sale Method", value='S')
            
            # Get initial postcode choices for the default suburb
            initial_postcodes = suburb_postcode_map.get('Reservoir', [])
            postcode = gr.Radio(choices=initial_postcodes, label="Postcode", value=initial_postcodes[0] if initial_postcodes else None)
            
            gr.Markdown("### Property Size")
            rooms = gr.Slider(minimum=1, maximum=10, step=1, label="Rooms", value=3)
            bedroom2 = gr.Slider(minimum=1, maximum=10, step=1, label="Bedrooms", value=3)
            bathroom = gr.Slider(minimum=1, maximum=8, step=1, label="Bathrooms", value=1)
            car = gr.Slider(minimum=0, maximum=10, step=1, label="Car Spaces", value=1)
            landsize = gr.Number(label="Land Size (sqm)", value=500)
            
        with gr.Column():
            gr.Markdown("### Location Details")
            distance = gr.Number(label="Distance to CBD (km)", value=10.0)
            property_count = gr.Number(label="Property Count in Suburb", value=21650)
            latitude = gr.Number(label="Latitude", value=-37.71)
            longitude = gr.Number(label="Longitude", value=145.02)
            
            gr.Markdown("---")
            submit_btn = gr.Button("Predict Price", variant="primary")
            
            gr.Markdown("### Predicted Price")
            output_price = gr.Textbox(label="Predicted Price ($AUD)")

    # --- Event Listeners ---
    # This connects the 'suburb' dropdown to the 'postcode' radio buttons.
    # When 'suburb' changes, the 'update_postcode_choices' function is called.
    suburb.change(fn=update_postcode_choices, inputs=suburb, outputs=postcode)
    
    # This connects the 'Predict' button to the main prediction logic.
    # When clicked, the 'predict_price' function is called with all the inputs.
    submit_btn.click(
        fn=predict_price,
        inputs=[suburb, p_type, method, postcode, rooms, bathroom, car, landsize, distance, bedroom2, latitude, longitude, property_count],
        outputs=output_price
    )

# This line launches the web application
if __name__ == "__main__":
    demo.launch()