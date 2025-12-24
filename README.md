# 🏡 Predicting Melbourne Housing Prices

A comprehensive machine learning project that predicts housing prices in Melbourne, Australia using advanced feature engineering and ensemble models. This project includes both a detailed Jupyter notebook for analysis and a user-friendly Gradio web application for interactive predictions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Key Features](#key-features)

## 🎯 Overview

This project implements a machine learning pipeline to predict Melbourne housing prices based on various property features such as location, size, type, and sale method. The solution includes:

- **Data Analysis & Visualization**: Comprehensive exploratory data analysis with multiple visualizations
- **Feature Engineering**: Advanced feature creation including date-based, location-based, and target-encoded features
- **Model Training**: Multiple ensemble models (Random Forest, Gradient Boosting, XGBoost, HistGradientBoosting) with cross-validation
- **Model Evaluation**: Detailed performance metrics and feature importance analysis using SHAP values
- **Web Application**: Interactive Gradio interface for real-time price predictions

## ✨ Features

- **Advanced Feature Engineering**:
  - Date-based features (sale year, month, quarter)
  - Location-based features (distance to CBD)
  - Density features (rooms per 100 sqm)
  - Target-encoded features (Leave-One-Out encoding for suburb-quarter mean prices)

- **Multiple Model Comparison**:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Histogram-based Gradient Boosting
  - XGBoost Regressor

- **Comprehensive Analysis**:
  - Price distribution analysis
  - Correlation analysis
  - Outlier detection
  - Feature importance visualization
  - SHAP value analysis for model interpretability

- **Interactive Web App**:
  - User-friendly interface for property price predictions
  - Real-time predictions based on property features
  - Dynamic postcode selection based on suburb

## 📁 Project Structure

```
Predicting Melbourne Housing Prices/
│
├── sit307 Task 8.2D.ipynb     # Main Jupyter notebook with analysis and modeling
├── app.py                     # Gradio web application for predictions
├── final_data_sorted.csv      # Dataset containing Melbourne housing data
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore file
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Yash-001100/Predicting-Melbourne-Housing-Prices.git
cd Predicting-Melbourne-Housing-Prices
```

### Step 2: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

Alternatively, install packages individually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap gradio
```

## 💻 Usage

### Option 1: Run the Jupyter Notebook

1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook "sit307 Task 8.2D.ipynb"
   ```

2. Run all cells sequentially to:
   - Load and explore the data
   - Perform feature engineering
   - Train and compare multiple models
   - Visualize results and feature importance

### Option 2: Run the Web Application

1. Make sure `final_data_sorted.csv` is in the same directory as `app.py`

2. Run the application:
   ```bash
   python app.py
   ```

3. The application will start a local web server. Open your browser and navigate to the URL shown in the terminal (typically `http://127.0.0.1:7860`)

4. Enter property details in the web interface:
   - **Property & Sale Details**: Suburb, Property Type, Sale Method, Postcode
   - **Property Size**: Rooms, Bedrooms, Bathrooms, Car Spaces, Land Size
   - **Location Details**: Distance to CBD, Property Count, Latitude, Longitude

5. Click "Predict Price" to get an estimated price prediction

## 📊 Model Performance

The project evaluates multiple models using 5-fold cross-validation. The Gradient Boosting Regressor achieved the best performance with the following metrics:

- **Mean Absolute Error (MAE)**: Competitive performance on test set
- **Root Mean Squared Error (RMSE)**: Low prediction error
- **R-squared Score**: High model fit

The final model uses:
- **Algorithm**: Gradient Boosting Regressor
- **Hyperparameters**:
  - n_estimators: 700
  - learning_rate: 0.03
  - max_depth: 5
  - subsample: 0.7

## 🛠️ Technologies Used

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost
- **Model Interpretability**: SHAP (SHapley Additive exPlanations)
- **Web Framework**: Gradio
- **Development Environment**: Jupyter Notebook

## 📈 Dataset

The dataset (`final_data_sorted.csv`) contains 1,659 property records with the following features:

- **Location**: suburb, postcode, latitude, longitude, distance to CBD, council_area, region_name
- **Property Details**: rooms, bedrooms (bedroom2), bathrooms, car spaces, landsize, property type
- **Sale Information**: price, sale method, seller, sale date
- **Derived Features**: sale year, sale month, distance to CBD, rooms per 100 sqm, target-encoded features

## 🔑 Key Features

### Feature Engineering Highlights

1. **Date-based Features**: Extracted year, month, and quarter from sale dates
2. **Geographic Features**: Calculated distance to Melbourne CBD using latitude/longitude
3. **Density Features**: Created rooms per 100 square meters ratio
4. **Target Encoding**: Leave-One-Out encoding for suburb-quarter mean prices to capture location and temporal trends

### Model Interpretability

- **Feature Importance**: Gradient Boosting built-in feature importance
- **SHAP Values**: Detailed analysis of feature contributions to predictions
- **Permutation Importance**: Model-agnostic feature importance scores
- **SHAP Interaction Values**: Understanding feature interactions

## 📝 Notes

- The model uses log-transformed prices for training to handle the skewed price distribution
- Predictions are converted back to original dollar amounts
- The web app uses pre-calculated mean prices for the Leave-One-Out feature
- For best results, use property details similar to those in the training dataset

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

**Yash**

- GitHub: [@Yash-001100](https://github.com/Yash-001100)

## 🙏 Acknowledgments

- Melbourne housing dataset
- scikit-learn and XGBoost communities
- SHAP library for model interpretability
- Gradio for the web interface framework

---

**Happy Predicting! 🏠💰**

