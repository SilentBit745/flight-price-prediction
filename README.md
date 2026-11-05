# Flight Price Prediction

A simple Machine Learning project to predict flight ticket prices.

## 🌐 Live Demo

**Website:** https://flight-price-prediction-ivuy.onrender.com

## About

This project predicts flight prices using airline, route, date, time, duration, and number of stops.

## Features

- Predict flight prices
- Select airline and route
- Enter journey details
- Check input values
- View data analysis and graphs
- Use the website online

## How It Works

1. Load the flight dataset.
2. Clean and prepare the data.
3. Train three ML models.
4. Compare their results.
5. Select the best model.
6. Save the model using Pickle.
7. Use Flask to show predictions.

## ML Models

- Linear Regression
- Decision Tree
- Random Forest

## Model Results

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Linear Regression | 2,022.64 | 2,977.74 | 0.575 |
| Decision Tree | 1,325.27 | 2,224.17 | 0.763 |
| Random Forest | **1,174.39** | **1,900.80** | **0.827** |

### Cross-Validation Results

| Model | CV MAE (₹) |
|---|---:|
| Linear Regression | 2,003.08 |
| Decision Tree | 1,386.98 |
| Random Forest | **1,238.05** |

**Random Forest** performed best.

## Dataset

[Flight Fare Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh)

- Training rows: 10,683
- Test rows: 2,671
- Duplicate rows removed: 220

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Flask
- HTML and CSS
- Git and GitHub
- Render

## Project Structure

- `app.py` – Runs the website.
- `train_model.py` – Trains and compares models.
- `analysis.py` – Creates graphs.
- `check_data.py` – Checks the dataset.
- `flight_price_model.pkl` – Saved model.
- `model_comparison.csv` – Model results.
- `dataset/` – Dataset files.
- `templates/` – HTML files.
- `static/` – CSS and graphs.

## How to Run

### 1. Install Packages

```bash
pip install -r requirements.txt