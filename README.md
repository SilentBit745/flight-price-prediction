
# ✈️ Flight Price Prediction

A simple Machine Learning project that predicts flight ticket prices.

## About the Project

This project estimates flight ticket prices based on flight details such as airline, source, destination, journey date, departure time, arrival time, duration and number of stops.

## Features

- Predict flight ticket prices
- Select different airlines
- Choose source and destination
- Enter journey details
- Input validation
- Data analysis dashboard
- Price visualization graphs

## How It Works

1. The project uses a flight price dataset.
2. The data is cleaned and prepared.
3. Three Machine Learning models are trained.
4. Their performance is compared.
5. Random Forest is selected as the best model.
6. The trained model is saved using Joblib.
7. Flask loads the model and displays a prediction on the website.

## Machine Learning Models

- Linear Regression
- Decision Tree
- Random Forest

## Model Results

| Model | MAE | R² Score |
|---|---:|---:|
| Linear Regression | 1777.48 | 0.689 |
| Decision Tree | 678.25 | 0.896 |
| Random Forest | 624.61 | 0.902 |

Random Forest performed best among the three models.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- HTML
- CSS
- Joblib

## Project Structure

- `app.py` – Runs the Flask website.
- `train_model.py` – Trains and compares models.
- `analysis.py` – Creates data visualizations.
- `check_data.py` – Checks the dataset.
- `flight_price_model.pkl` – Saved trained model.
- `dataset/` – Contains the flight dataset.
- `templates/` – Contains HTML pages.
- `static/` – Contains CSS styling.

## How to Run

**1. Install the required packages**

```bash
pip install -r requirements.txt
```

**2. Train the model**

```bash
python train_model.py
```

**3. Start the website**

```bash
python app.py
```

**4. Open in your browser**

http://127.0.0.1:5000

**Dashboard:**

http://127.0.0.1:5000/dashboard

## Model Limitations

- Predictions are estimates, not guaranteed ticket prices.
- Results depend on the training dataset.
- The project does not fetch live airline prices.

## 
