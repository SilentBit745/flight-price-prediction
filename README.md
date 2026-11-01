
# Flight Price Prediction ✈️

## About
This project predicts flight ticket prices using Machine Learning.

It uses Python, Scikit-learn and Flask.

## Features
- Flight price prediction
- Multiple airline options
- Source and destination selection
- Input validation
- Data analysis dashboard
- Three data visualization graphs

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- HTML
- CSS

## Machine Learning Models
Three models were trained:

1. Linear Regression
2. Decision Tree
3. Random Forest

Random Forest gave the best results.

## Model Results

| Model | MAE | R2 Score |
|---|---:|---:|
| Linear Regression | 1777.48 | 0.689 |
| Decision Tree | 678.25 | 0.896 |
| Random Forest | 624.61 | 0.902 |

## How to Run

Install packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the website:

```bash
python app.py
```

Open:

http://127.0.0.1:5000

Dashboard:

http://127.0.0.1:5000/dashboard

## Project Structure

- app.py – Flask application
- train_model.py – Model training
- analysis.py – Data analysis
- templates/ – HTML pages
- static/ – CSS files
- dataset/ – Dataset files
