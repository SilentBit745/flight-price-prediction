
from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import calendar
import os
import math

app = Flask(__name__)

# Load model and dataset
model = joblib.load("flight_price_model.pkl")

data = pd.read_excel("dataset/Data_Train.xlsx")

airlines = sorted(data["Airline"].dropna().unique())
sources = sorted(data["Source"].dropna().unique())
destinations = sorted(data["Destination"].dropna().unique())


@app.route("/", methods=["GET", "POST"])
def home():

    price = None
    error = None

    if request.method == "POST":

        try:
            airline = request.form.get("airline", "").strip()
            source = request.form.get("source", "").strip()
            destination = request.form.get("destination", "").strip()

            day = int(request.form.get("day", "0"))
            month = int(request.form.get("month", "0"))

            dep_hour = int(request.form.get("dep_hour", "-1"))
            arr_hour = int(request.form.get("arr_hour", "-1"))

            duration_hours = int(
                request.form.get("duration_hours", "-1")
            )

            duration_minutes = int(
                request.form.get("duration_minutes", "-1")
            )

            stops = int(request.form.get("stops", "-1"))

            # Convert duration to minutes
            duration = duration_hours * 60 + duration_minutes

            # Validate airline and cities
            if airline not in airlines:
                raise ValueError("Select a valid airline.")

            if source not in sources or destination not in destinations:
                raise ValueError("Select valid cities.")

            if source == destination:
                raise ValueError(
                    "Source and destination must be different."
                )

            # Validate date
            if not 1 <= month <= 12:
                raise ValueError("Enter a valid month.")

            if not 1 <= day <= calendar.monthrange(2024, month)[1]:
                raise ValueError("Enter a valid journey day.")

            # Validate hours
            if not 0 <= dep_hour <= 23:
                raise ValueError("Departure hour must be 0 to 23.")

            if not 0 <= arr_hour <= 23:
                raise ValueError("Arrival hour must be 0 to 23.")

            # Validate duration
            if not 0 <= duration_hours <= 47:
                raise ValueError("Duration hours must be 0 to 47.")

            if not 0 <= duration_minutes <= 59:
                raise ValueError("Duration minutes must be 0 to 59.")

            if not 5 <= duration <= 2860:
                raise ValueError("Enter a valid flight duration.")

            # Validate stops
            if not 0 <= stops <= 4:
                raise ValueError("Stops must be between 0 and 4.")

            # Prepare model input
            values = {
                "Airline": airline,
                "Source": source,
                "Destination": destination,
                "Day": day,
                "Month": month,
                "Dep_Hour": dep_hour,
                "Arr_Hour": arr_hour,
                "Duration_Min": duration,
                "Stops": stops
            }

            # Predict price
            result = float(
                model.predict(pd.DataFrame([values]))[0]
            )

            if not math.isfinite(result):
                raise ValueError("Prediction could not be generated.")

            price = round(max(0, result), 2)

        except (ValueError, TypeError) as e:
            error = str(e)

        except Exception:
            error = "Something went wrong. Please try again."

    return render_template(
        "index.html",
        price=price,
        error=error,
        airlines=airlines,
        sources=sources,
        destinations=destinations
    )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/<filename>")
def graphs(filename):

    allowed = [
        "airline_prices.png",
        "stops_prices.png",
        "price_distribution.png"
    ]

    if filename in allowed and os.path.isfile(filename):
        return send_file(filename)

    return "File not found", 404


if __name__ == "__main__":
    app.run(debug=True)
