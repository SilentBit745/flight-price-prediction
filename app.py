
from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import calendar

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
            # Get form values
            airline = request.form.get("airline", "")
            source = request.form.get("source", "")
            destination = request.form.get("destination", "")

            day = int(request.form.get("day", "0"))
            month = int(request.form.get("month", "0"))
            dep_hour = int(request.form.get("dep_hour", "-1"))
            arr_hour = int(request.form.get("arr_hour", "-1"))
            duration = int(request.form.get("duration", "0"))
            stops = int(request.form.get("stops", "-1"))

            # Validate airline and cities
            if airline not in airlines:
                raise ValueError("Please select a valid airline.")

            if source not in sources:
                raise ValueError("Please select a valid source.")

            if destination not in destinations:
                raise ValueError("Please select a valid destination.")

            if source == destination:
                raise ValueError("Source and destination must be different.")

            # Validate date
            if not 1 <= month <= 12:
                raise ValueError("Month must be between 1 and 12.")

            if not 1 <= day <= calendar.monthrange(2024, month)[1]:
                raise ValueError("Please enter a valid day for the selected month.")

            # Validate hours
            if not 0 <= dep_hour <= 23:
                raise ValueError("Departure hour must be between 0 and 23.")

            if not 0 <= arr_hour <= 23:
                raise ValueError("Arrival hour must be between 0 and 23.")

            # Validate duration
            if not 1 <= duration <= 3000:
                raise ValueError("Duration must be between 1 and 3000 minutes.")

            # Validate stops
            if not 0 <= stops <= 4:
                raise ValueError("Number of stops must be between 0 and 4.")

            # Prepare model input
            values = {
                "Airline": airline,
                "Source": source,
                "Destination": destination,
                "Additional_Info": "No info",
                "Day": day,
                "Month": month,
                "Dep_Hour": dep_hour,
                "Dep_Min": 0,
                "Arr_Hour": arr_hour,
                "Arr_Min": 0,
                "Duration_Min": duration,
                "Stops": stops
            }

            # Predict price
            result = model.predict(pd.DataFrame([values]))[0]
            price = round(max(0, result), 2)

        except (ValueError, TypeError):
            error = "Please enter valid flight details."

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

    if filename in allowed:
        return send_file(filename)

    return "File not found", 404


if __name__ == "__main__":
    app.run(debug=True)
