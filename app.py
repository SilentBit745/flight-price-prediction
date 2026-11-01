
from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd

app = Flask(__name__)

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
            values = {
                "Airline": request.form["airline"],
                "Source": request.form["source"],
                "Destination": request.form["destination"],
                "Additional_Info": "No info",
                "Day": int(request.form["day"]),
                "Month": int(request.form["month"]),
                "Dep_Hour": int(request.form["dep_hour"]),
                "Dep_Min": 0,
                "Arr_Hour": int(request.form["arr_hour"]),
                "Arr_Min": 0,
                "Duration_Min": int(request.form["duration"]),
                "Stops": int(request.form["stops"])
            }

            if not 1 <= values["Day"] <= 31:
                raise ValueError("Day must be 1 to 31")

            if not 1 <= values["Month"] <= 12:
                raise ValueError("Month must be 1 to 12")

            if not 0 <= values["Dep_Hour"] <= 23:
                raise ValueError("Departure hour must be 0 to 23")

            if not 0 <= values["Arr_Hour"] <= 23:
                raise ValueError("Arrival hour must be 0 to 23")

            if values["Duration_Min"] <= 0:
                raise ValueError("Duration must be greater than 0")

            if values["Source"] == values["Destination"]:
                raise ValueError("Source and destination must be different")

            if values["Airline"] not in airlines:
                raise ValueError("Invalid airline")

            if values["Source"] not in sources:
                raise ValueError("Invalid source")

            if values["Destination"] not in destinations:
                raise ValueError("Invalid destination")

            if not 0 <= values["Stops"] <= 4:
                raise ValueError("Invalid number of stops")

            result = model.predict(pd.DataFrame([values]))[0]
            price = round(max(0, result), 2)

        except ValueError as e:
            error = str(e)

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
