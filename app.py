from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
model = joblib.load(BASE_DIR / "model.joblib")

FIELDS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["0", "1"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

NUMERIC_FIELDS = ["tenure", "MonthlyCharges", "TotalCharges"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    data = {}

    if request.method == "POST":
        data = request.form.to_dict()

        try:
            customer = {}

            for field in NUMERIC_FIELDS:
                raw = data.get(field, "").strip()

                if not raw:
                    raise ValueError(f"Please enter {field}.")

                value = float(raw)

                if not pd.notna(value) or value < 0:
                    raise ValueError(f"Enter a valid {field}.")

                if field == "tenure" and (
                    not value.is_integer() or value > 72
                ):
                    raise ValueError(
                        "Tenure must be a whole number from 0 to 72."
                    )

                customer[field] = value

            for field, choices in FIELDS.items():
                value = data.get(field, "")

                if value not in choices:
                    raise ValueError(f"Please select a valid {field}.")

                customer[field] = (
                    int(value) if field == "SeniorCitizen" else value
                )

            columns = list(model.feature_names_in_)
            customer_df = pd.DataFrame(
                [[customer[column] for column in columns]],
                columns=columns
            )

            result = int(model.predict(customer_df)[0])
            prediction = "Churn" if result == 1 else "No Churn"

        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Please check your inputs."

    return render_template(
        "index.html",
        fields=FIELDS,
        numeric_fields=NUMERIC_FIELDS,
        prediction=prediction,
        error=error,
        data=data
    )


if __name__ == "__main__":
    app.run(debug=True)