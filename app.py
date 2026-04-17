from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ============================================
# LOAD MODEL AND FEATURES
# ============================================

model_path = "models/customer_churn_model.pkl"
features_path = "models/selected_features.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not os.path.exists(features_path):
    raise FileNotFoundError(f"Features file not found: {features_path}")

model = joblib.load(model_path)
selected_features = joblib.load(features_path)

# ============================================
# SIMPLE ENCODING MAP
# ============================================

encoding_maps = {
    "Contract": {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    },
    "OnlineSecurity": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2
    },
    "TechSupport": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2
    },
    "PaymentMethod": {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    },
    "InternetService": {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2
    },
    "PaperlessBilling": {
        "No": 0,
        "Yes": 1
    }
}

# ============================================
# HTML PAGE
# ============================================

html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f6f8;
            padding: 30px;
        }
        .container {
            max-width: 700px;
            margin: auto;
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px lightgray;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        label {
            font-weight: bold;
            display: block;
            margin-top: 12px;
        }
        input, select {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border: 1px solid #ccc;
            border-radius: 6px;
        }
        button {
            margin-top: 20px;
            width: 100%;
            padding: 12px;
            background-color: #2d89ef;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background-color: #1b5fbf;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #eef7ee;
            border-left: 5px solid green;
            border-radius: 6px;
        }
        .error {
            margin-top: 20px;
            padding: 15px;
            background-color: #fdeeee;
            border-left: 5px solid red;
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Live Customer Churn Prediction</h1>

        <form method="POST" action="/predict-form">

            <label>Tenure</label>
            <input type="number" name="tenure" required>

            <label>Monthly Charges</label>
            <input type="number" step="0.01" name="MonthlyCharges" required>

            <label>Total Charges</label>
            <input type="number" step="0.01" name="TotalCharges" required>

            <label>Contract</label>
            <select name="Contract" required>
                <option value="">Select</option>
                <option>Month-to-month</option>
                <option>One year</option>
                <option>Two year</option>
            </select>

            <label>Online Security</label>
            <select name="OnlineSecurity" required>
                <option value="">Select</option>
                <option>No</option>
                <option>No internet service</option>
                <option>Yes</option>
            </select>

            <label>Tech Support</label>
            <select name="TechSupport" required>
                <option value="">Select</option>
                <option>No</option>
                <option>No internet service</option>
                <option>Yes</option>
            </select>

            <label>Payment Method</label>
            <select name="PaymentMethod" required>
                <option value="">Select</option>
                <option>Bank transfer (automatic)</option>
                <option>Credit card (automatic)</option>
                <option>Electronic check</option>
                <option>Mailed check</option>
            </select>

            <label>Internet Service</label>
            <select name="InternetService" required>
                <option value="">Select</option>
                <option>DSL</option>
                <option>Fiber optic</option>
                <option>No</option>
            </select>

            <label>Paperless Billing</label>
            <select name="PaperlessBilling" required>
                <option value="">Select</option>
                <option>No</option>
                <option>Yes</option>
            </select>

            <button type="submit">Predict</button>
        </form>

        {% if prediction_text %}
        <div class="result">
            <h3>Prediction Result</h3>
            <p><strong>{{ prediction_text }}</strong></p>
            <p>Churn Probability: {{ churn_probability }}%</p>
            <p>No Churn Probability: {{ no_churn_probability }}%</p>
        </div>
        {% endif %}

        {% if error %}
        <div class="error">
            <h3>Error</h3>
            <p>{{ error }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# ============================================
# PREPARE INPUT DATA
# ============================================

def prepare_input(data):
    input_data = {
        "tenure": float(data["tenure"]),
        "MonthlyCharges": float(data["MonthlyCharges"]),
        "TotalCharges": float(data["TotalCharges"]),
        "Contract": encoding_maps["Contract"][data["Contract"]],
        "OnlineSecurity": encoding_maps["OnlineSecurity"][data["OnlineSecurity"]],
        "TechSupport": encoding_maps["TechSupport"][data["TechSupport"]],
        "PaymentMethod": encoding_maps["PaymentMethod"][data["PaymentMethod"]],
        "InternetService": encoding_maps["InternetService"][data["InternetService"]],
        "PaperlessBilling": encoding_maps["PaperlessBilling"][data["PaperlessBilling"]]
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[selected_features]

    return input_df

# ============================================
# HOME PAGE
# ============================================

@app.route("/")
def home():
    return render_template_string(html_page)

# ============================================
# FORM PREDICTION
# ============================================

@app.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        form_data = {
            "tenure": request.form["tenure"],
            "MonthlyCharges": request.form["MonthlyCharges"],
            "TotalCharges": request.form["TotalCharges"],
            "Contract": request.form["Contract"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "TechSupport": request.form["TechSupport"],
            "PaymentMethod": request.form["PaymentMethod"],
            "InternetService": request.form["InternetService"],
            "PaperlessBilling": request.form["PaperlessBilling"]
        }

        input_df = prepare_input(form_data)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        churn_probability = round(probabilities[1] * 100, 2)
        no_churn_probability = round(probabilities[0] * 100, 2)

        if prediction == 1:
            prediction_text = "Customer is likely to churn"
        else:
            prediction_text = "Customer is likely to stay"

        return render_template_string(
            html_page,
            prediction_text=prediction_text,
            churn_probability=churn_probability,
            no_churn_probability=no_churn_probability
        )

    except Exception as e:
        return render_template_string(html_page, error=str(e))

# ============================================
# API FOR POSTMAN
# ============================================

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        input_df = prepare_input(data)

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        return jsonify({
            "prediction": int(prediction),
            "prediction_label": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probabilities[1]), 4),
            "no_churn_probability": round(float(probabilities[0]), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    print("Flask app running at http://127.0.0.1:5004")
    app.run(debug=True, host="0.0.0.0", port=5004)