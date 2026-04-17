Live Customer Churn Prediction Dashboard
A complete end-to-end Machine Learning project to predict customer churn and display results using a Flask web application.
🚀 Project Overview
This project helps businesses identify whether a customer is likely to leave (churn) or stay, based on customer data.
It includes:
Data preprocessing
Exploratory Data Analysis (EDA)
Machine Learning model
Flask web application
REST API for predictions
Docker support
📁 Project Structure
customer-churn-prediction-dashboard/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── data/
│   └── cleaned_telco_churn.csv
├── models/
│   ├── customer_churn_model.pkl
│   ├── selected_features.pkl
│   └── label_encoders.pkl
📊 Dataset
Dataset used: Telco Customer Churn Dataset
Contains customer demographics and service details
Target variable: Churn (Yes / No)
⚙️ Features
✔ Data Cleaning and Preprocessing
✔ EDA before and after cleaning
✔ Feature Selection based on correlation
✔ Machine Learning Model (Random Forest)
✔ Model Evaluation
✔ Flask Web App
✔ REST API for predictions
✔ Docker container support
🤖 Machine Learning Model
Algorithm Used: Random Forest Classifier
Why Random Forest?
Works well with structured data
Handles non-linear relationships
Reduces overfitting
Provides feature importance
🖥️ Run the Project Locally
Step 1: Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction-dashboard.git
cd customer-churn-prediction-dashboard
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Run Flask app
python app.py
Step 4: Open in browser
http://127.0.0.1:5004
📮 API Usage (Postman)
Endpoint
POST /predict
Sample JSON Input
{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 850.0,
  "Contract": "Month-to-month",
  "OnlineSecurity": "No",
  "TechSupport": "No",
  "PaymentMethod": "Electronic check",
  "InternetService": "Fiber optic",
  "PaperlessBilling": "Yes"
}
Sample Output
{
  "prediction": 1,
  "prediction_label": "Yes",
  "churn_probability": 0.82,
  "no_churn_probability": 0.18
}
🐳 Run with Docker
Build Docker image
docker build -t churn-dashboard .
Run container
docker run -p 5004:5004 churn-dashboard
Open
http://localhost:5004
📈 Key Insights
Customers with month-to-month contracts churn more
High monthly charges increase churn probability
Customers without tech support or security are more likely to churn
🔮 Future Improvements
Streamlit dashboard with KPIs and filters
Real-time data integration
Model tuning for better accuracy
Deployment on cloud (Render / AWS)
