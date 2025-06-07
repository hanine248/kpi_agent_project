from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Add this import
from ml_model import PerformancePredictor
from pathlib import Path
import os

app = Flask(__name__, template_folder='../templates')  # Configure template folder
CORS(app)  # Enable CORS for all routes

# Initialize predictor with absolute paths
script_dir = Path(__file__).parent
data_path = script_dir.parent / 'data' / 'KPI_Dataset.csv'
model_path = script_dir / 'model' / 'performance_predictor.joblib'

predictor = PerformancePredictor()

@app.route('/')
def home():
    """Serve the HTML interface"""
    return render_template('index.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    return "HR Performance Prediction API (Simplified Version)"

@app.route('/train', methods=['POST'])
def train():
    """Train the model (call this first)"""
    try:
        predictor.train_model()
        return jsonify({
            "status": "success",
            "message": "Model trained successfully",
            "model_path": str(model_path)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction with raw employee data
    Example JSON input:
    {
        "Department": "IT",
        "Gender": "Male",
        "Age": 35,
        "Job_Title": "Developer",
        "Hire_Date": "2022-01-15",
        "Years_At_Company": 2,
        "Education_Level": "Bachelor",
        "Monthly_Salary": 7000.0,
        "Work_Hours_Per_Week": 40,
        "Projects_Handled": 15,
        "Overtime_Hours": 10,
        "Sick_Days": 2,
        "Remote_Work_Frequency": 50,
        "Team_Size": 8,
        "Training_Hours": 30,
        "Promotions": 1,
        "Resigned": false
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    try:
        data = request.get_json()
        prediction = predictor.predict(data)
        return jsonify({
            "status": "success",
            "predicted_score": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/features', methods=['GET'])
def get_expected_features():
    """Get the list of features the model expects"""
    try:
        return jsonify({
            "expected_features": predictor.features if predictor.features else "Model not trained yet"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print("Starting HR Performance Prediction Server...")
    print("Access the web interface at: http://localhost:5000")
    print("API endpoints available at: http://localhost:5000/api")
    
    app.run(host='0.0.0.0', port=5000, debug=True)