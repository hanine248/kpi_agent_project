import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Ensure backend folder is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_model import PerformancePredictor
from rule_engine import apply_rules



from database import get_all_employees, get_employee_by_id

# Flask app setup
app = Flask(__name__, template_folder='../templates')
CORS(app)

# Paths
script_dir = Path(__file__).parent
data_path = script_dir.parent / 'data' / 'KPI_Dataset.csv'
model_path = script_dir / 'model' / 'performance_predictor.joblib'

# ML predictor
predictor = PerformancePredictor()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api')
def api_info():
    return "HR Performance Prediction API (Simplified Version)"

@app.route('/employees', methods=['GET'])
def list_employees():
    try:
        from database import get_all_employees
        employees = get_all_employees()
        return jsonify(employees)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/employee/<int:emp_id>', methods=['GET'])
def get_employee(emp_id):
    emp = get_employee_by_id(emp_id)
    if not emp:
        return jsonify({"error": "Employee not found"}), 404
    return jsonify(emp)

@app.route('/train', methods=['POST'])
def train():
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
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.get_json()
        result = predictor.predict(data)

        return jsonify({
            "status": "success",
            "predicted_score": result["score"],
            "rule_explanations": result["rule_based_reasons"],
            "gpt_explanation": result["gpt_explanation"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/features', methods=['GET'])
def get_expected_features():
    try:
        return jsonify({
            "expected_features": predictor.features if predictor.features else "Model not trained yet"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print("Starting HR Performance Prediction Server...")
    print("Access the web interface at: http://localhost:5000")
    print("API endpoints available at: http://localhost:5000/api")
    app.run(host='0.0.0.0', port=5000, debug=True)
