import sys
import os
import hashlib
from pathlib import Path
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from database import get_connection, get_all_employees, get_employee_by_id
from ml_model import PerformancePredictor
from rule_engine import apply_rules

# Ensure backend folder is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Flask app setup
app = Flask(__name__, template_folder='../templates')
app.secret_key = 'your-very-secret-key'
CORS(app)

# Paths
script_dir = Path(__file__).parent
data_path = script_dir.parent / 'data' / 'KPI_Dataset.csv'
model_path = script_dir / 'model' / 'performance_predictor.joblib'

# ML predictor
predictor = PerformancePredictor()

# === ROUTES ===

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api')
def api_info():
    return "HR Performance Prediction API (Simplified Version)"

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM HR_Users WHERE username = ?", (username,))
    row = cursor.fetchone()

    if row:
        stored_hash = row[0]
        input_hash = hashlib.sha256(password.encode()).hexdigest()

        if input_hash == stored_hash:
            session["user"] = username
            return jsonify({"status": "success", "message": "Login successful"})

    return jsonify({"status": "error", "message": "Invalid credentials"}), 401


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
    "gpt_explanation": result["gpt_explanation"],
    "random_forest_score": result["random_forest_score"]  # ✅ fixed
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
