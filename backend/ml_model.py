import os
import sys
from pathlib import Path

# Ensure backend/ is in sys.path for imports to work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from rule_engine import apply_rules
from gpt_llama import explain_performance_with_gpt


class PerformancePredictor:
    def __init__(self):
        self.model = None
        self.features = None
        self.target = 'Performance_Score'

        # Path setup
        script_dir = Path(__file__).parent
        self.data_path = script_dir.parent / 'data' / 'KPI_Dataset.csv'
        self.model_path = script_dir / 'model' / 'performance_predictor.joblib'

        # Create model directory if needed
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def load_data(self):
        """Load raw data without feature engineering"""
        self.df = pd.read_csv(self.data_path)
        if 'Hire_Date' in self.df.columns:
            self.df['Hire_Date'] = pd.to_datetime(self.df['Hire_Date'])

    def train_model(self):
        """Train model using raw features only"""
        self.load_data()

        # Define features to exclude
        exclude_cols = [self.target, 'Employee_ID', 'Employee_Satisfaction_Score']
        self.features = [col for col in self.df.columns if col not in exclude_cols]

        # Separate features and target
        X = self.df[self.features]
        y = self.df[self.target]

        # Manually specify numeric and categorical features
        numeric_features = [
            'Years_At_Company', 'Monthly_Salary', 'Overtime_Hours',
            'Training_Hours', 'Promotions', 'Age', 'Projects_Handled',
            'Team_Size', 'Sick_Days', 'Work_Hours_Per_Week'
        ]

        categorical_features = [
            'Job_Title', 'Department', 'Gender', 'Education_Level',
            'Remote_Work_Frequency', 'Resigned'
        ]

        # Define preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor())
        ])

        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)

    def predict(self, input_data):
        print("📢 predict() function called")

        if not self.model:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print("📦 Model loaded successfully.")
            else:
                raise Exception("❌ Model not trained. Please train it first.")

        # Rule-based logic
        rule_explanations, adjustment = apply_rules(input_data)

        # ML prediction
        input_df = pd.DataFrame([input_data])
        print("🧾 Input DataFrame for prediction:")
        print(input_df)

        try:
            # Optional: Show model input features if needed
            # print("✅ Model expects:", self.model.named_steps['preprocessor'].get_feature_names_out())
            raw_score = float(self.model.predict(input_df)[0])
            print("⚙️ Raw score from model:", raw_score)
        except Exception as e:
            print("❌ Model prediction error:", e)
            raise  # Raise the error to see the root cause
            raw_score = None

        adjusted_score = max(0, min(5, raw_score + adjustment)) if raw_score is not None else 0
        print("🧪 Final adjusted score:", adjusted_score)

        # GPT explanation
        gpt_score, gpt_explanation = explain_performance_with_gpt(input_data)

        return {
            "score": round(gpt_score if gpt_score is not None else adjusted_score, 2),
            "random_forest_score": round(raw_score, 2) if raw_score is not None else None,
            "rule_based_reasons": rule_explanations,
            "gpt_explanation": gpt_explanation
        }
