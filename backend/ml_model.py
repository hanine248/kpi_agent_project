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

        # Preprocessing
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns

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
        if not self.model:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                raise Exception("Model not trained. Please train it first.")

        rule_explanations, adjustment = apply_rules(input_data)

        input_df = pd.DataFrame([input_data])
        raw_score = float(self.model.predict(input_df)[0])
        adjusted_score = max(0, min(5, raw_score + adjustment))

        gpt_score, gpt_explanation = explain_performance_with_gpt(input_data)

        return {
            "score": round(gpt_score if gpt_score is not None else adjusted_score, 2),
            "rule_based_reasons": rule_explanations,
            "gpt_explanation": gpt_explanation
        }
