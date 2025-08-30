import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from typing import Dict, Any, Tuple
import joblib

class CropYieldPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = ['State', 'Crop_Year', 'Crop', 'Season', 'Area', 
                               'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        self.is_trained = False
        
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate the dataset"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
                
            data = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = self.feature_columns + ['Yield']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Check for missing values
            if data.isnull().sum().sum() > 0:
                print("Warning: Dataset contains missing values. Cleaning...")
                data = data.dropna()
                
            print(f"Dataset loaded successfully. Shape: {data.shape}")
            return data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, pd.Series]:
        """Preprocess the data for training or prediction"""
        try:
            processed_data = data.copy()
            
            # Encode categorical variables
            categorical_columns = ['Crop', 'Season', 'State']
            
            for col in categorical_columns:
                if is_training:
                    # Create and fit label encoder during training
                    le = LabelEncoder()
                    processed_data[col] = le.fit_transform(processed_data[col])
                    self.label_encoders[col] = le
                else:
                    # Use existing label encoder for prediction
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        processed_data[col] = processed_data[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else 0
                        )
                    else:
                        raise ValueError(f"Label encoder for {col} not found. Train the model first.")
            
            # Extract features
            X = processed_data[self.feature_columns]
            
            # Scale features
            if is_training:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not found. Train the model first.")
                X_scaled = self.scaler.transform(X)
            
            # Extract target variable (only during training)
            y = processed_data['Yield'] if 'Yield' in processed_data.columns else None
            
            return X_scaled, y
            
        except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            raise

    def train_model(self, file_path: str) -> Dict[str, float]:
        """Train the Random Forest model"""
        try:
            # Load and validate data
            data = self.load_and_validate_data(file_path)
            
            # Preprocess data
            X_scaled, y = self.preprocess_data(data, is_training=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=44, shuffle=True
            )
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100, 
                random_state=44,
                max_depth=10,
                min_samples_split=5
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2_score': r2_score(y_test, y_pred)
            }
            
            self.is_trained = True
            
            print("Model Training Complete!")
            print(f"MSE: {metrics['mse']:.2f}")
            print(f"RMSE: {metrics['rmse']:.2f}")
            print(f"R² Score: {metrics['r2_score']:.2f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise

    def predict_yield(self, input_data: Dict[str, Any]) -> float:
        """Make prediction for a single input"""
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained yet. Train the model first.")
                
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Preprocess input
            X_scaled, _ = self.preprocess_data(input_df, is_training=False)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            return max(0, prediction)  # Ensure non-negative yield
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            raise

    def save_model(self, model_path: str = 'crop_yield_model.pkl'):
        """Save the trained model and preprocessors"""
        try:
            if not self.is_trained:
                raise ValueError("No trained model to save.")
                
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns
            }
            
            joblib.dump(model_data, model_path)
            print(f"Model saved to {model_path}")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str = 'crop_yield_model.pkl'):
        """Load a pre-trained model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            
            print(f"Model loaded from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
            
        importance_dict = dict(zip(
            self.feature_columns, 
            self.model.feature_importances_
        ))
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

# Utility functions for Flask app
def initialize_model():
    """Initialize the model for the Flask app"""
    predictor = CropYieldPredictor()
    
    # Try to load existing model, otherwise train new one
    try:
        predictor.load_model()
    except (FileNotFoundError, Exception):
        print("No pre-trained model found. You need to train the model first.")
        print("Place your dataset in the 'data' folder and run train_model.py")
    
    return predictor

def validate_input(input_data: Dict[str, Any]) -> bool:
    """Validate user input"""
    required_fields = ['State', 'Crop_Year', 'Crop', 'Season', 'Area', 
                      'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    
    for field in required_fields:
        if field not in input_data or input_data[field] is None:
            return False
            
        # Basic type validation
        if field == 'Crop_Year':
            try:
                year = int(input_data[field])
                if year < 1990 or year > 2030:
                    return False
            except (ValueError, TypeError):
                return False
                
        elif field in ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']:
            try:
                value = float(input_data[field])
                if value < 0:
                    return False
            except (ValueError, TypeError):
                return False
    
    return True

# Example usage and training script
if __name__ == "__main__":
    # Initialize predictor
    predictor = CropYieldPredictor()
    
    # Train model 
    dataset_path = "data/crop_yield.csv" 
    
    try:
        metrics = predictor.train_model(dataset_path)
        predictor.save_model()
        
        # Test prediction
        sample_input = {
            'State': 'Punjab',
            'Crop_Year': 2020,
            'Crop': 'Rice',
            'Season': 'Kharif',
            'Area': 100.0,
            'Annual_Rainfall': 1200.0,
            'Fertilizer': 150.0,
            'Pesticide': 50.0
        }
        
        prediction = predictor.predict_yield(sample_input)
        print(f"\nSample Prediction: {prediction:.2f} tons/hectare")
        
        # Show feature importance
        importance = predictor.get_feature_importance()
        print("\nFeature Importance:")
        for feature, importance_score in importance.items():
            print(f"{feature}: {importance_score:.3f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to place your crop_yield.csv file in the 'data' folder")