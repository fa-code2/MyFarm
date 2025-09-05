from flask import Flask, render_template, request, jsonify, flash
from model import CropYieldPredictor, validate_input
import os
import traceback

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'your-secret-key-here'  # Change this in production

# Global model instance
predictor = None

def initialize_app():
    """Initialize the Flask app with the trained model"""
    global predictor
    try:
        predictor = CropYieldPredictor()
        predictor.load_model('crop_yield_model.pkl')
        print(" Model loaded successfully!")
        return True
    except Exception as e:
        print(f" Failed to load model: {str(e)}")
        print("Please run 'python train_model.py' first to train the model.")
        return False

@app.route('/')
def home():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if predictor is None or not predictor.is_trained:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please contact administrator.'
            })

        # Get form data
        input_data = {
            'State': request.form.get('state', '').strip(),
            'Crop_Year': request.form.get('crop_year', ''),
            'Crop': request.form.get('crop', '').strip(),
            'Season': request.form.get('season', '').strip(),
            'Area': request.form.get('area', ''),
            'Annual_Rainfall': request.form.get('rainfall', ''),
            'Fertilizer': request.form.get('fertilizer', ''),
            'Pesticide': request.form.get('pesticide', '')
        }

        # Convert numeric fields
        numeric_fields = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        for field in numeric_fields:
            try:
                input_data[field] = float(input_data[field]) if input_data[field] else 0.0
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': f'Invalid value for {field}. Please enter a valid number.'
                })

        # Validate input
        if not validate_input(input_data):
            return jsonify({
                'success': False,
                'error': 'Please fill in all required fields with valid values.'
            })

        # Make prediction
        prediction = predictor.predict_yield(input_data)
        
        # Get feature importance for this prediction
        feature_importance = predictor.get_feature_importance()

        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'input_data': input_data,
            'feature_importance': feature_importance
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        })

@app.route('/results')
def results():
    """Display prediction results"""
    return render_template('results.html')

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    if predictor is None or not predictor.is_trained:
        return jsonify({'error': 'Model not loaded'})
    
    try:
        feature_importance = predictor.get_feature_importance()
        return jsonify({
            'feature_importance': feature_importance,
            'feature_columns': predictor.feature_columns
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print(" Starting MyFarm Application...")
    
    if initialize_app():
        print(" Flask app is running!")
        print(" Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print(" Failed to start application. Model not available.")
        print("Please run: python train_model.py")