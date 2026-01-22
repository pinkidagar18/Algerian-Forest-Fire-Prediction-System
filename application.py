import pickle
import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
application = Flask(__name__)
app = application

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Global variables for models
ridge_model = None
standard_scaler = None
MODELS_LOADED = False

# --- Model Loading ---
def load_models():
    """
    Load the trained Ridge Regression model and StandardScaler.
    Returns True if successful, False otherwise.
    """
    global ridge_model, standard_scaler, MODELS_LOADED
    
    model_paths = {
        'ridge': 'models/ridge.pkl',
        'scaler': 'models/scaler.pkl'
    }
    
    try:
        logger.info("Loading machine learning models...")
        
        # Load Ridge Regression model
        with open(model_paths['ridge'], 'rb') as f:
            ridge_model = pickle.load(f)
        logger.info("✓ Ridge model loaded successfully")
        
        # Load StandardScaler
        with open(model_paths['scaler'], 'rb') as f:
            standard_scaler = pickle.load(f)
        logger.info("✓ StandardScaler loaded successfully")
        
        MODELS_LOADED = True
        logger.info("All models loaded successfully!")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.error("Please ensure 'models/ridge.pkl' and 'models/scaler.pkl' exist")
        MODELS_LOADED = False
        return False
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        MODELS_LOADED = False
        return False


def validate_input_data(data):
    """
    Validate input data ranges and types.
    Returns (is_valid, error_message)
    """
    validations = {
        'Temperature': (0, 60, "Temperature must be between 0°C and 60°C"),
        'RH': (0, 100, "Relative Humidity must be between 0% and 100%"),
        'Ws': (0, 200, "Wind Speed must be between 0 and 200 km/h"),
        'Rain': (0, 500, "Rainfall must be between 0 and 500 mm"),
        'FFMC': (0, 101, "FFMC must be between 0 and 101"),
        'DMC': (0, 500, "DMC must be between 0 and 500"),
        'ISI': (0, 100, "ISI must be between 0 and 100"),
        'Classes': (0, 1, "Classes must be 0 (not fire) or 1 (fire)"),
        'Region': (0, 1, "Region must be 0 or 1")
    }
    
    for field, (min_val, max_val, error_msg) in validations.items():
        value = data.get(field)
        
        if value is None:
            return False, f"Missing required field: {field}"
        
        try:
            value = float(value)
            if not (min_val <= value <= max_val):
                return False, error_msg
        except (ValueError, TypeError):
            return False, f"Invalid value for {field}. Must be a number."
    
    return True, None


# --- Routes ---

@app.route('/')
def index():
    """Render the landing page"""
    logger.info("Index page accessed")
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handle FWI prediction requests.
    GET: Display the prediction form
    POST: Process form data and return prediction
    """
    
    # Check if models are loaded
    if not MODELS_LOADED:
        error_msg = "Error: Prediction models could not be loaded. Please contact the administrator."
        logger.error("Prediction attempted but models not loaded")
        return render_template('home.html', results=error_msg, fwi_result=None)
    
    if request.method == 'POST':
        try:
            logger.info("Processing prediction request...")
            
            # Extract form data (9 features matching the model training)
            form_data = {
                'Temperature': request.form.get('Temperature'),
                'RH': request.form.get('RH'),
                'Ws': request.form.get('Ws'),
                'Rain': request.form.get('Rain'),
                'FFMC': request.form.get('FFMC'),
                'DMC': request.form.get('DMC'),
                'ISI': request.form.get('ISI'),
                'Classes': request.form.get('Classes'),
                'Region': request.form.get('Region')
            }
            
            # Validate input data
            is_valid, error_message = validate_input_data(form_data)
            if not is_valid:
                logger.warning(f"Invalid input: {error_message}")
                return render_template('home.html', results=error_message, fwi_result=None)
            
            # Convert to float values
            Temperature = float(form_data['Temperature'])
            RH = float(form_data['RH'])
            Ws = float(form_data['Ws'])
            Rain = float(form_data['Rain'])
            FFMC = float(form_data['FFMC'])
            DMC = float(form_data['DMC'])
            ISI = float(form_data['ISI'])
            Classes = float(form_data['Classes'])
            Region = float(form_data['Region'])
            
            # Log input values
            logger.info(f"Input: Temp={Temperature}, RH={RH}, Ws={Ws}, Rain={Rain}, "
                       f"FFMC={FFMC}, DMC={DMC}, ISI={ISI}, Classes={Classes}, Region={Region}")
            
            # Prepare input array for prediction
            # ORDER MUST MATCH TRAINING: Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region
            input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            
            logger.info(f"Input array shape: {input_data.shape}")
            
            # Scale the input data
            input_scaled = standard_scaler.transform(input_data)
            
            # Make prediction
            prediction = ridge_model.predict(input_scaled)
            fwi_result = float(prediction[0])
            
            # Ensure non-negative FWI
            fwi_result = max(0, fwi_result)
            
            logger.info(f"Prediction successful: FWI = {fwi_result:.2f}")
            
            return render_template('home.html', results="Success", fwi_result=fwi_result)
            
        except ValueError as e:
            error_message = f"Invalid input: {str(e)}. All fields must be valid numbers."
            logger.warning(f"ValueError: {error_message}")
            return render_template('home.html', results=error_message, fwi_result=None)
            
        except Exception as e:
            error_message = f"An unexpected error occurred during prediction. Please try again."
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return render_template('home.html', results=error_message, fwi_result=None)
    
    else:
        # GET request - display the form
        logger.info("Prediction form accessed")
        return render_template('home.html', results=None, fwi_result=None)


@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring.
    Returns JSON with system status.
    """
    status = {
        'status': 'healthy' if MODELS_LOADED else 'unhealthy',
        'models_loaded': MODELS_LOADED,
        'ridge_model': ridge_model is not None,
        'scaler': standard_scaler is not None
    }
    return jsonify(status), 200 if MODELS_LOADED else 503


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.url}")
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"500 error: {str(e)}", exc_info=True)
    return render_template('home.html', 
                         results="An internal error occurred. Please try again later.", 
                         fwi_result=None), 500


# --- Application Startup ---
if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Get configuration from environment variables
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 7860))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Flask application on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    # Run the application
    app.run(host=host, port=port, debug=debug)