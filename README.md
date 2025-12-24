# 🔥 Algerian Forest Fire Prediction System

A machine learning-powered web application that predicts the Fire Weather Index (FWI) for forest fire risk assessment in Algeria using meteorological and environmental data.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Input Parameters](#input-parameters)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)
- [Common Issues & Solutions](#common-issues--solutions)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

Forest fires pose a significant threat to ecosystems, wildlife, and human settlements. This project leverages machine learning to predict the Fire Weather Index (FWI), a critical metric used by fire management agencies to assess fire danger levels. By analyzing meteorological and environmental factors, the system provides real-time predictions to aid in fire prevention and resource allocation.

### What is FWI?

The Fire Weather Index (FWI) is a numeric rating of fire intensity. It combines various factors including temperature, humidity, wind speed, and rainfall to produce a comprehensive fire danger rating. Higher FWI values indicate greater fire danger.

## ✨ Features

- **Real-time FWI Prediction**: Instant fire weather index predictions based on current conditions
- **User-Friendly Interface**: Clean, responsive web interface with modern design
- **Input Validation**: Comprehensive validation of all input parameters
- **Model Performance**: Ridge Regression model with optimized hyperparameters
- **RESTful API**: Health check endpoint for monitoring system status
- **Error Handling**: Robust error handling and logging system
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 📊 Dataset

The project uses the **Algerian Forest Fires Dataset**, which contains meteorological observations from two regions in Algeria:

- **Bejaia Region** (Northeast Algeria)
- **Sidi Bel-abbes Region** (Northwest Algeria)

### Dataset Features:

| Feature | Description | Range |
|---------|-------------|-------|
| **Temperature** | Temperature in Celsius | 0-60°C |
| **RH** | Relative Humidity | 0-100% |
| **Ws** | Wind Speed | 0-200 km/h |
| **Rain** | Rainfall | 0-500 mm |
| **FFMC** | Fine Fuel Moisture Code | 0-101 |
| **DMC** | Duff Moisture Code | 0-500 |
| **ISI** | Initial Spread Index | 0-100 |
| **Classes** | Fire/No Fire (0=No Fire, 1=Fire) | 0-1 |
| **Region** | Geographic Region (0=Bejaia, 1=Sidi Bel-abbes) | 0-1 |

**Target Variable**: FWI (Fire Weather Index)

## 🤖 Model

### Algorithm: Ridge Regression

Ridge Regression was selected after evaluating multiple regression algorithms due to its:
- Excellent performance on the dataset
- Ability to handle multicollinearity
- Regularization to prevent overfitting
- Stable predictions

### Model Pipeline:

1. **Data Preprocessing**:
   - Feature scaling using StandardScaler
   - Handling of categorical variables
   - Data cleaning and validation

2. **Model Training**:
   - Ridge Regression with optimized alpha parameter
   - Cross-validation for hyperparameter tuning
   - Train-test split for evaluation

3. **Model Persistence**:
   - Models saved as pickle files (`ridge.pkl`, `scaler.pkl`)
   - Easy deployment and version control

### Performance Metrics:

- Model evaluation metrics are available in the `Model_Training.ipynb` notebook
- Comprehensive analysis including R², MAE, and RMSE

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/algerian-forest-fire-prediction.git
cd algerian-forest-fire-prediction
```

2. **Create a virtual environment**:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up the project structure**:
```
project/
├── application.py
├── requirements.txt
├── models/
│   ├── ridge.pkl
│   └── scaler.pkl
└── templates/
    ├── index.html
    └── home.html
```

5. **Run the application**:
```bash
python application.py
```

The application will start on `http://localhost:5000`

## 💻 Usage

### Running Locally

1. Start the Flask server:
```bash
python application.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter the required meteorological parameters in the prediction form

4. Click "Predict FWI" to get the fire weather index prediction

### Using the Prediction Form

1. **Navigate to the prediction page**: Click "Get Started" or "Predict Now" on the landing page
2. **Fill in the form** with the following parameters:
   - Temperature (°C)
   - Relative Humidity (%)
   - Wind Speed (km/h)
   - Rainfall (mm)
   - FFMC (Fine Fuel Moisture Code)
   - DMC (Duff Moisture Code)
   - ISI (Initial Spread Index)
   - Classes (0 for No Fire, 1 for Fire conditions)
   - Region (0 for Bejaia, 1 for Sidi Bel-abbes)
3. **Submit** the form to receive your FWI prediction

### Environment Variables (Optional)

You can configure the application using environment variables:

```bash
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
export FLASK_DEBUG=False
export SECRET_KEY=your-secret-key-here
```

## 📁 Project Structure

```
algerian-forest-fire-prediction/
│
├── application.py                          # Main Flask application
├── requirements.txt                        # Python dependencies
├── README.md                              # Project documentation
│
├── models/                                # Trained models
│   ├── ridge.pkl                         # Ridge Regression model
│   └── scaler.pkl                        # StandardScaler for preprocessing
│
├── templates/                            # HTML templates
│   ├── index.html                        # Landing page
│   └── home.html                         # Prediction form page
│
├── notebooks/                            # Jupyter notebooks
│   ├── Algerian_Forest_Fires_Dataset.ipynb    # Data exploration
│   └── Model_Training.ipynb              # Model training and evaluation
│
└── data/                                 # Dataset files
    ├── Algerian_forest_fires_cleaned_dataset.csv
    └── Algerian_forest_fires_dataset_UPDATE.csv
```

## 🔌 API Endpoints

### 1. Home Page
- **URL**: `/`
- **Method**: `GET`
- **Description**: Landing page with project information

### 2. Prediction Page
- **URL**: `/predictdata`
- **Method**: `GET`, `POST`
- **Description**: 
  - GET: Display prediction form
  - POST: Process form data and return FWI prediction
- **Request Body** (POST):
```json
{
  "Temperature": 30,
  "RH": 50,
  "Ws": 15,
  "Rain": 0,
  "FFMC": 85,
  "DMC": 25,
  "ISI": 5,
  "Classes": 1,
  "Region": 0
}
```
- **Response**: HTML page with prediction result

### 3. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Check system status and model availability
- **Response**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "ridge_model": true,
  "scaler": true
}
```

## 📝 Input Parameters

### Required Parameters

| Parameter | Description | Type | Valid Range |
|-----------|-------------|------|-------------|
| Temperature | Air temperature in Celsius | Float | 0-60 |
| RH | Relative Humidity percentage | Float | 0-100 |
| Ws | Wind Speed in km/h | Float | 0-200 |
| Rain | Rainfall in mm | Float | 0-500 |
| FFMC | Fine Fuel Moisture Code | Float | 0-101 |
| DMC | Duff Moisture Code | Float | 0-500 |
| ISI | Initial Spread Index | Float | 0-100 |
| Classes | Fire condition (0=No Fire, 1=Fire) | Integer | 0-1 |
| Region | Geographic region (0=Bejaia, 1=Sidi Bel-abbes) | Integer | 0-1 |

### Example Input

```python
Temperature = 35.0
RH = 45.0
Ws = 20.0
Rain = 0.0
FFMC = 90.5
DMC = 30.0
ISI = 8.5
Classes = 1
Region = 0
```

## 🛠 Technologies Used

### Backend
- **Flask 3.0.0**: Web framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **Gunicorn**: WSGI HTTP Server (for production)

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling with modern gradients and animations
- **JavaScript**: Interactive elements
- **Google Fonts**: Typography (DM Sans, Crimson Pro)

### Development Tools
- **Jupyter Notebook**: Data analysis and model training
- **Pickle**: Model serialization
- **Logging**: Application monitoring

## 📸 Screenshots

### 1. Landing Page
The landing page features a modern, nature-inspired design with fire-themed colors and smooth animations, providing an intuitive entry point to the application.

![Landing Page](images/Home_page.png)

**Features:**
- Modern dark theme with gradient backgrounds
- Clear call-to-action buttons
- Feature highlights showcasing key capabilities
- Responsive design for all devices

### 2. Prediction Form
User-friendly form interface with clear input fields, real-time validation, and helpful placeholders for each meteorological parameter.

![Prediction Form](images_prediction.png)

**Features:**
- 9 input fields for meteorological data
- Input validation with range checking
- Clear labels and placeholder examples
- Intuitive layout for easy data entry

### 3. Prediction Results
Clean results display showing the calculated Fire Weather Index with risk level interpretation and actionable insights.

![Prediction Result](images/result_page.png)

**Features:**
- Large, clear FWI value display
- Risk level interpretation
- Color-coded results for quick assessment
- Option to make additional predictions

### 4. System Architecture
Overview of the application's architecture showing the flow from user input through the ML model to prediction output.

![System Architecture](images/04_architecture.png)

**Components:**
- Web interface layer (HTML/CSS/JS)
- Flask backend API
- ML prediction engine (Ridge Regression)
- Data processing pipeline

## 🔧 Troubleshooting

This section documents common issues encountered during development and their solutions to help you avoid the same pitfalls.

### Common Issues & Solutions

#### 1. Model Loading Errors

**Issue**: `FileNotFoundError: [Errno 2] No such file or directory: 'models/ridge.pkl'`

**Cause**: The Flask application expects model files in a `models/` directory, but they're in the root directory or wrong location.

**Solution**:
```bash
# Create models directory if it doesn't exist
mkdir -p models

# Move pickle files to models directory
mv ridge.pkl models/
mv scaler.pkl models/

# Verify the structure
ls -la models/
```

**Prevention**: Always ensure your project structure matches the code's expectations. The application.py file loads models from:
```python
model_paths = {
    'ridge': 'models/ridge.pkl',
    'scaler': 'models/scaler.pkl'
}
```

---

#### 2. Scikit-learn Version Mismatch

**Issue**: `InconsistentVersionWarning: Trying to unpickle estimator Ridge from version 1.3.0 when using version 1.5.0`

**Cause**: The model was trained with one version of scikit-learn but you're trying to load it with a different version.

**Solutions**:

**Option 1 - Match the training version** (Recommended):
```bash
pip install scikit-learn==1.3.0 --break-system-packages
```

**Option 2 - Retrain the model**:
```python
# Run the Model_Training.ipynb notebook again
# This will create new pickle files compatible with your current version
```

**Option 3 - Ignore the warning** (Not recommended for production):
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

**Best Practice**: Document the exact versions used during training in requirements.txt:
```
scikit-learn==1.3.0
numpy==1.26.0
pandas==2.1.0
```

---

#### 3. Template Not Found Error

**Issue**: `jinja2.exceptions.TemplateNotFound: index.html`

**Cause**: Flask can't find the HTML template files.

**Solution**:
```bash
# Ensure templates directory exists
mkdir -p templates

# Move HTML files to templates directory
mv index.html templates/
mv home.html templates/

# Verify structure
tree .
# Expected:
# .
# ├── application.py
# ├── models/
# │   ├── ridge.pkl
# │   └── scaler.pkl
# └── templates/
#     ├── index.html
#     └── home.html
```

**Flask Convention**: Flask automatically looks for templates in a `templates/` folder in the same directory as your application.

---

#### 4. Input Data Shape Mismatch

**Issue**: `ValueError: X has 8 features, but StandardScaler is expecting 9 features`

**Cause**: The number of input features doesn't match what the model was trained on.

**Root Causes**:
- Missing a form field in HTML
- Incorrect order of features
- Forgot to include a feature in the prediction array

**Solution**:
```python
# Ensure ALL 9 features are included in the CORRECT ORDER:
# Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region

input_data = np.array([[
    Temperature,  # Feature 1
    RH,          # Feature 2
    Ws,          # Feature 3
    Rain,        # Feature 4
    FFMC,        # Feature 5
    DMC,         # Feature 6
    ISI,         # Feature 7
    Classes,     # Feature 8
    Region       # Feature 9
]])
```

**Debugging Tip**:
```python
# Add this to check your input shape
print(f"Input shape: {input_data.shape}")  # Should be (1, 9)
print(f"Expected features: {standard_scaler.n_features_in_}")
```

---

#### 5. Flask Not Running / Import Errors

**Issue**: `ModuleNotFoundError: No module named 'flask'`

**Cause**: Flask or other dependencies not installed in the current environment.

**Solution**:
```bash
# Activate your virtual environment first
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -i flask
```

**If still having issues**:
```bash
# Clear pip cache and reinstall
pip cache purge
pip install --no-cache-dir -r requirements.txt
```

---

#### 6. Port Already in Use

**Issue**: `OSError: [Errno 48] Address already in use`

**Cause**: Another process is already using port 5000.

**Solution**:

**Option 1 - Kill the process using port 5000**:
```bash
# Find the process
lsof -i :5000
# OR on Windows
netstat -ano | findstr :5000

# Kill it (Linux/Mac)
kill -9 <PID>

# Kill it (Windows)
taskkill /PID <PID> /F
```

**Option 2 - Use a different port**:
```bash
# Set environment variable
export FLASK_PORT=5001

# Or modify application.py
app.run(host='0.0.0.0', port=5001)
```

---

#### 7. Negative FWI Predictions

**Issue**: Model sometimes predicts negative FWI values, which don't make sense.

**Cause**: Ridge Regression can predict negative values, but FWI should be ≥ 0.

**Solution** (Already implemented):
```python
# In application.py
fwi_result = float(prediction[0])
fwi_result = max(0, fwi_result)  # Ensure non-negative
```

**Alternative Solutions**:
- Use a different model (Random Forest, Gradient Boosting)
- Apply log transformation to target variable during training
- Add a ReLU activation: `fwi_result = max(0, fwi_result)`

---

#### 8. CSS/JavaScript Not Loading

**Issue**: Styles and scripts not applying to the web pages.

**Cause**: Static files not properly configured or linked.

**Solution**:

If using external CDN (current approach):
```html
<!-- Verify CDN links are accessible -->
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
```

If using local files:
```bash
# Create static directory
mkdir -p static/css static/js

# Update Flask configuration
app = Flask(__name__)
# Flask automatically serves from 'static' folder

# Reference in HTML
<link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
```

---

#### 9. Form Validation Not Working

**Issue**: Invalid data passes through validation or valid data gets rejected.

**Cause**: Validation ranges too strict or form values not being converted properly.

**Solution**:
```python
# Ensure proper type conversion before validation
try:
    Temperature = float(request.form.get('Temperature'))
except (ValueError, TypeError):
    return render_template('home.html', 
                         results="Invalid temperature value", 
                         fwi_result=None)

# Check validation ranges match your dataset
validations = {
    'Temperature': (0, 60),    # Adjust based on your data
    'RH': (0, 100),
    'Ws': (0, 200),
    # ... etc
}
```

**Debugging Tip**:
```python
# Log all form values for debugging
logger.info(f"Received form data: {request.form.to_dict()}")
```

---

#### 10. Model Accuracy Issues

**Issue**: Predictions seem inaccurate or inconsistent.

**Possible Causes & Solutions**:

**1. Data Quality Issues**:
```python
# Check for outliers in training data
df.describe()
df.boxplot()

# Handle outliers
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df_clean = df[(z_scores < 3).all(axis=1)]
```

**2. Feature Scaling Issues**:
```python
# Ensure you're scaling test data with the SAME scaler used in training
# DON'T create a new scaler for prediction
# DO use the saved scaler
input_scaled = standard_scaler.transform(input_data)  # ✓ Correct
```

**3. Missing Feature Engineering**:
```python
# Consider adding interaction features
df['temp_humidity_interaction'] = df['Temperature'] * df['RH']
df['wind_rain_interaction'] = df['Ws'] * df['Rain']
```

**4. Model Selection**:
```python
# Try different models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, ElasticNet

# Compare performance
models = {
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100)
}
```

---

### Development Best Practices Learned

Based on the issues encountered during development, here are key best practices:

#### 1. **Version Control Everything**
```bash
# Include a comprehensive requirements.txt with exact versions
pip freeze > requirements.txt

# Use git to track changes
git add .
git commit -m "Add model training notebook"
```

#### 2. **Logging is Essential**
```python
# Add comprehensive logging throughout
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log important operations
logger.info("Loading model...")
logger.error(f"Error: {str(e)}")
```

#### 3. **Validate Everything**
```python
# Validate inputs at multiple levels:
# 1. Frontend (HTML5 validation)
# 2. Backend (Python validation)
# 3. Before model prediction

def validate_input_data(data):
    """Comprehensive validation function"""
    # Check presence
    # Check types
    # Check ranges
    # Return clear error messages
```

#### 4. **Test Locally Before Deployment**
```bash
# Test different scenarios
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predictdata -d "Temperature=30&..."

# Test edge cases
# - Minimum values
# - Maximum values
# - Invalid values
# - Missing values
```

#### 5. **Document as You Go**
```python
# Add docstrings to all functions
def load_models():
    """
    Load the trained Ridge Regression model and StandardScaler.
    
    Returns:
        bool: True if successful, False otherwise.
        
    Raises:
        FileNotFoundError: If model files are not found.
    """
```

---

### Debugging Tools & Commands

Useful commands for troubleshooting:

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Check if Flask is running
ps aux | grep python

# Check network/port
netstat -an | grep 5000
lsof -i :5000

# Test application health
curl http://localhost:5000/health

# View application logs (if using systemd)
journalctl -u forest-fire-app -f

# Check file permissions
ls -la models/
ls -la templates/

# Verify pickle files
python -c "import pickle; print(pickle.load(open('models/ridge.pkl', 'rb')))"
```

---

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look at application.log or console output
2. **Enable debug mode**: Set `FLASK_DEBUG=True` for detailed error messages
3. **Verify environment**: Check all dependencies are installed
4. **Search issues**: Look at GitHub issues for similar problems
5. **Create an issue**: Provide:
   - Error message
   - Steps to reproduce
   - Python version
   - Operating system
   - Relevant logs

---

## 🔮 Future Improvements

- [ ] Add real-time weather API integration
- [ ] Implement user authentication and history tracking
- [ ] Create a dashboard for historical predictions
- [ ] Add map visualization of fire risk areas
- [ ] Develop a mobile application
- [ ] Implement ensemble models for improved accuracy
- [ ] Add multilingual support (French, Arabic)
- [ ] Create an alerting system for high fire risk
- [ ] Integrate satellite imagery analysis
- [ ] Add temporal forecasting (next 7 days)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Pinki** 

## 🙏 Acknowledgments

- Algerian Forest Fire Dataset contributors
- Flask and Scikit-learn communities
- Open source contributors
- Stack Overflow community for troubleshooting help

## 📧 Contact

For questions or feedback, please reach out:

- **Email**: pinkidagar18@gmail.com

**Note**: This project is for educational and research purposes. Always consult professional fire management agencies for actual fire risk assessments.

## 📚 References

- [Fire Weather Index System](https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

Made with ❤️ for forest conservation and fire prevention
