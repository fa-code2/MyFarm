document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsSection = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');
    const predictBtn = form.querySelector('.predict-btn');
    const btnText = predictBtn.querySelector('.btn-text');
    const loading = predictBtn.querySelector('.loading');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        setLoadingState(true);
        hideError();
        hideResults();

        try {
            // Get form data
            const formData = new FormData(form);
            
            // Send prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                displayResults(result);
            } else {
                showError(result.error || 'Prediction failed. Please try again.');
            }

        } catch (error) {
            console.error('Error:', error);
            showError('Connection error. Please check your internet connection and try again.');
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(isLoading) {
        if (isLoading) {
            form.classList.add('form-loading');
            btnText.style.display = 'none';
            loading.style.display = 'inline';
        } else {
            form.classList.remove('form-loading');
            btnText.style.display = 'inline';
            loading.style.display = 'none';
        }
    }

    function displayResults(result) {
        const predictionOutput = document.getElementById('prediction-output');
        const inputDetails = document.getElementById('input-details');
        const importanceChart = document.getElementById('importance-chart');

        // Display prediction
        predictionOutput.textContent = `${result.prediction} tons/hectare`;
        predictionOutput.classList.add('success-animation');

        // Display input summary
        inputDetails.innerHTML = '';
        const inputData = result.input_data;
        
        Object.entries(inputData).forEach(([key, value]) => {
            const inputItem = document.createElement('div');
            inputItem.className = 'input-item';
            
            const displayName = formatFieldName(key);
            const displayValue = formatFieldValue(key, value);
            
            inputItem.innerHTML = `<strong>${displayName}:</strong> ${displayValue}`;
            inputDetails.appendChild(inputItem);
        });

        // Display feature importance
        displayFeatureImportance(result.feature_importance, importanceChart);

        // Show results section
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function displayFeatureImportance(importance, container) {
        container.innerHTML = '';
        
        // Get top 5 most important features
        const topFeatures = Object.entries(importance)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);

        topFeatures.forEach(([feature, value]) => {
            const bar = document.createElement('div');
            bar.className = 'importance-bar';
            
            const percentage = (value * 100).toFixed(1);
            
            bar.innerHTML = `
                <div class="importance-name">${formatFieldName(feature)}</div>
                <div class="importance-visual">
                    <div class="importance-fill" style="width: ${percentage}%"></div>
                </div>
                <div class="importance-value">${percentage}%</div>
            `;
            
            container.appendChild(bar);
        });
    }

    function formatFieldName(field) {
        const fieldNames = {
            'State': 'State',
            'Crop_Year': 'Crop Year',
            'Crop': 'Crop Type',
            'Season': 'Season',
            'Area': 'Area',
            'Annual_Rainfall': 'Annual Rainfall',
            'Fertilizer': 'Fertilizer',
            'Pesticide': 'Pesticide'
        };
        return fieldNames[field] || field;
    }

    function formatFieldValue(field, value) {
        switch(field) {
            case 'Area':
                return `${value} hectares`;
            case 'Annual_Rainfall':
                return `${value} mm`;
            case 'Fertilizer':
                return `${value} kg/hectare`;
            case 'Pesticide':
                return `${value} kg/hectare`;
            case 'Crop_Year':
                return value;
            default:
                return value;
        }
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Hide error after 5 seconds
        setTimeout(() => {
            hideError();
        }, 5000);
    }

    function hideError() {
        errorMessage.style.display = 'none';
    }

    function hideResults() {
        resultsSection.style.display = 'none';
    }

    // Add some sample data for testing
    document.getElementById('state').addEventListener('focus', function() {
        if (!this.value) {
            this.placeholder = 'e.g., Punjab, Maharashtra, Uttar Pradesh';
        }
    });

    document.getElementById('crop').addEventListener('focus', function() {
        if (!this.value) {
            this.placeholder = 'e.g., Rice, Wheat, Cotton, Sugarcane';
        }
    });

    // Add input validation
    const numericInputs = ['area', 'rainfall', 'fertilizer', 'pesticide'];
    numericInputs.forEach(inputId => {
        const input = document.getElementById(inputId);
        input.addEventListener('input', function() {
            const value = parseFloat(this.value);
            if (value < 0) {
                this.value = 0;
            }
        });
    });

    // Add year validation
    const yearInput = document.getElementById('crop_year');
    yearInput.addEventListener('input', function() {
        const year = parseInt(this.value);
        if (year < 1990) {
            this.value = 1990;
        } else if (year > 2030) {
            this.value = 2030;
        }
    });
});