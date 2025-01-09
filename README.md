# **Asian Option Pricing with Neural Network**

This project implements a neural network model to predict the prices of Asian options based on market parameters (initial stock price, risk-free rate, time to maturity, strike price, and volatility). The project uses **QuantLib** to generate synthetic option prices, **PyTorch** for model training, and **Optuna** for hyperparameter tuning.

---

## **Installation**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-directory>
```

### **2. Install Dependencies**
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Data Generation**
Generate synthetic Asian option pricing data using **QuantLib**:
```bash
python scripts/generate_data.py
```
This creates a CSV file (`generated_asian_option_data.csv`) in the `./data/` folder.

### **2. Model Training**
Train the neural network to predict option prices:
```bash
python scripts/train_model.py
```
This runs **Optuna** to optimize the hyperparameters and saves the final trained model to the `./models/` directory.

### **3. Model Evaluation**
Evaluate the model’s performance on the test dataset:
```bash
python scripts/evaluate_model.py
```
This script outputs:
- Mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE).
- A scatter plot comparing predicted and true prices.
- A Q-Q plot of prediction errors.

### **4. Predict Option Prices**
To predict the price of an option based on specific inputs:
```bash
python scripts/predict_price.py
```
Enter the following inputs when prompted:
- `S0`: Initial stock price
- `r`: Risk-free interest rate
- `T`: Time to maturity (in years)
- `K`: Strike price
- `σ` (sigma): Volatility

The script outputs the predicted Asian option price.

---

## **Files and Scripts**

### **1. `generate_data.py`**
- Generates synthetic data for Asian options using **QuantLib**.
- Saves the data as a CSV file in the `data/` folder.

### **2. `train_nn.ipynb`**
- Trains the neural network on the generated dataset.
- Uses **Optuna** for hyperparameter tuning.
- Saves the trained model and scalers in the `models/` folder.

### **3. `evaluate_nn.ipynb`**
- Evaluates the trained model on the test set.
- Generates plots such as:
  - Predicted vs. True Prices (scatter plot)
  - Q-Q Plot of Residuals (to check for normality of errors)
  - Partial Dependence Plots (PDPs) to show how predictions vary with changes in features.

### **4. `predict_price.py`**
- Takes user inputs (market parameters) and outputs the predicted option price.

---

## **Results**

### **Sample Evaluation Metrics**
- **MSE (Mean Squared Error)**: 0.026089
- **RMSE (Root Mean Squared Error)**: 0.161520
- **MAE (Mean Absolute Error)**: 0.131204

### **Generated Plots**
- **Predicted vs True Prices**: Shows how closely the model’s predictions align with the true prices.
- **Q-Q Plot**: Checks if residuals follow a normal distribution.
- **PDP (Partial Dependence Plot)**: Visualizes how the predicted option price changes as a function of key features (e.g., `S0` and `sigma`).

---

### **More on Data Generation**
For each combination of S0, σ, T, and K, the option's price is calculated using a Monte Carlo simulation with 10,000 paths:

Process: The Black-Scholes model with flat interest and volatility curves.

Monte Carlo Engine: Low-discrepancy Sobol sequences for more efficient convergence.

Fixing dates: Time points over the option's lifetime to compute the average price.

---

## **Dependencies**
- Python 3.12
- **PyTorch** (for neural network training)
- **QuantLib** (for option pricing simulation)
- **Optuna** (for hyperparameter tuning)
- **Matplotlib** and **Seaborn** (for plots)
- **Pandas** and **NumPy** (for data handling)
- **Joblib** (for saving and loading scalers)

To see the full list of dependencies, refer to `requirements.txt`.

---

## **Contributing**
Feel free to submit issues or pull requests for improvements and bug fixes.

---

## **License**
This project is licensed under the MIT License.

