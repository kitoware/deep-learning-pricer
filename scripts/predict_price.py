import torch
import torch.nn as nn
import numpy as np
import joblib  # Use joblib to load previously saved scalers

# Same model definition as training
class AsianOptionNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

def load_model_and_scalers():
    """Load the final trained model's state dict and the saved scalers."""
    # 1. Load the scalers (identical to training)
    feature_scaler = joblib.load("./models/feature_scaler.pkl")
    label_scaler = joblib.load("./models/label_scaler.pkl")

    # 2. Load the model state dict
    #    Note: 'weights_only=True' requires PyTorch >= 2.1. For older versions, just omit it.
    state_dict = torch.load(
        "./models/asian_option_pricer.pt", 
        map_location=torch.device("cpu"),
        # weights_only=True  # Uncomment if you're on PyTorch 2.1+
    )

    # 3. Infer hidden_dim from the first weight
    #    (This works because we saved only the state_dict, not the entire model object)
    hidden_dim = state_dict['net.0.weight'].shape[0]
    
    model = AsianOptionNN(input_dim=5, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, feature_scaler, label_scaler

def predict_option_price(S0, r, T, K, sigma):
    """
    Predict the Asian option price based on input parameters.

    Parameters:
    S0 (float): Initial stock price.
    r (float): Risk-free interest rate.
    T (float): Time to maturity (in years).
    K (float): Strike price.
    sigma (float): Volatility.

    Returns:
    float: Predicted option price (unscaled).
    """
    model, feature_scaler, label_scaler = load_model_and_scalers()

    # Prepare input features
    input_features = np.array([[S0, r, T, K, sigma]], dtype=np.float32)

    # 1. Scale the inputs the same way as training
    input_features_scaled = feature_scaler.transform(input_features)

    # 2. Predict in scaled space
    input_tensor = torch.tensor(input_features_scaled, dtype=torch.float32)
    with torch.no_grad():
        predicted_price_scaled = model(input_tensor).numpy()

    # 3. Inverse transform to get the prediction in original price scale
    predicted_price = label_scaler.inverse_transform(predicted_price_scaled)

    return predicted_price[0, 0]

if __name__ == "__main__":
    # Prompt user for inputs
    S0 = float(input("Enter initial stock price (S0): "))
    r = float(input("Enter risk-free rate (r): "))
    T = float(input("Enter time to maturity in years (T): "))
    K = float(input("Enter strike price (K): "))
    sigma = float(input("Enter volatility (sigma): "))

    # Predict and display
    predicted_price = predict_option_price(S0, r, T, K, sigma)
    print(f"\nPredicted Asian Option Price: {predicted_price:.4f}")
