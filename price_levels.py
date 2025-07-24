
#complete and clean implementation of a Poisson Regression Neural Network (PRNN) in PyTorch, tailored to your use case: modeling the number of price levels (Y) from traded quantity (X), where Y is count data (non-negative integers)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Dummy data for illustration
# Replace with your actual Quantity (X) and Price Levels (Y) data
np.random.seed(0)
X = np.random.uniform(0, 1000, size=900)
Y = np.random.poisson(lam=3 + 0.002 * X)  # synthetic Poisson relationship

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

#Define the Poisson Regression Neural Network

class PoissonNN(nn.Module):
    def __init__(self):
        super(PoissonNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # Ensures λ > 0; softplus(x) = log(1 + e^x)
        )
        
    def forward(self, x):
        return self.model(x)

#Poisson Loss Function (Negative Log Likelihood)
def poisson_nll(y_pred, y_true):
    # y_pred: λ, expected rate
    # y_true: observed counts
    return torch.mean(y_pred - y_true * torch.log(y_pred + 1e-8))

 #Training the Model

 model = PoissonNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 300

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_tensor)
    loss = poisson_nll(y_pred, Y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#Plot predictions

model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.4, label="Observed")
plt.scatter(X, y_pred, color='red', alpha=0.6, label="Predicted λ (Price Levels)")
plt.xlabel("Traded Quantity (X)")
plt.ylabel("Number of Price Levels (Y)")
plt.title("Poisson Neural Network: Price Levels vs Quantity")
plt.legend()
plt.grid(True)
plt.show()

#Softplus ensures the model outputs are positive (as Poisson λ must be).

#The model learns a nonlinear function λ(X) predicting expected price levels.

#You can extend this to multiple features (e.g., order imbalance, time of day) by expanding input dimension.



