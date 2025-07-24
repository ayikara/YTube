#Define the PRNN Model Class

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PoissonNN(nn.Module):
    def __init__(self, input_dim):
        super(PoissonNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # Ensures positive output (λ)
        )

    def forward(self, x):
        return self.model(x)
#Poisson Loss

def poisson_nll(y_pred, y_true):
    return torch.mean(y_pred - y_true * torch.log(y_pred + 1e-8))

# Train and Save the Model

# X: numpy array with shape (n_samples, n_features)
# Y: numpy array with shape (n_samples,)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# Train the model
input_dim = X.shape[1]
model = PoissonNN(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_tensor)
    loss = poisson_nll(y_pred, Y_tensor)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save model and input_dim
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim
}, "poisson_model.pth")

#Load the Model and Use on New Data

# Load model
checkpoint = torch.load("poisson_model.pth")
input_dim = checkpoint['input_dim']

loaded_model = PoissonNN(input_dim)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

# Predict on new data

# X_new: new input data (e.g., shape = [n_new_samples, input_dim])
# Ensure X_new is a NumPy array of shape (n_samples, input_dim)

X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

with torch.no_grad():
    Y_new_pred = loaded_model(X_new_tensor).numpy()

# Y_new_pred is a (n_samples, 1) array of predicted λ values

