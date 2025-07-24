#Prepare Multivariate Input Features
# Example with dummy data — replace with real values
n = 900
quantity = np.random.uniform(0, 1000, size=n)
volatility = np.random.uniform(0.001, 0.01, size=n)
net_buy = np.random.normal(0, 500, size=n)
obi = np.random.normal(0, 1, size=n)
ofi = np.random.normal(0, 2, size=n)

# Target variable (e.g., price levels per second)
Y = np.random.poisson(lam=3 + 0.002 * quantity + 10 * volatility)

# Stack input features into a 2D array
X = np.stack([quantity, volatility, net_buy, obi, ofi], axis=1)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

#Update Network Architecture

class PoissonNN(nn.Module):
    def __init__(self, input_dim):
        super(PoissonNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # Ensures λ > 0
        )

    def forward(self, x):
        return self.model(x)
#Train the model
input_dim = X.shape[1]  # 5 features
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
        
#Make Predictions & Visualize

model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

plt.figure(figsize=(10, 6))
plt.plot(Y, label="Actual Price Levels")
plt.plot(y_pred, label="Predicted λ", alpha=0.7)
plt.legend()
plt.title("Predicted vs Actual Number of Price Levels")
plt.show()



