import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set random seed
set_seed(42)

# Data paths
train_excel_file = '/mnt/data/160-train.xlsx'
result_excel_file = '/mnt/data/DNN-160-result.xlsx'

# Load training data and prediction data from Excel file, only loading useful columns
train_df = pd.read_excel(train_excel_file, usecols=['input1', 'input2', 'input3', 'input4', 'input5', 'output1', 'output2'])

# Extract input and output columns from training data
X = train_df[['input1', 'input2', 'input3', 'input4', 'input5']].values
y = train_df[['output1', 'output2']].values

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors and move to GPU
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

# Define the fully connected neural network model (DNN) and move to GPU
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Create model instance and move to GPU
model = Net().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Train the model with early stopping mechanism
def train_model(model, train_loader, criterion, optimizer, num_epochs=100, patience=10):
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

# Train the model
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_model(model, train_loader, criterion, optimizer, num_epochs=100, patience=10)

# Load the best model parameters
model.load_state_dict(torch.load('best_model.pth', map_location=device))

# Evaluate the model
def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
        mse = mean_squared_error(y_test_np, predictions)
        mae = mean_absolute_error(y_test_np, predictions)
        r2 = r2_score(y_test_np, predictions)
        print(f'MSE: {mse}, MAE: {mae}, R2: {r2}')
        return y_test_np, predictions

# Evaluate the model on the training data
all_labels, all_predictions = evaluate_model(model, X_tensor, y_tensor)

# Save test parameters and predictions
def save_results(X_test_original, y_test, predictions, result_excel_file):
    results_df = pd.DataFrame(np.hstack((X_test_original, y_test, predictions)),
                              columns=['input1', 'input2', 'input3', 'input4', 'input5', 'output1_true', 'output2_true' , 'output1_pred', 'output2_pred'])
    results_df.to_excel(result_excel_file, index=False)

# Inverse transform to get the original input data
X_test_original = scaler.inverse_transform(X_tensor.cpu().numpy())

# Save results
save_results(X_test_original, y_tensor.cpu().numpy(), all_predictions, result_excel_file)


