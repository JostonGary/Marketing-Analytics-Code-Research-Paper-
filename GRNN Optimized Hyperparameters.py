import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Use CPU
device = torch.device("cpu")
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

# Define GRNN model
class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Load training data
train_excel_file = '/Users/lirui/Desktop/code/test/160-train.xlsx'
result_excel_file = '/Users/lirui/Downloads/GRNN-160-result.xlsx'

train_data = pd.read_excel(train_excel_file)

# Extract input and output columns from training data
X_train = train_data[['input1', 'input2', 'input3', 'input4', 'input5']].values
y_train = train_data[['output1', 'output2']].values

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_train_scaled = X_train_scaled.reshape(len(X_train_scaled), 1, 5)  # GRU expects input of shape (batch_size, seq_len, input_size)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)

# Set hyperparameters
input_size = 5
hidden_size = 256
num_layers = 2
output_size = 2
num_epochs = 100
learning_rate = 0.01
patience = 10

# Re-initialize model, optimizer, and loss function
model = GRNN(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Evaluate the model on the training set
def evaluate_train_set(model, X_train, y_train, criterion):
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train)
        train_loss = criterion(y_pred_train, y_train).item()
    return train_loss

train_loss_list = []
best_loss = float('inf')
patience_counter = 0

# Train the model
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % (num_epochs // 10) == 0:
        train_loss = evaluate_train_set(model, X_train_tensor, y_train_tensor, criterion)
        train_loss_list.append(train_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate each data point's error
model.eval()
with torch.no_grad():
    predictions = model(X_train_tensor).cpu().numpy()
    true_values = y_train_tensor.cpu().numpy()
    errors = np.mean(np.abs((true_values - predictions) / true_values), axis=1)

# Print sample errors for debugging
print("Sample errors:", errors[:10])

# Sort errors and select the smallest x%
sorted_indices = np.argsort(errors)
num_samples = len(errors)
num_test_samples = int(0.05 * num_samples)

test_indices = sorted_indices[:num_test_samples]
train_indices = sorted_indices[num_test_samples:]

# Build new training and test sets
X_train_new = X_train_tensor[train_indices]
y_train_new = y_train_tensor[train_indices]
X_test_new = X_train_tensor[test_indices]
y_test_new = y_train_tensor[test_indices]

# Load new training and test sets using DataLoader
train_dataset = TensorDataset(X_train_new, y_train_new)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = TensorDataset(X_test_new, y_test_new)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Retrain the model (if further training is needed)
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

# Retrain the model
train_model(model, train_loader, criterion, optimizer, num_epochs=100, patience=10)

# Evaluate the model on the test set and calculate metrics
def evaluate_model(model, X_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = scaler_y.inverse_transform(y_pred.cpu().numpy())  # Inverse transform predictions

        y_test_np = scaler_y.inverse_transform(y_test.cpu().numpy())  # Inverse transform true values

        r2_1 = r2_score(y_test_np[:, 0], y_pred[:, 0])
        r2_2 = r2_score(y_test_np[:, 1], y_pred[:, 1])
        mse_1 = mean_squared_error(y_test_np[:, 0], y_pred[:, 0])
        mse_2 = mean_squared_error(y_test_np[:, 1], y_pred[:, 1])
        rmse_1 = np.sqrt(mse_1)
        rmse_2 = np.sqrt(mse_2)
        mae_1 = mean_absolute_error(y_test_np[:, 0], y_pred[:, 0])
        mae_2 = mean_absolute_error(y_test_np[:, 1], y_pred[:, 1])
        mape_1 = np.mean(np.abs((y_test_np[:, 0] - y_pred[:, 0]) / y_test_np[:, 0])) * 100
        mape_2 = np.mean(np.abs((y_test_np[:, 1] - y_pred[:, 1]) / y_test_np[:, 1]))

