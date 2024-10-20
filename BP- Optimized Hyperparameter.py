import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Load training data
train_excel_file = '/Users/lirui/Desktop/code/test/160-train.xlsx'
train_df = pd.read_excel(train_excel_file, usecols=['input1', 'input2', 'input3', 'input4', 'input5', 'output1', 'output2'])

# Extract inputs and outputs
X = train_df[['input1', 'input2', 'input3', 'input4', 'input5']].values
y = train_df[['output1', 'output2']].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the BP model with optimal hyperparameters
class BPNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BPNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

# Instantiate the model with optimal hyperparameters
model = BPNet(input_size=5, hidden_sizes=[256, 128], output_size=2)

# Define loss function and optimizer with optimal learning rate
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training the model with optimal number of epochs and batch size
num_epochs = 100
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()
    y_test_np = y_test_tensor.numpy()
    mse = mean_squared_error(y_test_np, predictions)
    mae = mean_absolute_error(y_test_np, predictions)
    r2 = r2_score(y_test_np, predictions)
    print(f'MSE: {mse}, MAE: {mae}, R2: {r2}')