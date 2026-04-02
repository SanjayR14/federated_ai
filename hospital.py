import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Load the hospital file
HOSPITAL_FILE = 'Hospital_E_Master_Records.csv' 
df = pd.read_csv(HOSPITAL_FILE)

# Base features we want
base_features = [
    'Hemoglobin', 'RBC_Count', 'Glucose', 
    'Total_Cholesterol', 'HDL', 'LDL', 
    'Triglycerides', 'Creatinine', 'Urea'
]

# Identify target automatically
target_col = [c for c in df.columns if c.startswith('HbA1c')][0]

# --- ROBUST CLEANING ---
final_features = []
for col in base_features:
    # Convert to numeric, force errors to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN with median. If median is NaN (empty column), fill with 0
    median_val = df[col].median()
    if pd.isna(median_val):
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna(median_val)
    
    # Only keep the feature if it's not all zeros and has variation
    # (StandardScaler fails if all values are the same)
    if df[col].std() > 0:
        final_features.append(col)

# Clean target
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df[target_col] = df[target_col].fillna(df[target_col].median() if not pd.isna(df[target_col].median()) else 0)

print(f"Features used for training: {final_features}")
print(f"Total valid samples: {len(df)}")

X = df[final_features].values
y = df[target_col].values

# 2. Scale and Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Dynamic Model Definition (Adapts to number of features)
class HealthModel(nn.Module):
    def __init__(self, input_dim):
        super(HealthModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize with the number of valid features found
model = HealthModel(input_dim=len(final_features))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # Lower LR for stability

# 4. Training Loop
print(f"Starting Training for {HOSPITAL_FILE}...")
model.train()
for epoch in range(100): # Increased epochs for better convergence
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    if torch.isnan(loss):
        print("Error: Loss became NaN. Checking data...")
        break
        
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# 5. Save Weights
hospital_id = HOSPITAL_FILE.split('_')[1]
torch.save(model.state_dict(), f'weights_{hospital_id}.pth')
print(f"Success! weights_{hospital_id}.pth created.")