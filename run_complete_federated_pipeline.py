"""
Complete Federated Learning Pipeline with Regression Metrics Only
Implements round-based, sample-weighted FedAvg with a shared model,
Differential Privacy (noise addition), centralized baseline comparison,
patient-grouped evaluation splits, and records communication efficiency.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import json
import time
from pathlib import Path
import copy

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATASET_DIR = Path('dataset')
HOSPITALS = ['A', 'B', 'C', 'D', 'E']
BASE_FEATURES = [
    'Hemoglobin', 'RBC_Count', 'Glucose',
    'Total_Cholesterol', 'HDL', 'LDL',
    'Triglycerides', 'Creatinine', 'Urea'
]
FED_ROUNDS = 10
EPOCHS_PER_ROUND = 5
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Differential Privacy Settings
DP_ENABLED = True
NOISE_MULTIPLIER = 0.01  # scale of Gaussian noise to add to weights
CLIPPING_NORM = 1.0

# ==============================================================================
# NEURAL NETWORK MODEL
# ==============================================================================
class HealthModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def _clean_numeric_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    median_val = s.median()
    if pd.isna(median_val):
        return s.fillna(0)
    return s.fillna(median_val)

def load_hospital_data(hospital_id):
    hospital_file = DATASET_DIR / f'Hospital_{hospital_id}_Master_Records.csv'
    df = pd.read_csv(hospital_file)
    
    target_col = [c for c in df.columns if c.startswith('HbA1c')]
    if not target_col:
        return None
    target_col = target_col[0]

    final_features = [col for col in BASE_FEATURES if col in df.columns]
    
    # Soft cleaning
    for col in final_features:
        df[col] = _clean_numeric_series(df[col])
    df[target_col] = _clean_numeric_series(df[target_col])
    
    return df, final_features, target_col

def prepare_patient_split(df, final_features, target_col):
    if 'MedID' not in df.columns:
        print("MedID not found, unable to do patient-level split.")
        return None
        
    unique_patients = df['MedID'].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(unique_patients)
    
    split_idx = int(len(unique_patients) * (1 - TEST_SIZE))
    train_patients = unique_patients[:split_idx]
    test_patients = unique_patients[split_idx:]
    
    train_df = df[df['MedID'].isin(train_patients)]
    test_df = df[df['MedID'].isin(test_patients)]
    
    X_train = train_df[final_features].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32)
    X_test = test_df[final_features].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)
    
    return X_train, y_train, X_test, y_test, len(train_patients), len(test_patients)

# ==============================================================================
# STEP 1: PREPARE DATA & SCALER
# ==============================================================================
def prepare_all_data():
    hospital_data = {}
    all_X_train = []
    
    for hid in HOSPITALS:
        res = load_hospital_data(hid)
        if not res: continue
        df, feats, tcol = res
        
        split_res = prepare_patient_split(df, feats, tcol)
        if not split_res: continue
        X_train, y_train, X_test, y_test, n_train, n_test = split_res
        
        hospital_data[hid] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'features': feats
        }
        all_X_train.append(X_train)
        
    # Fit global scaler
    global_scaler = StandardScaler()
    global_scaler.fit(np.vstack(all_X_train))
    
    # Scale data
    for hid in hospital_data:
        hospital_data[hid]['X_train'] = global_scaler.transform(hospital_data[hid]['X_train'])
        hospital_data[hid]['X_test'] = global_scaler.transform(hospital_data[hid]['X_test'])
        
    return hospital_data, global_scaler

# ==============================================================================
# DIFFERENTIAL PRIVACY UTILS
# ==============================================================================
def apply_dp_to_update(update_dict):
    if not DP_ENABLED:
        return update_dict
        
    dp_update = {}
    
    # Calculate global norm of the update
    total_norm = 0.0
    for k, v in update_dict.items():
        total_norm += v.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = CLIPPING_NORM / (total_norm + 1e-6)
    clip_coef = min(1.0, clip_coef)
    
    for k, v in update_dict.items():
        clipped_v = v * clip_coef
        noise = torch.randn_like(clipped_v) * NOISE_MULTIPLIER * CLIPPING_NORM
        dp_update[k] = clipped_v + noise
        
    return dp_update

# ==============================================================================
# STEP 2: FEDERATED LEARNING
# ==============================================================================
def run_federated_learning(hospital_data):
    print(f"\n{'='*70}")
    print("STARTING ROUND-BASED FEDERATED LEARNING (FedAvg)")
    if DP_ENABLED:
        print("DIFFERENTIAL PRIVACY: ENABLED")
    print(f"{'='*70}")
    
    input_dim = len(hospital_data[HOSPITALS[0]]['features'])
    global_model = HealthModel(input_dim=input_dim)
    
    comm_bytes = 0
    start_time = time.time()
    
    # Calculate model size for communication metric
    model_size_bytes = sum(p.numel() for p in global_model.parameters()) * 4 # 32-bit floats
    
    for round_num in range(1, FED_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{FED_ROUNDS} ---")
        global_weights = global_model.state_dict()
        
        local_updates = []
        total_samples = 0
        
        for hid in HOSPITALS:
            data = hospital_data[hid]
            X_train = torch.tensor(data['X_train'], dtype=torch.float32)
            y_train = torch.tensor(data['y_train'], dtype=torch.float32).view(-1, 1)
            n_samples = len(X_train)
            
            # Send global model to client (downlink)
            comm_bytes += model_size_bytes
            
            local_model = HealthModel(input_dim=input_dim)
            local_model.load_state_dict(copy.deepcopy(global_weights))
            optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)
            criterion = nn.MSELoss()
            
            local_model.train()
            for _ in range(EPOCHS_PER_ROUND):
                optimizer.zero_grad()
                out = local_model(X_train)
                loss = criterion(out, y_train)
                loss.backward()
                optimizer.step()
                
            # Calculate weight update
            local_weights = local_model.state_dict()
            update_dict = {k: local_weights[k] - global_weights[k] for k in global_weights}
            
            # Apply Differential Privacy
            if DP_ENABLED:
                update_dict = apply_dp_to_update(update_dict)
                
            # Send update to server (uplink)
            comm_bytes += model_size_bytes
            
            local_updates.append((n_samples, update_dict))
            total_samples += n_samples
            print(f"Hospital {hid}: trained on {n_samples} samples. Loss: {loss.item():.4f}")
            
        # Aggregate updates (Sample-weighted FedAvg)
        aggregated_update = {k: torch.zeros_like(v) for k, v in global_weights.items()}
        for n_samples, update in local_updates:
            weight = n_samples / total_samples
            for k in aggregated_update:
                aggregated_update[k] += update[k] * weight
                
        # Apply aggregated update to global model
        new_global_weights = copy.deepcopy(global_weights)
        for k in new_global_weights:
            new_global_weights[k] += aggregated_update[k]
            
        global_model.load_state_dict(new_global_weights)
        
    runtime = time.time() - start_time
    print(f"\nFederated Training Complete. Runtime: {runtime:.2f}s, Comm Overhead: {comm_bytes / (1024*1024):.2f} MB")
    
    return global_model, runtime, comm_bytes

# ==============================================================================
# STEP 3: CENTRALIZED BASELINE
# ==============================================================================
def train_centralized_model(hospital_data):
    print(f"\n{'='*70}")
    print("TRAINING CENTRALIZED BASELINE MODEL")
    print(f"{'='*70}")
    
    X_train_all = np.vstack([hospital_data[h]['X_train'] for h in HOSPITALS])
    y_train_all = np.concatenate([hospital_data[h]['y_train'] for h in HOSPITALS])
    
    X_train = torch.tensor(X_train_all, dtype=torch.float32)
    y_train = torch.tensor(y_train_all, dtype=torch.float32).view(-1, 1)
    
    input_dim = len(hospital_data[HOSPITALS[0]]['features'])
    model = HealthModel(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    total_epochs = FED_ROUNDS * EPOCHS_PER_ROUND
    
    model.train()
    for epoch in range(total_epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}")
            
    return model

# ==============================================================================
# EVALUATION
# ==============================================================================
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_tensor).numpy().flatten()
        
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100)
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}, y_pred

def evaluate_and_compare(global_model, centralized_model, hospital_data):
    print(f"\n{'='*70}")
    print("EVALUATION: FEDERATED VS CENTRALIZED")
    print(f"{'='*70}")
    
    all_y_true = []
    all_y_pred_fed = []
    all_y_pred_cent = []
    
    fed_results = []
    cent_results = []
    
    for hid in HOSPITALS:
        X_test = hospital_data[hid]['X_test']
        y_test = hospital_data[hid]['y_test']
        
        fed_metrics, fed_preds = evaluate_model(global_model, X_test, y_test)
        cent_metrics, cent_preds = evaluate_model(centralized_model, X_test, y_test)
        
        all_y_true.extend(y_test)
        all_y_pred_fed.extend(fed_preds)
        all_y_pred_cent.extend(cent_preds)
        
        fed_results.append({'Hospital': hid, **fed_metrics})
        cent_results.append({'Hospital': hid, **cent_metrics})
        
        print(f"\nHospital {hid}:")
        print(f"  Federated  - RMSE: {fed_metrics['RMSE']:.4f}, R2: {fed_metrics['R2']:.4f}")
        print(f"  Centralized- RMSE: {cent_metrics['RMSE']:.4f}, R2: {cent_metrics['R2']:.4f}")
        
    all_y_true = np.array(all_y_true)
    all_y_pred_fed = np.array(all_y_pred_fed)
    all_y_pred_cent = np.array(all_y_pred_cent)
    
    avg_fed = {
        'RMSE': float(np.sqrt(mean_squared_error(all_y_true, all_y_pred_fed))),
        'MAE': float(mean_absolute_error(all_y_true, all_y_pred_fed)),
        'R2': float(r2_score(all_y_true, all_y_pred_fed)),
        'MAPE': float(np.mean(np.abs((all_y_true - all_y_pred_fed) / (all_y_true + 1e-8))) * 100)
    }
    
    avg_cent = {
        'RMSE': float(np.sqrt(mean_squared_error(all_y_true, all_y_pred_cent))),
        'MAE': float(mean_absolute_error(all_y_true, all_y_pred_cent)),
        'R2': float(r2_score(all_y_true, all_y_pred_cent)),
        'MAPE': float(np.mean(np.abs((all_y_true - all_y_pred_cent) / (all_y_true + 1e-8))) * 100)
    }
    
    print(f"\nOVERALL METRICS:")
    print(f"  Federated  - RMSE: {avg_fed['RMSE']:.4f}, R2: {avg_fed['R2']:.4f}, MAPE: {avg_fed['MAPE']:.2f}%")
    print(f"  Centralized- RMSE: {avg_cent['RMSE']:.4f}, R2: {avg_cent['R2']:.4f}, MAPE: {avg_cent['MAPE']:.2f}%")
    
    return fed_results, cent_results, avg_fed, avg_cent

def main():
    hospital_data, global_scaler = prepare_all_data()
    
    global_model, runtime, comm_bytes = run_federated_learning(hospital_data)
    centralized_model = train_centralized_model(hospital_data)
    
    fed_results, cent_results, avg_fed, avg_cent = evaluate_and_compare(global_model, centralized_model, hospital_data)
    
    results_summary = {
        'Pipeline': 'Multi-Hospital Federated Learning (Round-based DP-FedAvg)',
        'Hospitals': HOSPITALS,
        'Privacy': {
            'Method': 'Differential Privacy (Gaussian Noise on weight updates)',
            'Enabled': DP_ENABLED,
            'Noise_Multiplier': NOISE_MULTIPLIER,
            'Clipping_Norm': CLIPPING_NORM
        },
        'Communication_Efficiency': {
            'Rounds': FED_ROUNDS,
            'Epochs_per_round': EPOCHS_PER_ROUND,
            'Total_Bytes_Communicated': comm_bytes,
            'Runtime_Seconds': runtime
        },
        'Federated_Model_Results': fed_results,
        'Centralized_Model_Results': cent_results,
        'Global_Average_Federated': avg_fed,
        'Global_Average_Centralized': avg_cent
    }
    
    with open('federated_learning_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
        
    torch.save(global_model.state_dict(), 'Global_Model.pth')
    print("\nSaved federated_learning_results.json and Global_Model.pth")

if __name__ == "__main__":
    main()
