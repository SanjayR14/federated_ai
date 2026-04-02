import torch

def aggregate_fedavg(weight_files):
    # Load the first hospital's weights to get the model structure
    global_weights = torch.load(weight_files[0])
    
    # Initialize the sum with the first file's weights
    for key in global_weights.keys():
        global_weights[key] = global_weights[key].float()
        
    # Add weights from the remaining 4 hospitals
    for i in range(1, len(weight_files)):
        local_weights = torch.load(weight_files[i])
        for key in global_weights.keys():
            global_weights[key] += local_weights[key].float()
            
    # Calculate the average (Divide by 5)
    for key in global_weights.keys():
        global_weights[key] = global_weights[key] / len(weight_files)
        
    return global_weights

# List of files produced by your 5 hospitals
files = ['weights_A.pth', 'weights_B.pth', 'weights_C.pth', 'weights_D.pth', 'weights_E.pth']

print("Aggregating weights from 5 hospitals...")
global_model_weights = aggregate_fedavg(files)

# Save the final Global Model
torch.save(global_model_weights, 'Global_Model.pth')
print("Successfully created Global_Model.pth!")