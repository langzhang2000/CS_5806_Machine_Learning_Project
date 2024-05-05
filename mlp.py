# Importing the required libraries
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Load the data
path = os.path.abspath(__file__)
train_data_path = os.path.join(os.path.dirname(path), 'INTERACTION-Dataset-DR-single-v1_2/train/DR_DEU_Merging_MT_train.csv')
test_data_path = os.path.join(os.path.dirname(path), 'INTERACTION-Dataset-DR-single-v1_2/test_single-agent/DR_DEU_Merging_MT_obs.csv')
val_data_path = os.path.join(os.path.dirname(path), 'INTERACTION-Dataset-DR-single-v1_2/val/DR_DEU_Merging_MT_val.csv')

# Load the data
def load_data(data_path, num_columns=12):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    last_known_state = defaultdict(lambda: defaultdict(list))
    with open(data_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        for row in tqdm(rows[1:], desc="Loading data"):
            case_id = int(float(row[0]))
            track_id = int(float(row[1]))
            frame_id = int(float(row[2]))
            timestamp_ms = int(float(row[3]))
            state = [timestamp_ms] + [float(x) for x in row[5:num_columns]]
            data[case_id][track_id][timestamp_ms] = state
            last_known_state[(case_id, track_id)] = state
    return data, last_known_state

# Group the data
def group_data(data):
    grouped_data = defaultdict(list)
    for case_id, case_data in tqdm(data.items(), desc="Grouping data"):
        for track_id, track_data in case_data.items():
            for timestamp_ms, state in track_data.items():
                grouped_data[(case_id, track_id)].append((timestamp_ms, state))
    return grouped_data

# Create sequences
def create_sequences(grouped_data, last_known_state, input_seq_len=10, output_seq_len=30, is_test_data=False):
    input_data = []
    output_data = []
    for (case_id, track_id), states in tqdm(grouped_data.items(), desc="Creating sequences"):
        states.sort(key=lambda x: x[0])
        for i in range(len(states) - input_seq_len - output_seq_len + 1):
            timestamps = [ts for ts, s in states[i:i + input_seq_len + output_seq_len]]
            if all(t2 - t1 == 100 for t1, t2 in zip(timestamps, timestamps[1:])):
                input_seq = []
                output_seq = []
                for j in range(i, i + input_seq_len + output_seq_len):
                    ts, s = states[j]
                    if s is not None:
                        last_known_state[(case_id, track_id)] = s
                    if j < i + input_seq_len:
                        input_seq.append(last_known_state[(case_id, track_id)])
                    else:
                        output_seq.append(last_known_state[(case_id, track_id)])
                if len(input_seq) == input_seq_len and len(output_seq) == output_seq_len:
                    input_data.append(input_seq)
                    output_data.append(output_seq)
    if not input_data or (not is_test_data and not output_data):
        raise ValueError("No valid sequences were created.")
    if is_test_data:
        return input_data, None
    else:
        return input_data, output_data
    
# Normalize the data
def normalize_data(input_data, output_data, input_scaler=None, output_scaler=None):
    input_shape = np.array(input_data).shape
    input_data_flat = np.array(input_data).reshape(-1, input_shape[1]*input_shape[2])
    
    if input_scaler is None:
        input_scaler = StandardScaler()
        input_scaler.fit(input_data_flat)
        
    input_data_norm = input_scaler.transform(input_data_flat).reshape(-1, input_shape[1], input_shape[2])

    if output_data:
        output_shape = np.array(output_data).shape
        output_data_flat = np.array(output_data).reshape(-1, output_shape[1]*output_shape[2])

        if output_scaler is None:
            output_scaler = StandardScaler()
            output_scaler.fit(output_data_flat)

        output_data_norm = output_scaler.transform(output_data_flat).reshape(-1, output_shape[1], output_shape[2])
    else:
        output_data_norm = []


    return input_data_norm, output_data_norm, input_scaler, output_scaler

# Create the dataset and dataloader
class InteractionDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

# Load, group, and create sequences for the training, validation, and test data
train_data, train_last_known_state = load_data(train_data_path)
train_grouped_data = group_data(train_data)
train_input_data, train_output_data = create_sequences(train_grouped_data, train_last_known_state, is_test_data=False)
train_input_data, train_output_data, train_input_scaler, train_output_scaler = normalize_data(train_input_data, train_output_data)

test_data, test_last_known_state = load_data(test_data_path)
test_grouped_data = group_data(test_data)
test_input_data, _ = create_sequences(test_grouped_data, test_last_known_state, input_seq_len=10, output_seq_len=0, is_test_data=True)
test_input_data, _, _, _ = normalize_data(test_input_data, [], train_input_scaler, train_output_scaler)

val_data, val_last_known_state = load_data(val_data_path)
val_grouped_data = group_data(val_data)
val_input_data, val_output_data = create_sequences(val_grouped_data, val_last_known_state, is_test_data=False)
val_input_data, val_output_data, _, _ = normalize_data(val_input_data, val_output_data, train_input_scaler, train_output_scaler)

# Create the dataset and dataloader for the training, validation, and test data
train_dataset = InteractionDataset(train_input_data, train_output_data)
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

val_dataset = InteractionDataset(val_input_data, val_output_data)
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

test_dataset = InteractionDataset(test_input_data, []) 
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Implement the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 30 * 8)

    def forward(self, x):
        x = x.view(-1, 10 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 30, 8)
        return x

# Initialize the model
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
criterion = nn.MSELoss()

train_losses = []
val_losses = []

predictions = []
actual = []

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for input_data, output_data in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        input_data = input_data.clone().detach().float()
        output_data = output_data.clone().detach().float()
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, output_data)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss)
    print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss}')

    # Update the validation loop
    model.eval()
    epoch_val_loss = 0
    for input_data, output_data in val_dataloader:
        input_data = input_data.clone().detach().float()
        output_data = output_data.clone().detach().float()
        output = model(input_data)
        output = output.view(-1, 30, 8)
        loss = criterion(output, output_data)
        epoch_val_loss += loss.item()
    epoch_val_loss /= len(val_dataloader)
    val_losses.append(epoch_val_loss)
    print(f'Epoch {epoch + 1}, Validation Loss: {epoch_val_loss}')

# Evaluate the model on the test data
target_agent = None
for i, key in enumerate(test_grouped_data.keys()):
    if key[1] == 4:
        target_agent = i
        break

if target_agent is None:
    print("No agent with track_id = 3 found in the test data.")
else:
    print(f"Target agent: {target_agent}")
model.eval()
with torch.no_grad():
    input_data = torch.FloatTensor(test_input_data[target_agent])
    output = model(input_data)

# The output is the model's prediction for the 3s future states of the target agent

############# Evaluation Metrics ################
# 1 - RSME (Root Mean Squared Error)
# 2 - MAE (Mean Absolute Error)
# 3 - MRE (Mean Relative Error)
# 4 - R2 Score (Coefficient of Determination)
# 5 - Explained Variance Score (EVS)

# Convert the output and ground truth to numpy arrays
train_predicted_output = model(torch.FloatTensor(train_input_data)).detach().numpy()
train_actual_output = np.array(train_output_data)

val_predicted_output = model(torch.FloatTensor(val_input_data)).detach().numpy()
val_actual_output = np.array(val_output_data)

train_predicted_output_flat = train_predicted_output.reshape(train_predicted_output.shape[0], -1)
train_actual_output_flat = train_actual_output.reshape(train_actual_output.shape[0], -1)

val_predicted_output_flat = val_predicted_output.reshape(val_predicted_output.shape[0], -1)
val_actual_output_flat = val_actual_output.reshape(val_actual_output.shape[0], -1)

# Calculate RMSE for the training, and validation data
train_rmse = root_mean_squared_error(train_actual_output_flat, train_predicted_output_flat)
print(f'Training RMSE: {train_rmse}')
val_rmse = root_mean_squared_error(val_actual_output_flat, val_predicted_output_flat)
print(f'Validation RMSE: {val_rmse}')

# Calculate MAE for the training, and validation data
train_mae = mean_absolute_error(train_actual_output_flat, train_predicted_output_flat)
print(f'Training MAE: {train_mae}')
val_mae = mean_absolute_error(val_actual_output_flat, val_predicted_output_flat)
print(f'Validation MAE: {val_mae}')

# Calculate MRE for the training, and validation data
epsilon = 1e-7  # Small value to avoid division by zero
train_mre = np.mean(np.abs((train_actual_output_flat - train_predicted_output_flat) / (train_actual_output_flat + epsilon)))
print(f'Training MRE: {train_mre}')
val_mre = np.mean(np.abs((val_actual_output_flat - val_predicted_output_flat) / (val_actual_output_flat + epsilon)))
print(f'Validation MRE: {val_mre}')

# Calculate R2 score for the training, and validation data
train_r2 = r2_score(train_actual_output_flat, train_predicted_output_flat)
print(f'Training R2 Score: {train_r2}')

val_r2 = r2_score(val_actual_output_flat, val_predicted_output_flat)
print(f'Validation R2 Score: {val_r2}')

# Calculate Explained Variance Score for the training, and validation data
train_evs = explained_variance_score(train_actual_output_flat, train_predicted_output_flat)
print(f'Training Explained Variance Score: {train_evs}')

val_evs = explained_variance_score(val_actual_output_flat, val_predicted_output_flat)
print(f'Validation Explained Variance Score: {val_evs}')


############# INTERACTION CHALLENGE METRICS ################
# 1 - minADE (Minimum Average Displacement Error)
# 2 - minFDE (Minimum Final Displacement Error)
# 3 - MR (Miss Rate)

# Convert the output and ground truth to numpy arrays
train_predicted_output = model(torch.FloatTensor(train_input_data)).detach().numpy()
train_actual_output = np.array(train_output_data)

val_predicted_output = model(torch.FloatTensor(val_input_data)).detach().numpy()
val_actual_output = np.array(val_output_data)

# Calculate ADE for for the training, and validation data
train_diff = train_predicted_output - train_actual_output
train_ade = np.mean(np.sqrt(np.sum(train_diff**2, axis=-1)), axis=0)
train_minADE = np.min(train_ade)
print(f'Training minADE: {train_minADE}')

val_diff = val_predicted_output - val_actual_output
val_ade = np.mean(np.sqrt(np.sum(val_diff**2, axis=-1)), axis=0)
val_minADE = np.min(val_ade)
print(f'Validation minADE: {val_minADE}')

# Calculate minFDE for the training, and validation data
train_fde = np.sqrt(np.sum((train_predicted_output[:, -1, :] - train_actual_output[:, -1, :])**2, axis=-1))
train_avgFDE = np.mean(train_fde)
print(f'Training avgFDE: {train_avgFDE}')

val_fde = np.sqrt(np.sum((val_predicted_output[:, -1, :] - val_actual_output[:, -1, :])**2, axis=-1))
val_avgFDE = np.mean(val_fde)
print(f'Validation avgFDE: {val_avgFDE}')

# Calculate Miss Rate for the training, and validation data
train_velocity = np.linalg.norm(train_actual_output[:, -1, 4:6] - train_actual_output[:, -2, 4:6], axis=-1)
train_threshold_lon = np.where(train_velocity < 1.4, 1, np.where(train_velocity <= 11, 1 + (train_velocity - 1.4) / (11 - 1.4), 2))
train_diff = train_predicted_output[:, -1, 4:6] - train_actual_output[:, -1, 4:6]
train_miss = np.any(np.abs(train_diff) > np.stack([train_threshold_lon, np.ones_like(train_threshold_lon)], axis=-1), axis=-1)
train_MR = np.mean(train_miss)
print(f'Training Miss Rate: {train_MR}')

val_velocity = np.linalg.norm(val_actual_output[:, -1, 4:6] - val_actual_output[:, -2, 4:6], axis=-1)
val_threshold_lon = np.where(val_velocity < 1.4, 1, np.where(val_velocity <= 11, 1 + (val_velocity - 1.4) / (11 - 1.4), 2))
val_diff = val_predicted_output[:, -1, 4:6] - val_actual_output[:, -1, 4:6]
val_miss = np.any(np.abs(val_diff) > np.stack([val_threshold_lon, np.ones_like(val_threshold_lon)], axis=-1), axis=-1)
val_MR = np.mean(val_miss)
print(f'Validation Miss Rate: {val_MR}')


# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot the trajectory based on normalized data
def plot_trajectory(input_data, predicted_output, actual_output, title):
    input_x = input_data[:, :, 1]
    input_y = input_data[:, :, 2]
    predicted_x = predicted_output[:, :, 1]
    predicted_y = predicted_output[:, :, 2]
    actual_x = actual_output[:, :, 1]
    actual_y = actual_output[:, :, 2]

    plt.figure()

    # Plot the input, predicted, and actual data
    plt.plot(input_x[0], input_y[0], color='blue', label='Input')
    plt.plot(predicted_x[0], predicted_y[0], color='red', label='Predicted')
    plt.plot(actual_x[0], actual_y[0], color='green', label='Actual')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Plot the trajectories
plot_trajectory(train_input_data, train_predicted_output, train_actual_output, 'Training Data')
plot_trajectory(val_input_data, val_predicted_output, val_actual_output, 'Validation Data')

# Save the model
model_path = os.path.join(os.path.dirname(path), 'model.pth')
torch.save(model.state_dict(), model_path)
