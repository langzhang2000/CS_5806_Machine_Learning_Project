import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import numpy as np
import matplotlib.pyplot as plt

def mean_relative_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def minADE(y_true, y_pred):
    return np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)))

def minFDE(y_true, y_pred):
    return np.sqrt(np.sum((y_true[-1] - y_pred[-1]) ** 2))

def miss_rate(y_true, y_pred, speeds):
    y_true_reshaped = y_true.reshape(-1, 30, 2)
    y_pred_reshaped = y_pred.reshape(-1, 30, 2)

    distances = np.linalg.norm(y_true_reshaped - y_pred_reshaped, axis=2)

    thresholds = np.where(speeds < 1.4, 1, np.where(speeds <= 11, 1 + (speeds - 1.4) / (11 - 1.4), 2))

    misses = distances > thresholds

    return np.mean(misses)

def calculate_speed(data):
    data['speed'] = np.sqrt(data['vx']**2 + data['vy']**2)
    return data

train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')
test_data = pd.read_csv('test.csv')

train_data = calculate_speed(train_data)
val_data = calculate_speed(val_data)
test_data = calculate_speed(test_data)

scaler = StandardScaler()
output_scaler = StandardScaler()

input_cols = ['x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width']
output_cols = ['x', 'y']

def create_sequences(data, input_len, output_len):
    inputs = []
    outputs = []
    speeds = []

    for case_id in data['case_id'].unique():
        case_data = data[data['case_id'] == case_id]
        
        for track_id in case_data['track_id'].unique():
            track_data = case_data[case_data['track_id'] == track_id]
            track_data = track_data.sort_values(by='frame_id')

            for i in range(len(track_data) - input_len - output_len + 1):
                input_sequence = track_data.iloc[i:i+input_len][input_cols].values
                output_sequence = track_data.iloc[i+input_len:i+input_len+output_len][output_cols].values
                speed_sequence = track_data.iloc[i+input_len:i+input_len+output_len]['speed'].values

                inputs.append(input_sequence)
                outputs.append(output_sequence)
                speeds.append(speed_sequence)
    
    return np.array(inputs), np.array(outputs), np.array(speeds)

input_length = 10 
output_length = 30 
x_train, y_train, speeds_train = create_sequences(train_data, input_length, output_length)
x_val, y_val, speeds_val = create_sequences(val_data, input_length, output_length)
x_test, y_test, speeds_test = create_sequences(test_data, input_length, output_length)

x_train = scaler.fit_transform(x_train.reshape(-1, 7)).reshape(-1, input_length, 7)
x_val = scaler.transform(x_val.reshape(-1, 7)).reshape(-1, input_length, 7)
x_test = scaler.transform(x_test.reshape(-1, 7)).reshape(-1, input_length, 7)

y_train = y_train.reshape(-1, len(output_cols))
y_val = y_val.reshape(-1, len(output_cols))
y_test = y_test.reshape(-1, len(output_cols))

output_scaler.fit(y_train)
y_train = output_scaler.transform(y_train).reshape(-1, output_length, len(output_cols))
y_val = output_scaler.transform(y_val).reshape(-1, output_length, len(output_cols))
y_test = output_scaler.transform(y_test).reshape(-1, output_length, len(output_cols))

model_file_path = 'best_model_3.h5'

checkpoint_callback = ModelCheckpoint(
    filepath=model_file_path,
    monitor='val_loss', 
    save_best_only=True,
    mode='min',  
    verbose=1 
)

model = keras.Sequential([
    layers.GRU(64, activation='relu', input_shape=(10, 7), return_sequences=True),
    layers.GRU(64, activation='relu', return_sequences=True),
    layers.GRU(64, activation='relu', return_sequences=True),
    layers.GRU(64, activation='relu', return_sequences=True),
    layers.GRU(64, activation='relu'),
    layers.Dense(60, activation='relu'),
    layers.Dense(60)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    x_train, y_train.reshape(-1, 60),
    validation_data=(x_val, y_val.reshape(-1, 60)),
    epochs=200,
    batch_size=64,
    callbacks=[checkpoint_callback]
)

best_model = keras.models.load_model(model_file_path)

test_loss, test_mae = model.evaluate(x_test, y_test.reshape(-1, 60))
print("Test loss:", test_loss)
print("Test MAE:", test_mae)

y_train_pred = best_model.predict(x_train).reshape(-1, 60)
y_val_pred = best_model.predict(x_val).reshape(-1, 60)

y_train = y_train.reshape(-1, 60)
y_val = y_val.reshape(-1, 60)

y_train_unscaled = output_scaler.inverse_transform(y_train.reshape(-1, len(output_cols))).reshape(-1, output_length, len(output_cols))
y_val_unscaled = output_scaler.inverse_transform(y_val.reshape(-1, len(output_cols))).reshape(-1, output_length, len(output_cols))
y_train_pred_unscaled = output_scaler.inverse_transform(y_train_pred.reshape(-1, len(output_cols))).reshape(-1, output_length, len(output_cols))
y_val_pred_unscaled = output_scaler.inverse_transform(y_val_pred.reshape(-1, len(output_cols))).reshape(-1, output_length, len(output_cols))

train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mre = mean_relative_error(y_train, y_train_pred)
train_r2 = r2_score(y_train.reshape(-1, 60), y_train_pred.reshape(-1, 60))
train_ev = explained_variance_score(y_train.reshape(-1, 60), y_train_pred.reshape(-1, 60))
train_minade = minADE(y_train, y_train_pred)
train_minfde = minFDE(y_train, y_train_pred)
train_miss_rate = miss_rate(y_train_unscaled, y_train_pred_unscaled, speeds_train)

val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_mre = mean_relative_error(y_val, y_val_pred)
val_r2 = r2_score(y_val.reshape(-1, 60), y_val_pred.reshape(-1, 60))
val_ev = explained_variance_score(y_val.reshape(-1, 60), y_val_pred.reshape(-1, 60))
val_minade = minADE(y_val, y_val_pred)
val_minfde = minFDE(y_val, y_val_pred)
val_miss_rate = miss_rate(y_val_unscaled, y_val_pred_unscaled, speeds_val)

print("Training Metrics:")
print(" - RMSE:", train_rmse)
print(" - MAE:", train_mae)
print(" - MRE:", train_mre)
print(" - R^2 Score:", train_r2)
print(" - Explained Variance Score:", train_ev)
print(" - minADE:", train_minade)
print(" - minFDE:", train_minfde)
print(" - Miss Rate:", train_miss_rate)

print("\nValidation Metrics:")
print(" - RMSE:", val_rmse)
print(" - MAE:", val_mae)
print(" - MRE:", val_mre)
print(" - R^2 Score:", val_r2)
print(" - Explained Variance Score:", val_ev)
print(" - minADE:", val_minade)
print(" - minFDE:", val_minfde)
print(" - Miss Rate:", val_miss_rate)

sample_case_id = val_data['case_id'].unique()[0]
sample_track_id = val_data[val_data['case_id'] == sample_case_id]['track_id'].unique()[0]

sample_data = val_data[(val_data['case_id'] == sample_case_id) & 
                        (val_data['track_id'] == sample_track_id)]

sample_data = sample_data.sort_values(by='frame_id')

input_sequence = sample_data.iloc[:10][input_cols].values

actual_output = sample_data.iloc[10:40][output_cols].values

input_sequence_normalized = scaler.transform(input_sequence)
actual_output_normalized = output_scaler.transform(actual_output)

predicted_output_normalized = model.predict(input_sequence_normalized[np.newaxis, :, :]).reshape(-1, 2)

plt.figure(figsize=(10, 6))
plt.plot(actual_output_normalized[:, 0], actual_output_normalized[:, 1], 'g-', label='Actual')
plt.plot(predicted_output_normalized[:, 0], predicted_output_normalized[:, 1], 'r-', label='Predicted')
plt.plot(input_sequence_normalized[:, 0], input_sequence_normalized[:, 1], 'b-', label='Input')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Validation Data')
plt.legend()
plt.show()

sample_case_id = train_data['case_id'].unique()[0]
sample_track_id = train_data[train_data['case_id'] == sample_case_id]['track_id'].unique()[0]

sample_data = train_data[(train_data['case_id'] == sample_case_id) & 
                        (train_data['track_id'] == sample_track_id)]

sample_data = sample_data.sort_values(by='frame_id')

input_sequence = sample_data.iloc[:10][input_cols].values

actual_output = sample_data.iloc[10:40][output_cols].values

input_sequence_normalized = scaler.transform(input_sequence)
actual_output_normalized = output_scaler.transform(actual_output)

predicted_output_normalized = model.predict(input_sequence_normalized[np.newaxis, :, :]).reshape(-1, 2)

plt.figure(figsize=(10, 6))
plt.plot(actual_output_normalized[:, 0], actual_output_normalized[:, 1], 'g-', label='Actual')
plt.plot(predicted_output_normalized[:, 0], predicted_output_normalized[:, 1], 'r-', label='Predicted')
plt.plot(input_sequence_normalized[:, 0], input_sequence_normalized[:, 1], 'b-', label='Input')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Training Data')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()