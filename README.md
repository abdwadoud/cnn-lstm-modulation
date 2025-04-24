# cnn-lstm-modulation
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
import keras
import keras.backend as K
import tensorflow.keras as tk
from keras.layers import LeakyReLU
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Dataset loading and preparation
dataset_file = h5py.File("/kaggle/input/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5","r")

base_modulation_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                           'FM', 'GMSK', 'OQPSK']

# Modified modulation classes as specified
selected_modulation_classes = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', 
                              '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 
                              'OQPSK', '16APSK', '4ASK', '8ASK']

selected_classes_id = [base_modulation_classes.index(cls) for cls in selected_modulation_classes]

# Calculate SNR indices for all SNR values from -20 to 30
# The dataset has SNR values from -20 to 30 in steps of 2 dB
# Total of 26 SNR values (-20, -18, -16, ..., 26, 28, 30)
SNR_VALUES = list(range(-20, 32, 2))  # -20 to 30 in steps of 2
N_SNR = len(SNR_VALUES)  # Should be 26

# Initialize arrays
X_data = None
y_data = None
snr_data = None  # To keep track of SNR values

# Calculate samples per SNR level based on dataset structure
samples_per_snr = 4096  # From the original code

for id in selected_classes_id:
    # For each modulation class, select all SNR levels
    X_slice = dataset_file['X'][(106496*(id) + samples_per_snr*0):(106496*(id) + samples_per_snr*N_SNR)]
    y_slice = dataset_file['Y'][(106496*(id) + samples_per_snr*0):(106496*(id) + samples_per_snr*N_SNR)]
    
    # Get SNR values for this slice (just for reference)
    z_slice = dataset_file['Z'][(106496*(id) + samples_per_snr*0):(106496*(id) + samples_per_snr*N_SNR)]
    
    if X_data is not None:
        X_data = np.concatenate([X_data, X_slice], axis=0)
        y_data = np.concatenate([y_data, y_slice], axis=0)
        snr_data = np.concatenate([snr_data, z_slice], axis=0)
    else:
        X_data = X_slice
        y_data = y_slice
        snr_data = z_slice

# Reshape data for CNN processing
X_data = X_data.reshape(len(X_data), 32, 32, 2)

# Create DataFrame for modulation labels
y_data_df = pd.DataFrame(y_data)
for column in y_data_df.columns:
    if sum(y_data_df[column]) == 0:
        y_data_df = y_data_df.drop(columns=[column])

y_data_df.columns = selected_modulation_classes

# Create a DataFrame for SNR values to understand distribution
snr_df = pd.DataFrame(snr_data, columns=['SNR'])
print("SNR distribution:")
print(snr_df['SNR'].value_counts().sort_index())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_df, test_size=0.2, stratify=y_data_df)


def create_model():
    learning_rate = 0.0001
    i_input = keras.layers.Input(shape=(32,32,1))
    q_input = keras.layers.Input(shape=(32,32,1))

    cnn_q_1 = tk.layers.Conv2D(64, 3, activation=LeakyReLU(alpha=0.1))(q_input)
    cnn_q_1_2 = tk.layers.Conv2D(64, 3, activation=LeakyReLU(alpha=0.1))(cnn_q_1)
    pool_q_1 = tk.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(cnn_q_1_2)
    cnn_q_2 = tk.layers.Conv2D(128, 3, activation=LeakyReLU(alpha=0.1))(pool_q_1)
    cnn_q_2_2 = tk.layers.Conv2D(128, 3, activation=LeakyReLU(alpha=0.1))(cnn_q_2)
    pool_q_2 = tk.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(cnn_q_2_2)
    flatten_q = tk.layers.Flatten()(pool_q_2)
    
    cnn_i_1 = tk.layers.Conv2D(64, 3, activation=LeakyReLU(alpha=0.1))(i_input)
    cnn_i_1_2 = tk.layers.Conv2D(64, 3, activation=LeakyReLU(alpha=0.1))(cnn_i_1)
    pool_i_1 = tk.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(cnn_i_1_2)
    cnn_i_2 = tk.layers.Conv2D(128, 3, activation=LeakyReLU(alpha=0.1))(pool_i_1)
    cnn_i_2_2 = tk.layers.Conv2D(128, 3, activation=LeakyReLU(alpha=0.1))(cnn_i_2)
    pool_i_2 = tk.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(cnn_i_2_2)
    flatten_i = tk.layers.Flatten()(pool_i_2)
    
    concat = keras.layers.concatenate([flatten_q, flatten_i])
    
    reshape = keras.layers.Reshape((1, -1))(concat)
    lstm = tk.layers.LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)(reshape)
    
    dense1 = keras.layers.Dense(1024, activation=LeakyReLU(alpha=0.1))(lstm)
    dropout1 = tk.layers.Dropout(0.5)(dense1)
    
    dense2 = keras.layers.Dense(1024, activation=LeakyReLU(alpha=0.1))(dropout1)
    dropout2 = tk.layers.Dropout(0.5)(dense2)
    
    dense3 = keras.layers.Dense(1024, activation=LeakyReLU(alpha=0.1))(dropout2)
    dropout3 = tk.layers.Dropout(0.5)(dense3)
    
    dense4 = keras.layers.Dense(256, activation=LeakyReLU(alpha=0.1))(dropout3)
    dropout4 = tk.layers.Dropout(0.5)(dense4)
    
    dense5 = keras.layers.Dense(32, activation=LeakyReLU(alpha=0.1))(dropout4)
    outputs = keras.layers.Dense(len(selected_modulation_classes), activation='softmax')(dense5)
    
    model = keras.Model(inputs=[i_input, q_input], outputs=outputs)
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=tk.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model


model = create_model()
batch_size = 64
epochs = 25

path_checkpoint = "model_checkpoint.weights.h5"
es_callback = tk.callbacks.EarlyStopping(monitor="accuracy", min_delta=0, patience=10)

modelckpt_callback = tk.callbacks.ModelCheckpoint(
    monitor="accuracy",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

# Add a callback to analyze per-SNR performance during training
class SNRPerformanceCallback(tk.callbacks.Callback):
    def __init__(self, X_test, y_test, snr_test, snr_values):
        self.X_test = X_test
        self.y_test = y_test
        self.snr_test = snr_test
        self.snr_values = snr_values
        self.snr_accuracies = {snr: [] for snr in snr_values}
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Check every 5 epochs to save time
            print(f"\nEvaluating SNR performance at epoch {epoch}:")
            for snr in self.snr_values:
                # Get indices for this SNR
                indices = np.where(self.snr_test == snr)[0]
                if len(indices) > 0:
                    X_snr = self.X_test[indices]
                    y_snr = self.y_test.iloc[indices]
                    
                    # Evaluate on this SNR
                    loss, acc = self.model.evaluate(
                        [X_snr[:,:,:,0], X_snr[:,:,:,1]], 
                        y_snr, 
                        verbose=0
                    )
                    self.snr_accuracies[snr].append(acc)
                    print(f"SNR {snr} dB: {acc:.4f}")

# Split test data by SNR for evaluation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Train the model
history = model.fit(
    x=[X_train[:,:,:,0], X_train[:,:,:,1]],
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([X_val[:,:,:,0], X_val[:,:,:,1]], y_val),
    callbacks=[es_callback, modelckpt_callback],
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.show()

# Make predictions on test set
model_predictions = model.predict([X_test[:,:,:,0], X_test[:,:,:,1]])

# Convert predictions to class indices
y_pred = np.argmax(model_predictions, axis=1)
y_true = np.argmax(y_test.values, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 20))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=selected_modulation_classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Evaluate model on test set
loss, acc = model.evaluate([X_test[:,:,:,0], X_test[:,:,:,1]], y_test, verbose=2)
print("Final model accuracy: {:.2f}%".format(100 * acc))

# Save model weights
model.save_weights('/kaggle/working/CNN_LSTMmodel_extended.weights.h5')

# Analyze performance by SNR
# Assuming we have SNR values for the test set (you would need to modify earlier code to save this)
snr_test = snr_data[X_test.shape[0]:X_test.shape[0]*2]  # This is just a placeholder
snr_values = np.unique(snr_data)

accuracy_by_snr = {}
for snr in snr_values:
    # Get indices for this SNR
    indices = np.where(snr_test == snr)[0]
    if len(indices) > 0:
        X_snr = X_test[indices]
        y_snr = y_test.iloc[indices]
        
        # Evaluate on this SNR
        loss, acc = model.evaluate([X_snr[:,:,:,0], X_snr[:,:,:,1]], y_snr, verbose=0)
        accuracy_by_snr[snr] = acc

# Plot accuracy by SNR
plt.figure(figsize=(10, 6))
snrs = list(accuracy_by_snr.keys())
accs = list(accuracy_by_snr.values())
plt.plot(snrs, accs, 'bo-')
plt.grid()
plt.xlabel('Signal to Noise Ratio (dB)')
plt.ylabel('Classification Accuracy')
plt.title('Classification Accuracy vs SNR')
plt.xticks(snrs)
plt.tight_layout()
plt.show()

# Calculate per-class accuracy by SNR
class_acc_by_snr = {}
for snr in snr_values:
    indices = np.where(snr_test == snr)[0]
    if len(indices) > 0:
        X_snr = X_test[indices]
        y_true_snr = y_test.iloc[indices].values
        y_pred_snr = model.predict([X_snr[:,:,:,0], X_snr[:,:,:,1]])
        
        # Calculate accuracy for each class
        class_acc = []
        for i in range(len(selected_modulation_classes)):
            class_indices = np.where(np.argmax(y_true_snr, axis=1) == i)[0]
            if len(class_indices) > 0:
                class_correct = np.sum(np.argmax(y_pred_snr[class_indices], axis=1) == i)
                class_acc.append(class_correct / len(class_indices))
            else:
                class_acc.append(np.nan)
        
        class_acc_by_snr[snr] = class_acc

# Plot per-class accuracy by SNR
plt.figure(figsize=(14, 8))
for i, mod in enumerate(selected_modulation_classes):
    accs = [class_acc_by_snr[snr][i] for snr in snr_values if snr in class_acc_by_snr]
    snrs = [snr for snr in snr_values if snr in class_acc_by_snr]
    plt.plot(snrs, accs, 'o-', label=mod)

plt.grid()
plt.xlabel('Signal to Noise Ratio (dB)')
plt.ylabel('Classification Accuracy')
plt.title('Per-Class Classification Accuracy vs SNR')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Create a function to determine the best modulation scheme for a given SNR value
def recommend_best_modulation(snr, class_acc_by_snr, selected_modulation_classes):
    """
    Determines the best modulation scheme for a given SNR value
    based on classification accuracy data.
    
    Args:
        snr: Signal-to-Noise Ratio value
        class_acc_by_snr: Dictionary with SNR values as keys and lists of class accuracies as values
        selected_modulation_classes: List of modulation class names
        
    Returns:
        Tuple of (best_modulation, accuracy)
    """
    # Find the closest SNR value in our data
    available_snrs = list(class_acc_by_snr.keys())
    closest_snr = min(available_snrs, key=lambda x: abs(x - snr))
    
    # Get accuracies for that SNR
    accuracies = class_acc_by_snr[closest_snr]
    
    # Find the modulation with highest accuracy
    best_idx = np.argmax(accuracies)
    best_modulation = selected_modulation_classes[best_idx]
    best_accuracy = accuracies[best_idx]
    
    return best_modulation, best_accuracy, closest_snr

# Create a lookup table for best modulation by SNR
print("\nGenerating best modulation lookup table by SNR...")
snr_modulation_table = {}

for snr in sorted(snr_values):
    if snr in class_acc_by_snr:
        best_mod, accuracy, _ = recommend_best_modulation(snr, class_acc_by_snr, selected_modulation_classes)
        snr_modulation_table[snr] = (best_mod, accuracy)
        print(f"SNR {snr} dB: Best modulation is {best_mod} with accuracy {accuracy:.4f}")

# Plot the best modulation schemes across SNR range
plt.figure(figsize=(14, 8))
snrs = sorted(snr_modulation_table.keys())
best_mods = [snr_modulation_table[snr][0] for snr in snrs]
best_accs = [snr_modulation_table[snr][1] for snr in snrs]

# Create a colormap for different modulation schemes
unique_mods = list(set(best_mods))
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_mods)))
color_map = {mod: colors[i] for i, mod in enumerate(unique_mods)}

# Plot each modulation scheme's region
for mod in unique_mods:
    mod_snrs = [snr for snr, (best_mod, _) in snr_modulation_table.items() if best_mod == mod]
    if mod_snrs:
        mod_accs = [snr_modulation_table[snr][1] for snr in mod_snrs]
        plt.plot(mod_snrs, mod_accs, 'o-', label=mod, color=color_map[mod], linewidth=2)

plt.grid(True)
plt.xlabel('Signal to Noise Ratio (dB)')
plt.ylabel('Best Modulation Accuracy')
plt.title('Best Modulation Scheme by SNR')
plt.legend(title="Modulation Scheme", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Create an interactive predictor function
def predict_best_modulation_for_snr(snr_value):
    """
    Predicts the best modulation scheme for a given SNR value.
    
    Args:
        snr_value: Signal-to-Noise Ratio value
        
    Returns:
        Dictionary with prediction details
    """
    best_mod, accuracy, closest_reference_snr = recommend_best_modulation(
        snr_value, class_acc_by_snr, selected_modulation_classes
    )
    
    # Get top 3 modulation schemes for this SNR
    if closest_reference_snr in class_acc_by_snr:
        accuracies = class_acc_by_snr[closest_reference_snr]
        top_indices = np.argsort(accuracies)[-3:][::-1]  # Top 3, highest first
        alternatives = [
            {
                "modulation": selected_modulation_classes[idx],
                "accuracy": accuracies[idx]
            }
            for idx in top_indices if selected_modulation_classes[idx] != best_mod
        ]
    else:
        alternatives = []
    
    return {
        "snr_requested": snr_value,
        "closest_reference_snr": closest_reference_snr,
        "best_modulation": best_mod,
        "expected_accuracy": accuracy,
        "alternative_options": alternatives[:2]  # Return top 2 alternatives
    }

# Example usage
test_snr_values = [-18, -10, 0, 10, 20, 30]
print("\nTesting modulation prediction for different SNR values:")
for test_snr in test_snr_values:
    result = predict_best_modulation_for_snr(test_snr)
    print(f"\nFor SNR = {test_snr} dB:")
    print(f"  - Best modulation: {result['best_modulation']} (accuracy: {result['expected_accuracy']:.4f})")
    print(f"  - Based on reference SNR: {result['closest_reference_snr']} dB")
    if result['alternative_options']:
        print("  - Alternative options:")
        for i, alt in enumerate(result['alternative_options'], 1):
            print(f"    {i}. {alt['modulation']} (accuracy: {alt['accuracy']:.4f})")

# Create a function to visualize SNR-based modulation regions
def plot_snr_modulation_regions():
    plt.figure(figsize=(12, 6))
    
    # Create a colormap for different modulation schemes
    unique_mods = list(set([info[0] for info in snr_modulation_table.values()]))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_mods)))
    color_map = {mod: colors[i] for i, mod in enumerate(unique_mods)}
    
    # Plot SNR regions with their best modulation
    snrs = sorted(snr_modulation_table.keys())
    
    for i, snr in enumerate(snrs):
        mod, acc = snr_modulation_table[snr]
        
        # Determine region boundaries
        left = snr - 1 if i == 0 else (snr + snrs[i-1]) / 2
        right = snr + 1 if i == len(snrs)-1 else (snr + snrs[i+1]) / 2
        
        # Plot the region
        plt.axvspan(left, right, alpha=0.3, color=color_map[mod])
        plt.text(snr, 0.5, mod, ha='center', va='center', fontsize=9,
                rotation=90, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.ylim(0, 1)
    plt.xlim(min(snrs)-2, max(snrs)+2)
    plt.title('Optimal Modulation Regions by SNR')
    plt.xlabel('Signal to Noise Ratio (dB)')
    plt.yticks([])
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

# Plot the SNR modulation regions
plot_snr_modulation_regions()

# Save the SNR modulation table to a CSV file
snr_modulation_df = pd.DataFrame([
    {"SNR": snr, "Best_Modulation": mod, "Accuracy": acc}
    for snr, (mod, acc) in snr_modulation_table.items()
])
snr_modulation_df = snr_modulation_df.sort_values("SNR")
snr_modulation_df.to_csv("best_modulation_by_snr.csv", index=False)
print("\nSaved best modulation lookup table to 'best_modulation_by_snr.csv'")

# Create a simple function for real-time prediction
def get_best_modulation(snr):
    """Simple function to look up the best modulation for a given SNR value.
    
    Args:
        snr: SNR value in dB
        
    Returns:
        Best modulation scheme name
    """
    prediction = predict_best_modulation_for_snr(snr)
    return prediction["best_modulation"]

# Add a simple interactive test
if __name__ == "__main__":
    print("\nInteractive SNR-based modulation predictor")
    print("------------------------------------------")
    while True:
        try:
            user_input = input("\nEnter SNR value in dB (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
                
            snr_value = float(user_input)
            prediction = predict_best_modulation_for_snr(snr_value)
            
            print(f"\nRecommendation for SNR = {snr_value} dB:")
            print(f"  Best modulation: {prediction['best_modulation']}")
            print(f"  Expected accuracy: {prediction['expected_accuracy']:.2%}")
            
            if prediction['alternative_options']:
                print("  Alternative options:")
                for i, alt in enumerate(prediction['alternative_options'], 1):
                    print(f"    {i}. {alt['modulation']} (accuracy: {alt['accuracy']:.2%})")
                    
        except ValueError:
            print("Please enter a valid SNR value (a number).")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

# ====================== NEW CODE: REINFORCEMENT LEARNING COMPONENTS ======================

# Define a ModulationEnvironment class that implements the gym interface
class ModulationEnvironment(gym.Env):
    """
    Custom Environment for adaptive modulation selection
    """
    def __init__(self, snr_modulation_table, selected_modulation_classes):
        super(ModulationEnvironment, self).__init__()
        
        # Store the modulation table and classes
        self.snr_modulation_table = snr_modulation_table
        self.modulation_classes = selected_modulation_classes
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(selected_modulation_classes))
        
        # Observation space: SNR value (continuous) and channel state info (optional)
        self.observation_space = spaces.Box(
            low=np.array([-30.0]),  # Minimum SNR
            high=np.array([40.0]),  # Maximum SNR
            dtype=np.float32
        )
        
        # Current state
        self.current_snr = None
        self.current_channel_quality = None
        self.step_count = 0
        self.max_steps = 100
        
        # Define modulation parameters: approximate spectral efficiency (bits/symbol)
        self.spectral_efficiency = {
            'BPSK': 1.0,
            'QPSK': 2.0,
            '8PSK': 3.0,
            '16PSK': 4.0,
            '32PSK': 5.0,
            '16QAM': 4.0,
            '32QAM': 5.0,
            '64QAM': 6.0,
            '128QAM': 7.0,
            '256QAM': 8.0,
            'OQPSK': 2.0,
            '16APSK': 4.0,
            '4ASK': 2.0,
            '8ASK': 3.0
        }
        
        # Define a simplified BER model based on modulation type and SNR
        # These are approximate threshold SNRs for achieving BER of 10^-5
        self.snr_thresholds = {
            'BPSK': 9.6,
            'QPSK': 12.5,
            '8PSK': 18.0,
            '16PSK': 24.0,
            '32PSK': 30.0,
            '16QAM': 18.0,
            '32QAM': 23.0,
            '64QAM': 28.0,
            '128QAM': 32.0,
            '256QAM': 36.0,
            'OQPSK': 12.5,
            '16APSK': 19.0,
            '4ASK': 14.0,
            '8ASK': 20.0
        }
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        # Reset the environment with a random SNR
        self.current_snr = np.random.uniform(-20.0, 30.0)
        self.step_count = 0
        return np.array([self.current_snr]), {}
    
    def step(self, action):
        # Execute one step in the environment
        self.step_count += 1
        
        # Get the selected modulation scheme
        selected_modulation = self.modulation_classes[action]
        
        # Calculate reward based on throughput and error probability
        reward = self._calculate_reward(selected_modulation, self.current_snr)
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        # Optionally change SNR for next step to simulate changing channel
        if not done:
            # Simulate channel variations
            self.current_snr += np.random.normal(0, 1.0)  # Add Gaussian noise
            self.current_snr = np.clip(self.current_snr, -20.0, 30.0)  # Clip to valid range
        
        return np.array([self.current_snr]), reward, done, False, {}
    
    def _calculate_reward(self, modulation, snr):
        """
        Calculate reward based on:
        1. Spectral efficiency (throughput)
        2. Error probability (BER)
        3. Power consumption (optional)
        
        We want to maximize throughput while keeping BER below a target value.
        """
        # Get the spectral efficiency (bits/symbol)
        spectral_eff = self.spectral_efficiency[modulation]
        
        # Calculate approximate BER based on modulation and SNR
        # This is a simplified model - in reality, you'd use theoretical or empirical BER curves
        ber = self._estimate_ber(modulation, snr)
        
        # Calculate effective throughput (considering packet errors)
        # Assuming packet size of 1000 bits
        packet_size = 1000
        packet_error_rate = 1 - (1 - ber) ** packet_size
        effective_throughput = spectral_eff * (1 - packet_error_rate)
        
        # Calculate reward components
        throughput_reward = effective_throughput
        
        # Penalize high BER
        ber_penalty = 0
        target_ber = 1e-5
        if ber > target_ber:
            ber_penalty = -10 * (ber / target_ber)
        
        # Combine rewards
        # We can adjust these weights based on system priorities
        total_reward = throughput_reward + ber_penalty
        
        # Add a bonus for selecting the optimal modulation according to your model
        closest_snr = min(self.snr_modulation_table.keys(), key=lambda x: abs(x - snr))
        best_mod, _ = self.snr_modulation_table[closest_snr]
        if modulation == best_mod:
            total_reward += 1.0  # Bonus for selecting what your model thinks is best
        
        return total_reward
    
    def _estimate_ber(self, modulation, snr):
        """
        Estimate BER based on modulation type and SNR.
        This is a simplified model using threshold SNRs and exponential decay.
        """
        threshold_snr = self.snr_thresholds[modulation]
        
        if snr >= threshold_snr:
            # BER decreases exponentially when SNR is above threshold
            return 1e-5 * np.exp(-(snr - threshold_snr) / 2)
        else:
            # BER increases exponentially when SNR is below threshold
            return 1e-5 * np.exp((threshold_snr - snr) / 2)


# Define a simple DQN agent for the modulation selection
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model
    
    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            act_values = self.model(state_tensor)
            return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            
            with torch.no_grad():
                target = self.model(state_tensor)
                t = target.clone()
                
                if done:
                    t[0][action] = reward
                else:
                    next_actions = self.target_model(next_state_tensor)
                    t[0][action] = reward + self.gamma * torch.max(next_actions[0]).item()
            
            states.append(state)
            targets.append(t)
        
        states = torch.FloatTensor(np.array(states).reshape(-1, self.state_size))
        targets = torch.cat(targets, dim=0)
        
        # Train the model
        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


# Add a function to train the RL agent for adaptive modulation
def train_rl_agent(env, agent, episodes=1000, batch_size=32):
    """
    Train the reinforcement learning agent for adaptive modulation.
    
    Args:
        env: The ModulationEnvironment instance
        agent: The DQNAgent instance
        episodes: Number of training episodes
        batch_size: Batch size for experience replay
        
    Returns:
        List of rewards per episode
    """
    rewards_history = []
    
    for e in range(episodes):
        # Reset the environment
        state, _ = env.reset()
        state = np.reshape(state, [1, 1])  # Reshape for the neural network
        total_reward = 0
        
        for time in range(env.max_steps):
            # Select action
            action = agent.act(state)
            
            # Take action and observe result
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 1])
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            
            # Learn from experience
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
            
            if done:
                break
                
        # Update target network every few episodes
        if e % 10 == 0:
            agent.update_target_model()
            
        # Record rewards
        rewards_history.append(total_reward)
        
        # Print progress
        if e % 100 == 0:
            print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    return rewards_history


# Function to test the trained agent
def test_agent(env, agent, episodes=100):
    """
    Test the trained agent's performance.
    
    Args:
        env: The ModulationEnvironment instance
        agent: The trained DQNAgent
        episodes: Number of test episodes
        
    Returns:
        Average reward and selected modulations
    """
    total_rewards = []
    modulation_selections = {}
    snr_modulation_pairs = []
    
    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, 1])
        rewards = 0
        
        for time in range(env.max_steps):
            # Use the trained policy (no exploration)
            action = agent.act(state) if np.random.rand() > 0.05 else random.randrange(agent.action_size)
            selected_mod = env.modulation_classes[action]
            
            # Record the SNR and selected modulation
            snr = state[0][0]
            snr_modulation_pairs.append((snr, selected_mod))
            
            # Update modulation selection count
            if selected_mod in modulation_selections:
                modulation_selections[selected_mod] += 1
            else:
                modulation_selections[selected_mod] = 1
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 1])
            rewards += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(rewards)
    
    # Calculate average reward
    avg_reward = sum(total_rewards) / len(total_rewards)
    
    # Sort modulation selections by frequency
    sorted_selections = {k: v for k, v in sorted(
        modulation_selections.items(), 
        key=lambda item: item[1], 
        reverse=True
    )}
    
    # Analyze SNR to modulation mapping
    snr_ranges = {}
    for snr, mod in snr_modulation_pairs:
        snr_bin = round(snr * 2) / 2  # Round to nearest 0.5
        if snr_bin not in snr_ranges:
            snr_ranges[snr_bin] = {}
        
        if mod in snr_ranges[snr_bin]:
            snr_ranges[snr_bin][mod] += 1
        else:
            snr_ranges[snr_bin][mod] = 1
    
    # Find most common modulation for each SNR bin
    snr_mod_mapping = {}
    for snr, mods in sorted(snr_ranges.items()):
        most_common = max(mods.items(), key=lambda x: x[1])
        snr_mod_mapping[snr] = most_common[0]
    
    return avg_reward, sorted_selections, snr_mod_mapping


# Add this to your main code to train and evaluate the RL agent
def add_rl_adaptive_modulation(snr_modulation_table, selected_modulation_classes):
    """
    Add reinforcement learning-based adaptive modulation capabilities.
    
    Args:
        snr_modulation_table: Dictionary mapping SNR values to best modulation
        selected_modulation_classes: List of available modulation schemes
        
    Returns:
        Trained agent and environment
    """
    print("\nInitializing reinforcement learning for adaptive modulation...")
    
    # Create environment
    env = ModulationEnvironment(snr_modulation_table, selected_modulation_classes)
    
    # Create agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Train the agent
    print("Training RL agent...")
    rewards = train_rl_agent(env, agent, episodes=500)
    
    # Test the agent
    print("\nTesting RL agent performance...")
    avg_reward, mod_selections, snr_mapping = test_agent(env, agent)
    
    print(f"Average reward: {avg_reward:.2f}")
    print("\nModulation selection frequency:")
    for mod, count in mod_selections.items():
        print(f"  {mod}: {count}")
    
    print("\nLearned SNR to modulation mapping:")
    for snr, mod in sorted(snr_mapping.items()):
        print(f"  SNR = {snr} dB: {mod}")
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('RL Agent Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('rl_training_progress.png')
    plt.show()
    
    # Plot learned policy vs. model-based policy
    plt.figure(figsize=(12, 6))
    
    # Sort SNRs
    snrs_learned = sorted(snr_mapping.keys())
    mods_learned = [snr_mapping[snr] for snr in snrs_learned]
    
    # Get model-based policy
    snrs_model = sorted(snr_modulation_table.keys())
    mods_model = [snr_modulation_table[snr][0] for snr in snrs_model]
    
    # Create unique modulation list for color mapping
    all_mods = list(set(mods_learned + mods_model))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_mods)))
    color_map = {mod: colors[i] for i, mod in enumerate(all_mods)}
    
    # Plot learned policy
    plt.subplot(1, 2, 1)
    for i, (snr, mod) in enumerate(zip(snrs_learned, mods_learned)):
        if i < len(snrs_learned) - 1:
            plt.axvspan(snr, snrs_learned[i+1], alpha=0.3, color=color_map[mod])
        else:
            plt.axvspan(snr, snr+1, alpha=0.3, color=color_map[mod])
        
        if i % 3 == 0:  # Label every 3rd for clarity
            plt.text(snr, 0.5, mod, ha='center', va='center', fontsize=8,
                    rotation=90, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.ylim(0, 1)
    plt.xlim(min(snrs_learned)-1, max(snrs_learned)+1)
    plt.title('RL Agent Policy')
    plt.xlabel('Signal to Noise Ratio (dB)')
    plt.yticks([])
    plt.grid(True, axis='x')
    
    # Plot model-based policy
    plt.subplot(1, 2, 2)
    for i, (snr, mod) in enumerate(zip(snrs_model, mods_model)):
        if i < len(snrs_model) - 1:
            plt.axvspan(snr, snrs_model[i+1], alpha=0.3, color=color_map[mod])
        else:
            plt.axvspan(snr, snr+1, alpha=0.3, color=color_map[mod])
        
        if i % 3 == 0:  # Label every 3rd for clarity
            plt.text(snr, 0.5, mod, ha='center', va='center', fontsize=8,
                    rotation=90, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.ylim(0, 1)
    plt.xlim(min(snrs_model)-1, max(snrs_model)+1)
    plt.title('Model-Based Policy')
    plt.xlabel('Signal to Noise Ratio (dB)')
    plt.yticks([])
    plt.grid(True, axis='x')
    
    plt.tight_layout()
    plt.savefig('rl_vs_model_policy.png')
    plt.show()
    
    return agent, env

# Add this function call to your main code where you want to integrate RL
# For example, after you've created your snr_modulation_table:
# agent, env = add_rl_adaptive_modulation(snr_modulation_table, selected_modulation_classes)

# Create a function to use the trained RL agent for modulation selection
def rl_select_modulation(agent, snr):
    """
    Use the trained RL agent to select the best modulation for a given SNR.
    
    Args:
        agent: Trained DQNAgent instance
        snr: Current SNR value
        
    Returns:
        Selected modulation index
    """
    state = np.reshape(np.array([snr]), [1, 1])
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state)
        action_values = agent.model(state_tensor)
        return torch.argmax(action_values[0]).item()
