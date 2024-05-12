from dataproc import dataproc
from const import directoryArrFull
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

X, y = dataproc(directoryArrFull)
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])  # Flatten the time steps and features for scaling
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(X.shape)  # Reshape back to original dimensions

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(3, activation='softmax')  # Output layer: 3 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Save model
model.save('sensor_model.h5')

"""
# Load and predict new data
new_model = tf.keras.models.load_model('sensor_model.h5')
new_data = ...  # shape (1, 400, n_features) after scaling
predictions = new_model.predict(new_data)
predicted_label = encoder.inverse_transform([np.argmax(predictions)])
print(f"Predicted Label: {predicted_label}")
"""