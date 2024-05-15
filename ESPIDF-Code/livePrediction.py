import socket
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq
import numpy as np
from copy import copy
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout


def formatUDPMessage(message):
    data = ["xacc","yacc","zacc","xgyro","ygyro","zgyro"]
    values = message.split(",")
    dict = {}
    for x in range(0,len(data)):
        dict[data[x]] = float(values[x])


    return pd.DataFrame(dict, index=[0])

def separate_data(data):
    xacc = data["xacc"]
    yacc = data["yacc"]
    zacc = data["zacc"]
    xgyro = data["xgyro"]
    ygyro = data["ygyro"]
    zgyro = data["zgyro"]
    return xacc, yacc, zacc, xgyro, ygyro, zgyro
scaler = StandardScaler()

data_str = ["xacc","yacc","zacc","xgyro","ygyro","zgyro"]
df = pd.DataFrame(columns=data_str)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('192.168.1.9', 3333))
data = []
first = True
encoder = LabelEncoder()
encoder.fit(["abnormal","noise","normal"])
new_model = tf.keras.models.load_model('TrainedCNNModel.h5')
# enable interactive (live) plotting
plt.ion()
plt.show()
plt.figure(figsize=(12, 8))
plt.suptitle("Live Data Readings")
oldTime = time.time()
i = 0
while True:
    print(time.time()-oldTime)
    oldTime = time.time()
    i += 1
    message, address = server_socket.recvfrom(1024)
    message = message.decode().upper()

    if len(df.index) <= 400:
        #Will need to add handling to convert everything to floats
        df = pd.concat([df,formatUDPMessage(message)], ignore_index=True)
    else:
        df = df.iloc[1:]
        
        if i >= 100:
            i = 0
            y = df.values
            z = []
            z.append(y)
            new_data = np.array(z)
            X = new_data
            print(len(new_data))
            X_reshaped = X.reshape(-1, X.shape[-1])
            #X_scaled = scaler.fit_transform(X_reshaped)
            #X_scaled = X_scaled.reshape(X.shape)  # Reshape back to original dimensions

            #new_data = ...  # shape (1, 400, n_features) after scaling
            predictions = new_model.predict(X)
            prediction_label = encoder.inverse_transform([np.argmax(predictions)])[0]
            print("Predicted Label:"  + str(prediction_label))
            print(predictions)
            # create live graphs
            # separate data 

            xacc, yacc, zacc, xgyro, ygyro, zgyro = separate_data(df)
            t = [i/100 for i in range(400)]
            # plot acceleration data
            plt.clf()
            plt.suptitle(prediction_label)

            plt.subplot(2, 2, 1)
            plt.plot(t, xacc, label='X Acceleration')
            plt.plot(t, yacc, label='Y Acceleration')
            plt.plot(t, zacc, label='Z Acceleration')
            plt.title('Acceleration Data')
            plt.xlabel('Time (1/100 s)')
            plt.ylabel('Acceleration (m/s^2)')
            plt.legend()

            # plot gyroscope data
            plt.subplot(2, 2, 2)
            plt.plot(t, xgyro, label='X Gyroscope')
            plt.plot(t, ygyro, label='Y Gyroscope')
            plt.plot(t, zgyro, label='Z Gyroscope')
            plt.title('Gyroscope Data')
            plt.xlabel('Time (1/100 s)')
            plt.ylabel('Gyration (deg/s)')
            plt.legend()

            # plot spectrograms of x acceleration
            plt.subplot(2, 2, 3)
            f, ts, Sxx = spectrogram(np.array(xacc), fs=100, nperseg=50, noverlap=25)
            plt.pcolormesh(ts, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.title('Spectrogram of X Acceleration')
            plt.xlabel('Time (1/100 s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Power (dB)')

            # plot spectrograms of x gyroscope
            plt.subplot(2, 2, 4)
            f, ts, Sxx = spectrogram(np.array(xgyro), fs=100, nperseg=50, noverlap=25)
            plt.pcolormesh(ts, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.title('Spectrogram of X Gyroscope')
            plt.xlabel('Time (1/100 s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Power (dB)')

            plt.pause(0.001)
        df = pd.concat([df,formatUDPMessage(message)], ignore_index=True)

        
        

