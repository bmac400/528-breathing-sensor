import socket
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq
import numpy as np
import tensorflow as tf

def formatUDPMessage(message):
    data = ["xacc","yacc","zacc","xgyro","ygyro","zgyro"]
    values = message.split(",")
    dict = {}
    for x in range(0,len(data)):
        dict[data[x]] = float(values[x])

    return message

def separate_data(data):
    xacc = [entry[0] for entry in data]
    yacc = [entry[1] for entry in data]
    zacc = [entry[2] for entry in data]
    xgyro = [entry[3] for entry in data]
    ygyro = [entry[4] for entry in data]
    zgyro = [entry[5] for entry in data]
    return xacc, yacc, zacc, xgyro, ygyro, zgyro

data_str = ["xacc","yacc","zacc","xgyro","ygyro","zgyro"]
df = pd.DataFrame(columns=data_str)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('192.168.1.47', 3333))
data = []
first = True
encoder = LabelEncoder()
encoder.fit(["abnormal","noise","normal"])
new_model = tf.keras.models.load_model('TrainedCNNModel.h5')
# enable interactive (live) plotting
plt.ion()
plt.suptitle("Live Data Readings")

while True:
    message, address = server_socket.recvfrom(1024)
    message = message.decode().upper()
    if len(data) < 400:
        #Will need to add handling to convert everything to floats
        df.append(formatUDPMessage(message), ignore_index=True)
    else:
        df = df.iloc[1:]
        df.append(formatUDPMessage(message), ignore_index=True)
        #runModel
        if "time" in df:
            df.drop(columns=["time"], inplace=True)
       
        new_data = [df.values]
        #new_data = ...  # shape (1, 400, n_features) after scaling
        predictions = new_model.predict(new_data)
        predicted_label = encoder.inverse_transform([np.argmax(predictions)])
        print(f"Predicted Label: {predicted_label}")

        # create live graphs
        # separate data 
        xacc, yacc, zacc, xgyro, ygyro, zgyro = separate_data(data)
        t = [i/100 for i in range(400)]

        # plot acceleration data
        plt.figure(figsize=(12, 8))

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

        plt.show()
        plt.pause(0.1)
        plt.close()

        
        

