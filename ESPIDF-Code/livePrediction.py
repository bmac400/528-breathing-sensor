import socket
import pandas as pd    

def formatUDPMessage(message):
    data = ["xacc","yacc","zacc","xgyro","ygyro","zgyro"]
    values = message.split(",")
    dict = {}
    for x in range(0,len(data)):
        dict[data[x]] = values[x]

    return message

data_str = ["xacc","yacc","zacc","xgyro","ygyro","zgyro"]
df = pd.DataFrame(columns=data_str)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('192.168.1.47', 3333))
data = []
first = True
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
        print("Prediction:")

