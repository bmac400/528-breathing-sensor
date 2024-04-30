import random
import socket
import time
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('192.168.1.24', 3333))
time.sleep(3)
first = True
for i in range(0,21):
    start_time = time.time()
    print("start")
    fAcc = open("up_" + str(i)+".txt","w+")
    fAcc.write("time,xacc,yacc,zacc,xgyro,ygyro,zgyro\n")
    message = ""
    while time.time() - start_time < 4:
        message, address = server_socket.recvfrom(1024)
        message = message.decode().upper()
        fAcc.write(message + "\n")

    print("stop Rest" + str(i))
    fAcc.close()
    time.sleep(4)
