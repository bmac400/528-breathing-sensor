import matplotlib.pyplot as plt
import numpy as np
import time

n = 10

x = [i for i in range(n)]
plt.ion()

for i in range(100):
    time.sleep(0.5)
    if i % 10 == 0:
        plt.figure(figsize=(12,8))
        y = [np.random.random() for _ in range(n)]
        z = [np.random.random() for _ in range(n)]
        plt.suptitle("Live Data Readings")

        plt.subplot(2, 1, 1)
        plt.plot(x, y, label="y")
        plt.title("Title")
        plt.xlabel("Time")
        plt.ylabel("Y value")

        plt.subplot(2, 1, 2)
        plt.plot(x, z, label="z")
        plt.title("Title")
        plt.xlabel("Time")
        plt.ylabel("Z value")
        plt.grid()

        plt.show()
        plt.pause(0.1)
        plt.close()
