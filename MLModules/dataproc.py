import os
import pandas as pd
import torch

def dataproc(directoryArr):
    X = []
    y = []
    for directory in directoryArr:
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]
        for f in files:
            df = pd.read_csv(f)
            if df.shape[0] != 400:
                continue
            if "time" in df:
                df.drop(columns=["time"], inplace=True)
            if directory == "Data/Abnormal":
                y.append("abnormal")
            elif directory == "Data/Noise":
                y.append("noise")
            else:
                y.append("normal")
            X.append(df.values)
    return X, y