import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skorch import NeuralNetClassifier
import os
import pandas as pd
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


class Classifier(nn.Module):
    def __init__(
        self,
        input_cnt=2394,
        first_cnt=1000,
        second_cnt=50,
        nonlin=F.relu,
        dropout=0.5,
    ):
        super().__init__()
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)

        self.layer0 = nn.Linear(input_cnt, first_cnt)
        self.layer1 = nn.Linear(first_cnt, second_cnt)
        self.output = nn.Linear(second_cnt, 2)

    def forward(self, X):
        X = X.to(torch.float32)
        X = self.nonlin(self.layer0(X))
        X = self.dropout(X)
        X = self.nonlin(self.layer1(X))
        return self.output(X)


net = NeuralNetClassifier(
    Classifier,
    max_epochs=10,
    criterion=nn.CrossEntropyLoss,
    lr=0.1,
    device=device,
)

directoryArr = ["Data/Abnormal", "Data/Normal"]

# Get all files in the directory with absolute paths
X = []
y = []
# Process each file
for directory in directoryArr:
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    for f in files:
        df = pd.read_csv(f)
        if df.shape[0] < 399:  # drop bad data (less than 399 rows)
            continue
        if "time" in df:
            df.drop(columns=["time"], inplace=True)
        if directory == "Data/Abnormal":
            y.append(torch.tensor(1, dtype=torch.long))
        else:
            y.append(torch.tensor(0, dtype=torch.long))
        flat_X = df.values.flatten()

        X.append(flat_X[:2394])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

net.fit(np.array(X_train), np.array(y_train))
net.score(np.array(X_train), np.array(y_train))
# net.score(np.array(X_test), np.array(y_test))
