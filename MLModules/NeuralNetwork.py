import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split
from dataproc import dataproc
from const import directoryArr

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

X, y = dataproc(directoryArr)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

net.fit(np.array(X_train), np.array(y_train))
net.score(np.array(X_train), np.array(y_train))
# net.score(np.array(X_test), np.array(y_test))

""" Current Output:

  epoch    train_loss    valid_acc    valid_loss     dur
-------  ------------  -----------  ------------  ------
      1        0.8639       0.4419        0.7018  0.2719
      2        0.6877       0.6512        0.6459  0.0724
      3        0.6155       0.6512        0.5700  0.1191
      4        0.5430       0.6512        0.5276  0.1162
      5        0.4992       0.6977        0.4997  0.1355
      6        0.4654       0.6977        0.5178  0.0999
      7        0.4730       0.7209        0.4710  0.0731
      8        0.3935       0.7907        0.4426  0.0953
      9        0.3892       0.7209        0.5468  0.0922
     10        0.5009       0.7442        0.4469  0.0654
"""

""" Update 1:
Not working since I changed the data processing function - Kevin
"""
