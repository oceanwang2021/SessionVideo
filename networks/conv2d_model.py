import numpy as np
import torch
from torch import imag, nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU
import numpy as np


class Conv2dModel(nn.Module):

    def __init__(self, num_classes, clip_len, image_size) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.clip_len = clip_len
        self.image_size = image_size

        self.model2022 = Sequential(
            # 第一层
            Conv2d(1, 32, 3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 第二层
            Conv2d(32, 64, 3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # 第三层
            Conv2d(64, 64, 3, stride=1, padding=1),
            ReLU(),

            # 线性层
            Flatten(),
            Linear(64 * ((image_size // 4) * (image_size // 4)), 64),
            ReLU(),
            Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.model2022(x)
        return x
    
    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        return self(inputs).detach().cpu().numpy().argmax(axis=1)


if __name__ == '__main__':

    print(torch.__version__)
    conv2d_model = Conv2dModel(10, clip_len=1, image_size=32)
    X = torch.ones((64, 1, 1, 32, 32))
    print(X.shape)
    X = torch.reshape(X, (64, 1, 32, 32))
    print(X.shape)
    y = conv2d_model(X)
    print(y.shape)