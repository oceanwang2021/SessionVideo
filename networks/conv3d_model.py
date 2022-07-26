from pydoc import cli
import torch
from torch import nn, set_flush_denormal
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Conv3d, MaxPool3d
import numpy as np


class Conv3dModel(nn.Module):
    def __init__(self, num_classes, clip_len, image_size):

        self.num_classess = num_classes
        self.clip_len = clip_len
        self.image_size = image_size
        # 调用父构造
        super(Conv3dModel, self).__init__()

        self.model_exp = Sequential(
            # 第一层
            Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # 第二层
            Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # 第三层
            Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ReLU(),
            MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # 线性层
            Flatten(),
            Linear(128 * (clip_len // 4) * ((image_size // 8) ** 2), 64 * (clip_len // 4) * ((image_size // 8) ** 2)),
            ReLU(),
            Linear(64 * (clip_len // 4) * ((image_size // 8) ** 2), 64 * (clip_len // 4) * ((image_size // 8) ** 2)),
            ReLU(),
            Linear(64 * (clip_len // 4) * ((image_size // 8) ** 2), num_classes)
        )


        
    def forward(self, X):
        X = self.model_exp(X)
        return X

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        return self(inputs).detach().cpu().numpy().argmax(axis=1)


if __name__ == '__main__':
    conv3d_model = Conv3dModel(16, 8, 32)
    # x = torch.ones((64, 3, 32, 32))
    x = torch.ones((64, 1, 8, 32, 32))
    y = conv3d_model(x)
    print(y.shape)