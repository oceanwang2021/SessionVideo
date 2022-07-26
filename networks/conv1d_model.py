import torch
from torch import nn
from torch.nn import Sequential, Conv1d, MaxPool1d, Flatten, Linear, ReLU
import numpy as np
from torchsummary import summary


class Conv1dModel(nn.Module):

    def __init__(self, num_classes, clip_len, image_size) -> None:
        
        super().__init__()

        self.num_classes = num_classes
        self.clip_len = clip_len
        self.image_size = image_size  

        self.deep_packet = Sequential(

             # 第一层
            Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool1d(kernel_size=4, stride=4),

            # 第二层
            Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            ReLU(),
            MaxPool1d(kernel_size=4, stride=4),

            # 第三层
            Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            ReLU(),
            MaxPool1d(kernel_size=4, stride=4),


            # 线性层
            Flatten(),
            Linear(64 * ((image_size * image_size) // (4 ** 3)), 64),
            ReLU(),
            Linear(64, self.num_classes)

        )

        self.wangendtoend = Sequential(
            
             # 第一层
            Conv1d(1, 32, kernel_size=25, stride=1, padding=12),
            ReLU(),
            MaxPool1d(kernel_size=3, stride=3),

            # 第二层
            Conv1d(32, 64, kernel_size=25, stride=1, padding=12),
            ReLU(),
            MaxPool1d(kernel_size=3, stride=3),


            # 线性层
            Flatten(),
            Linear(64 * ((image_size * image_size) // (3 ** 2)), 1024),
            ReLU(),
            Linear(1024, self.num_classes)

        )

    def forward(self, x):
        x = self.wangendtoend(x)
        return x
    
    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        return self(inputs).detach().cpu().numpy().argmax(axis=1)


if __name__ == '__main__':

    print(torch.__version__)
    conv1d_model = Conv1dModel(10, clip_len=1, image_size=32)
    summary(conv1d_model, (1, 1024), device='cpu')
    X = torch.ones((64, 1, 1024))
    print(X.shape)

    y = conv1d_model(X)
    print(y.shape)
    
