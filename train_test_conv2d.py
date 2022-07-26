from email.mime import image
import os
import time
from datetime import datetime

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, classification_report
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloaders import my_dataset_new1, my_dataset_allinone
from network import conv3d_model
import numpy as np
import pandas as pd
from torchsummary import summary

from network.conv2d_model import Conv2dModel


print('=========================== 参数设置 =========================')
# todo: 数据集参数
data_path = r"/home/ocean/dataset/BUAA-CST2022/AppZeroPort_dataset"
label_path = r'BUAA-CST2022_AppZeroPort_label.txt'
model_type = 'conv2d'
num_classes = 20
clip_len = 1
image_size = 32
batch_size = 64

# todo： 训练参数
epochs = 30
lr_index = 5
learning_rate = 1 * 10 ** (-lr_index)

# todo：创建tensorboard日志
_, file = os.path.split(data_path)
log_dir = os.path.join('.', 'log', f'BUAA_{model_type}_{file}_{epochs}_{batch_size}_{lr_index}--{clip_len}_{image_size}')
if os.path.exists(log_dir):
    log_dir = f'{log_dir}_新'
log_writer = SummaryWriter(log_dir)


print(f'epochs = {epochs}')
print(f'learning_rate = {learning_rate}')
print(f'日志地址：{log_dir}')
print(f'开始时间：{datetime.now()}')

print('==============================================================')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('设备：{}'.format(device))
my_model = Conv2dModel(num_classes=num_classes, clip_len=clip_len, image_size=image_size)
my_model = my_model.to(device)
summary(my_model, (1, image_size, image_size), batch_size=batch_size, device='cuda')

# 损失函数
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.to(device)

# 优化器
# optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate) # , momentum=0.9, weight_decay=5e-4
optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

# todo：数据加载
print(f'开始加载数据......')
data_start_time = time.time()
train_data = my_dataset_allinone.SessionVideoDataset(data_path, mode='train', model_type=model_type, clip_len=clip_len, image_size=image_size)
test_data = my_dataset_allinone.SessionVideoDataset(data_path, mode='test', model_type=model_type, clip_len=clip_len, image_size=image_size)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)
data_end_time = time.time()
print('数据加载用时：{}'.format(data_end_time-data_start_time))

evaluation = {
    "acc": 0,
    "p": 0,
    "f1": 0,
    "r": 0,
}


# todo：开始训练
for epoch in range(epochs):
    epoch_start_time = time.time()

    total_train_step = 0
    total_train_loss = 0.0
    my_model.train()
    for X, y in train_dataloader:       # tqdm(train_dataloader, desc=f"Train[{epoch}/{epochs}]"):
        # print(X.shape)
        X, y = X.to(device), y.to(device)
        # 损失函数已经包含Softmax，因此不使用
        outputs = my_model(X)
        loss = loss_function(outputs, y)
        total_train_loss += loss.item()
        # 梯度清零并反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 100次打印一次
        total_train_step = total_train_step + 1
        if total_train_step % 1000 == 0:
            print(f"训练batch数: {total_train_step:7}  Loss: {round(loss.item(), 10):12} \t {datetime.now()}")

    log_writer.add_scalar(tag='训练损失值', scalar_value=total_train_loss, global_step=epoch)

    # todo: 验证
    my_model.eval()
    predicted = []
    targets = []
    total_test_loss = 0.0
    total_test_step = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            targets.append(y)

            X, y = X.to(device), y.to(device)

            outputs = my_model(X)
            loss = loss_function(outputs, y)
            total_test_loss += loss.item()

            predicted.append(my_model.predict(X))
            total_test_step += 1

        log_writer.add_scalar(tag='测试损失值', scalar_value=total_test_loss, global_step=epoch)
        predicted = np.concatenate(predicted, axis=0).reshape((-1, ))
        targets = np.concatenate(targets, axis=0).reshape((-1, ))

        # 使用验证集绘制评估数值的图像和保存模型
        acc = accuracy_score(targets, predicted)
        p = precision_score(targets, predicted, average='weighted')
        f1 = f1_score(targets, predicted, average='weighted')
        r = recall_score(targets, predicted, average='weighted')

        # TODO: 读取标签名称
        label_data = pd.read_csv(label_path, sep=' ', header=None, names=['class_num', 'class_name'])
        num2name = dict(zip(label_data['class_num'], label_data['class_name']))
        cr = classification_report(targets, predicted, labels=list(num2name.keys()), target_names=list(num2name.values()))
        cm = confusion_matrix(targets, predicted, labels=list(num2name.keys()))

        log_writer.add_scalar(tag="Accuracy", scalar_value=acc, global_step=epoch)
        log_writer.add_scalar(tag="Precision", scalar_value=p, global_step=epoch)
        log_writer.add_scalar(tag="F1", scalar_value=f1, global_step=epoch)
        log_writer.add_scalar(tag="Recall", scalar_value=r, global_step=epoch)

        # TODO: 模型保存
        if f1 > evaluation['f1']:
            evaluation['f1'] = f1
            torch.save(my_model, f'{model_type}_f1_model.pth')
            print(f'第 {epoch + 1} 轮 模型训练需要保存')

    epoch_end_time = time.time()
    print(f'第 {epoch + 1} 轮epoch时间: {epoch_end_time - epoch_start_time}')

    print(f'训练损失值：{total_train_loss}')
    print(f'测试损失值：{total_test_loss}')
    print(f'Accuracy: {acc} \t Precision: {p} \t F1: {f1} \t Recall: {r}')
    print(f'混淆矩阵: \n {cm}')

log_writer.close()
print('=' * 150)
print(f'结束时间：{datetime.now()}')
print('=' * 150)