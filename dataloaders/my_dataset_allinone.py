import os.path
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class SessionVideoDataset(Dataset):

    """ 参数设置
        dataset:    数据集存放路径
        mode:       train / test
        model_type: conv3d / conv2d / conv1d etc
        clip_len:   conv3d 大于 1, conv2d, conv1d 会设置成 1
        image_size: 图片的尺寸
    """

    def __init__(self, dataset_path, mode, model_type, clip_len, image_size):
        self.dataset_path = dataset_path
        self.mode = mode
        self.model_type = model_type
        if model_type == 'conv3d': 
            self.clip_len = clip_len
        else:
            # 其他模型 默认切片为1
            self.clip_len = 1
        self.image_size = image_size


        self.mode_path = os.path.join(dataset_path, mode)

        self.session_videos, labels = [], []

        label_real_len_dict = {}
        label_load_len_dict = {}

        time_0 = time.time()
        read_len_time = 0
        load_time = 0

        for label in sorted(os.listdir(self.mode_path)):
            # todo： 读取数据长度，占用时间过多，尝试去除
            time_1 = time.time()
            # TODO: 真实长度读取

            time_2 = time.time()

            read_len_time += time_2 - time_1


            for i, session in enumerate(sorted(os.listdir(os.path.join(self.mode_path, label)))):
                session_path = os.path.join(self.mode_path, label, session)
                # todo: 计算每一个会话 可拆分多少个 SessionVideo
                pkts = pd.read_csv(session_path)
                # pkts.info(memory_usage='deep')
                
                session_len = len(pkts)

                # TODO: 三维的 SessionVideo
                if self.model_type == 'conv3d':
                    for pkt_i in range(self.clip_len, session_len + 1, self.clip_len):
                        buffer = np.empty((self.clip_len, 1, self.image_size, self.image_size)).astype(dtype=np.uint8)
                        # TODO: 设置取出设定的 帧数 和 图片大小
                        session_video = pkts.iloc[pkt_i-self.clip_len: pkt_i, 0: self.image_size * self.image_size]
                        for image_i in range(self.clip_len):
                            pkt = np.array(session_video.iloc[image_i]).reshape((1, self.image_size, self.image_size)).astype(dtype=np.uint8)
                            buffer[image_i] = pkt

                        # todo： 正式加载数据
                        # self.session_videos.append(buffer)
                        self.session_videos.append(buffer)
                        labels.append(label)

                # TODO: 二维图片
                elif self.model_type == 'conv2d':
                    for pkt_i in range(session_len):
                        # buffer = np.empty((1, self.image_size, self.image_size)).astype(dtype=np.uint8)
                        session_image2d = np.array(pkts.iloc[pkt_i, 0: self.image_size * self.image_size]).reshape((1, self.image_size, self.image_size)).astype(dtype=np.uint8)
                        # todo： 正式加载数据
                        self.session_videos.append(session_image2d)
                        labels.append(label)

                # TODO: 一维图片
                elif self.model_type == 'conv1d':
                    for pkt_i in range(session_len):
                        # buffer = np.empty((1, self.image_size, self.image_size)).astype(dtype=np.uint8)
                        session_image1d = np.array(pkts.iloc[pkt_i, 0: self.image_size * self.image_size]).reshape((1, self.image_size * self.image_size)).astype(dtype=np.uint8)
                        self.session_videos.append(session_image1d)
                        labels.append(label)

            time_3 = time.time()
            load_time += time_3 - time_2

        time_4 = time.time()
        all_time = time_4-time_0
        for k, v in label_load_len_dict.items():
            print(f'{k: <20} : {v}')
        
        assert len(labels) == len(self.session_videos)
        print(f'{mode}数据集共有：{len(self.session_videos)} 个SessionVideo')
        print(f'{self.mode}数据集生成总时间：{round(all_time, 2)} \t 读取长度用时：{round(read_len_time, 2)} \t 加载数据用时：{round(load_time, 2)}')

        # todo: 将标签转换为整数
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # todo: 将 标签名列表 转换为 标签索引数组
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        path, file2 = os.path.split(self.dataset_path )
        _, file1 = os.path.split(path)
        with open(f'./{file1}_{file2.split("_")[0]}_label.txt', 'w') as f:
            for id, label in enumerate(sorted(self.label2index)):
                f.writelines(f'{id} {label}\n')

    def __len__(self):
        return len(self.session_videos)

    def __getitem__(self, index):

        session_video = self.session_videos[index]
        # TODO: 对于 SessionVideo 才需要变换
        if self.model_type == 'conv3d':
            session_video = session_video.transpose((1, 0, 2, 3))
        label = np.array(self.label_array[index])
        # normalize 先不做
        return torch.from_numpy(session_video).to(torch.float32), torch.from_numpy(label).to(torch.long)


if __name__ == '__main__':
    data_path = r"/home/ocean/dataset/"
    test_data = SessionVideoDataset(dataset_path=data_path, mode='test', model_type='conv3d', clip_len=8, image_size=32)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

    for i, sample in enumerate(test_dataloader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs[0][0])
        print(inputs.size())
        print(labels)

        if i % 1 == 0:
            break
