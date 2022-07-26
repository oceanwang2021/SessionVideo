import my_dataset_allinone
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import os


class ShowSessionVideo():

    def __init__(self, save_path, label_path) -> None:
        self.save_path = save_path
        label_data = pd.read_csv(label_path, sep=' ', header=None, names=['class_num', 'class_name'])
        self.num2name = dict(zip(label_data['class_num'], label_data['class_name']))

        print(f'label字典：\n {self.num2name}')
        print(self.num2name.values())
        print(type(np.array(self.num2name.values())))

        exit(1000)
        # 创建存储sessionvideo的文件夹
        for class_num, class_name in self.num2name.items():
            class_path = os.path.join(self.save_path, class_name)
            if not os.path.exists(class_path):
                os.mkdir(class_path)

        # 记录每一个类别的存储个数
        self.save_count = [0 for _ in range(len(self.num2name))]
        

    def save_sessionvideo(self, sessionvideo, class_num, clip_len=8, image_size=32):
        # 将Tensor数据转换
        sessionvideo = np.array(sessionvideo)
        class_num = int(class_num.item()) 

        self.save_count[class_num] += 1
        class_name = self.num2name[class_num]
        class_path = os.path.join(self.save_path, class_name)

        # TODO: 将SessionVideo转换为 长图片
        
        buffer = np.empty((image_size, image_size * clip_len))
        for i, image in enumerate(sessionvideo):
            buffer[0: image_size, i * image_size: (i+1) * image_size] = image
        im = Image.fromarray(buffer)
        im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’

        # 按照类别保存长图片
        im.save(os.path.join(class_path, f'{class_name}_{self.save_count[class_num]:03d}.png'))
    

    def show_count(self):
        for class_num, class_name in self.num2name.items():
            print(f'{class_num} {class_name:<20}\t 保存 {self.save_count[class_num]} 张' )

if __name__ == '__main__':
    data_path = r"/home/ocean/dataset/BUAA-CST2022/AppZeroPort_dataset"
    test_data = my_dataset_allinone.SessionVideoDataset(dataset_path=data_path, mode='test', model_type='conv3d', clip_len=8, image_size=32, max_len=100)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

    save_path = r'save_sessionvideo'
    label_path = r'BUAA-CST2022_AppZeroPort_label.txt'
    ssv = ShowSessionVideo(save_path, label_path)

    for i, sample in enumerate(test_dataloader):
        inputs = sample[0]
        labels = sample[1]
        # print(inputs[0][0].shape)
        # print(inputs.size())
        # print(labels)
        
        for input, label in zip(inputs, labels):
            video = input[0]
            # print(video.shape)
            # print(type(label))
            ssv.save_sessionvideo(video, label)
            # print(f'已保存类别：{label}')
            
    ssv.show_count()