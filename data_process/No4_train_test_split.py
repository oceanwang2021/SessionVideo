
import os.path
import shutil
import time
from datetime import datetime

from sklearn.model_selection import train_test_split


class TrainTestSplit():

    def __init__(self, input_path):
        self.input_path = input_path
        root_path, input_dir = os.path.split(input_path)
        output_dir = f'{input_dir}_dataset'
        self.output_path = os.path.join(root_path, output_dir)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            os.mkdir(os.path.join(self.output_path, 'train'))
            os.mkdir(os.path.join(self.output_path, 'test'))

    def run(self):
        class_name_list = os.listdir(self.input_path)

        for class_name in class_name_list:
            start_time = time.time()

            class_path = os.path.join(self.input_path, class_name)
            session_videos = [name for name in os.listdir(class_path)]

            train, test = train_test_split(session_videos, test_size=0.2, random_state=2022)

            train_class_path = os.path.join(self.output_path, 'train', f'{class_name}.txt')
            test_class_path = os.path.join(self.output_path, 'test', f'{class_name}.txt')

            train = [f'{os.path.join(class_path, session_video)}\n' for session_video in train]
            test = [f'{os.path.join(class_path, session_video)}\n' for session_video in test]

            with open(train_class_path, 'w', encoding='utf-8') as f:
                f.writelines(train)

            with open(test_class_path, 'w', encoding='utf-8') as f:
                f.writelines(test)

            """
            **复制文件的方式，太慢了，改成直接存储 文件地址**
            train_class_path = os.path.join(self.output_path, 'train', class_name)
            test_class_path = os.path.join(self.output_path, 'test', class_name)

            if not os.path.exists(train_class_path):
                os.mkdir(train_class_path)
            if not os.path.exists(test_class_path):
                os.mkdir(test_class_path)

            for session_video in train:
                shutil.copyfile(os.path.join(class_path, session_video), os.path.join(train_class_path, session_video))

            for session_video in test:
                shutil.copyfile(os.path.join(class_path, session_video), os.path.join(test_class_path, session_video))
            """

            end_time = time.time()
            print(f'->{class_name} 拆分完毕！ 用时：{round((end_time - start_time), 2):6} \t 当前时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 150)
        print(f'-->> {self.input_path} 全部拆分完毕！')


if __name__ == '__main__':
    input_path = r"E:\datasets\data\SessionVideo\ISCXVPN2016\application_clip_8_1000"
    tts = TrainTestSplit(input_path)
    tts.run()





