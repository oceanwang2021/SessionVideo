import os

import pandas as pd
import numpy as np


class SplitSession():

    def __init__(self, csv_path, save_path, cur_len, max_len, clip_len=8):
        self.csv_path = csv_path
        self.save_path = save_path
        self.clip_len = clip_len
        self.cur_len = cur_len
        self.max_len = max_len

    def run(self):
        pkts = pd.read_csv(self.csv_path)
        session_len = len(pkts)

        path, csv_name = os.path.split(self.csv_path)
        csv_name = csv_name.split('.')[0]
        for i in range(self.clip_len, session_len + 1, self.clip_len):
            clip_name = f'{csv_name}_{self.cur_len:05d}.csv'
            clip_path = os.path.join(self.save_path, clip_name)

            clip = pkts.iloc[i-self.clip_len: i]
            clip.to_csv(clip_path, index=0)

            self.cur_len += 1
            if self.cur_len >= self.max_len:
                print(f'数据集已经达到最大长度 {self.max_len}，停止拆分！')
                break

        return self.cur_len


class SplitSessionWorker():

    def __init__(self, data_src, data_dst, max_len):
        self.data_src = data_src
        self.data_dst = data_dst
        self.class_name_list = os.listdir(data_src)
        self.max_len = max_len

    def run(self):
        print(self.class_name_list)

        for class_name in self.class_name_list:
            class_path = os.path.join(self.data_src, class_name)
            file_name_list = os.listdir(class_path)

            save_class_path = os.path.join(self.data_dst, class_name)

            if not os.path.exists(save_class_path):
                os.mkdir(save_class_path)

            file_count = 0
            cur_len = 0

            for file_name in file_name_list:
                file_path = os.path.join(class_path, file_name)

                sp = SplitSession(file_path, save_class_path, cur_len, self.max_len)
                file_count += 1
                cur_len = sp.run()
                # if clip_count != 0:
                #     print(f'->{file_name} 拆分成 {clip_count} 个SessionVideo')
                if cur_len >= self.max_len:
                    break

            print('=' * 150)
            print(f'{class_name} 拆分了 {file_count} 个文件，共生成了 {cur_len} 个SessionVideo')
            print('=' * 150)


if __name__ == '__main__':
    root_path = r"E:\datasets\data\SessionVideo"
    # 生成ISCXVPN2016数据集application部分
    max_len = 1000
    data_src = os.path.join(root_path, 'ISCXVPN2016', 'application')
    data_dst = os.path.join(root_path, 'ISCXVPN2016', f'application_clip_8_{max_len}')
    if not os.path.exists(data_dst):
        os.mkdir(data_dst)
    spw = SplitSessionWorker(data_src, data_dst, max_len=max_len)
    spw.run()






