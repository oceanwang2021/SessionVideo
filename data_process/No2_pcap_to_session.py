import csv
import gc
import os.path

from scapy.all import *

class PcapToSession():
    """
    将清洗后的pcap文件转换成Session，通过scapy的get_session来实现，一个session以一个csv文件来保存
    参数：
        pcap_dir: pcap文件夹
        save_path: 用来保存获取的session的文件件路径
        label: 
        pkt_len: 每个包截取的长度
    """

    def __init__(self, pcap_path: str, save_path: str, zero_port, pkt_len=(32 * 32)) -> None:
        self.pcap_path = pcap_path
        self.save_path = save_path
        self.zero_port = zero_port
        self.pkt_len = pkt_len

    def run(self):
        pkts = rdpcap(self.pcap_path)
        sessions = pkts.sessions()

        path, pcap_name = os.path.split(self.pcap_path)
        pcap_name = pcap_name.split(".")[0]
        
        del pkts
        gc.collect()

        ip_count, ipv6_count = 0, 0
        for k, v in sessions.items():

            session_name = f'{pcap_name}_{ip_count:06d}_{k.split(" ")[0]}.csv'
            session_path = os.path.join(self.save_path, session_name)
            images = []
            for pkt in v:
                if pkt.getlayer('IP'):
                    image = self.pkt_to_image(pkt, self.pkt_len)
                    images.append(image)
                else:
                    continue
            if len(images) > 0:
                ip_count += 1
                with open(session_path, 'w', newline='') as f:
                    # todo: 设置表头
                    writer = csv.writer(f)
                    indexes = []
                    for i in range(self.pkt_len):
                        indexes.append(f'p{i}')
                    writer.writerow(indexes)
                    writer.writerows(images)
            else:
                ipv6_count += 1
        print(f'-> {pcap_name} 共有 {len(sessions)} 个会话 \t 保存 {ip_count}  \t 丢弃 {ipv6_count}')

    def pkt_to_image(self, pkt, pkt_len) -> list:
        pixels = [0 for _ in range(pkt_len)]  # 1024 + 1label

        # TODO: IP 地址置零
        ip_layer = pkt.getlayer('IP')

        ip_layer.src = '0'
        ip_layer.dst = '0'

        # TODO: 端口号 置零
        if self.zero_port == True:
            if ip_layer.haslayer('TCP') or ip_layer.haslayer('UDP'):
                ip_layer.sport = 0
                ip_layer.dport = 0

        hex_pkt = hexstr(ip_layer, onlyhex=1).split(' ')
        oct_pkt = self.hex2oct(hex_pkt)

        if len(oct_pkt) >= pkt_len:
            pixels[0: pkt_len] = oct_pkt[0: pkt_len]
        else:
            pixels[: len(oct_pkt)] = oct_pkt[: len(oct_pkt)]
        return pixels

    def hex2oct(self, hex_pkt: list) -> list:
        oct_pkt = [0] * len(hex_pkt)
        for i in range(len(hex_pkt)):
            oct_pkt[i] = int(hex_pkt[i], 16)
        return oct_pkt


class PcapToSessionWorker():
    def __init__(self, data_src, data_dst, zero_port, labels):
        self.data_src = data_src
        self.data_dst = data_dst
        self.zero_port = zero_port
        self.labels = labels

    def work(self):
        file_name_list = os.listdir(self.data_src)

        dir_read_start_time = time.time()
        file_count = 0

        for file in file_name_list:
            file_count += 1
            label = self.labels[file.split('.')[0]]
            file_path = os.path.join(self.data_src, file)
            class_path = os.path.join(self.data_dst, label)

            if not os.path.exists(class_path):
                os.mkdir(class_path)

            start_time = time.time()
            pts = PcapToSession(file_path, class_path, self.zero_port)
            pts.run()
            end_time = time.time()

            print(f' {file_count:3} - {file:30} \t 转换用时：{round((end_time - start_time), 2):10} \t 当前时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':

    
    buaa_app_labels ={
        'QQ': 'QQ',
        'WeChat': 'WeChat',

        'Outlook': 'Outlook',
        'Skype': 'Skype',
        'VoovMeeting': 'VoovMeeting',

        'QQMusic': 'QQMusic',
        'NetEaseCloudMusic': 'NetEaseCloudMusic',

        'TencentVideo': 'TencentVideo',
        'YouKu': 'YouKu',
        'iQIYI': 'iQIYI',

        'DoubanWithChrome': 'DoubanWithChrome',
        'WeiboWithChrome': 'WeiboWithChrome',
        'AmazonWithChrome': 'AmazonWithChrome',
        'TaobaoWithChrome': 'TaobaoWithChrome',

        'Minecraft': 'Minecraft',
        'LeagueOfLegends': 'LeagueOfLegends',
        'CrossFire': 'CrossFire',
        'BaiduNetdisk': 'BaiduNetdisk',

        'Thunder': 'Thunder',
        'BitComet': 'BitComet',

    }

    root_path = r""
    data_path = os.path.join(root_path, 'DeepClear')
    image_path = os.path.join(root_path, 'SessionVideo')

    data_src = os.path.join(data_path, 'BUAA-CST20212')
    data_dst = os.path.join(image_path, 'BUAA-CST20212', 'AppZeroPort')

    ptsw = PcapToSessionWorker(data_src, data_dst, zero_port=True, labels=buaa_app_labels)
    ptsw.work()







