
import time

from scapy.all import *


class PcapDeepClear(object):
    """pcap文件深层清理类"""
    def __init__(self, src: str, dst: str) -> None:
        """
        :param src: 要清洗的pcap文件
        :param dst: 清洗后保存的pcap文件
        :param log: 日志记录对象
        """
        self.src = src
        self.dst = dst

    def run(self) -> None:
        """开始执行清理"""
        packets = 0
        no_ip_tcp_udp = 0
        dns_others = 0
        tcp_SAF = 0
        saved = 0
        pr = PcapReader(self.src)
        pkt = pr.read_packet()
        pw = PcapWriter(self.dst)
        while pkt:
            packets += 1
            # TODO: 去除没有 IP 或 tcp 或 udp 的数据包
            if not (pkt.haslayer('IP') or pkt.haslayer('IPv6') or pkt.haslayer('TCP') or pkt.haslayer('UDP')):
                no_ip_tcp_udp += 1
                pkt = pr.next()
                continue

            # TODO: 去除DNS NBNS LLMNR 等寻址相关协议
            if pkt.haslayer('DNS') or pkt.haslayer('NBNS query request') or pkt.haslayer('Link Local Multicast Node Resolution - Query'):
                dns_others += 1
                pkt = pr.next()
                continue
            # TODO: 去除 TCP 建立连接过程的包
            if pkt.haslayer('TCP') and (pkt.getlayer('TCP').flags == "S"
                                        or pkt.getlayer('TCP').flags == "SA"
                                        or pkt.getlayer('TCP').flags == "F"
                                        or pkt.getlayer('TCP').flags == "FA"):
                tcp_SAF += 1
                pkt = pr.next()
                continue
            # TODO: 深度清洗完毕完毕，保存
            saved += 1
            pw.write(pkt)

            pkt = pr.next()
        # 写入日志信息
        pr.close()
        pw.close()
        print(f'-- 原文件共有 {packets: 6} 个包, 新文件保存 {saved: 6} 个包 ---- 未保存的包中 IP层传输层缺失: {no_ip_tcp_udp:3} \t DNS寻址相关： {dns_others: 4} \t TCP连接建立过程： {tcp_SAF: 4} ')


class PcapDeepCleaner():

    def __init__(self, path: str) -> None:

        self.data_path = path
        self.file_name_list = os.listdir(path)

    def run(self, save_path: str) -> None:
        index = 0

        dir_read_start_time = time.time()
        for file_name in self.file_name_list:
            file_read_start_time = time.time()
            index += 1

            print(f"->{index} - {file_name}")
            pdc = PcapDeepClear(src=os.path.join(self.data_path, file_name), dst=os.path.join(save_path, file_name))
            pdc.run()

            file_read_end_time = time.time()
            print('--读取 {} 文件用时：{}'.format(file_name, file_read_end_time-file_read_start_time))
        dir_read_end_time = time.time()
        print('=========================================================')
        print('{} 全部文件读取完毕，用时：{}'.format(self.data_path, dir_read_end_time-dir_read_start_time))


if __name__ == '__main__':
    # 开始清洗
    datasets = [
        'BUAA-CST2022',
    ]

    root_path = r""
    clear_data_path = os.path.join(root_path, 'Original')
    deep_clear_save_path = os.path.join(root_path, 'DeepClear')

    for dataset in datasets:
        data_src = os.path.join(clear_data_path, dataset)
        data_dst = os.path.join(deep_clear_save_path, dataset)
        scan = PcapDeepCleaner(path=data_src)
        scan.run(save_path=data_dst)


