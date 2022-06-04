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

    vpn_app_labels = {
        # 0 AIM chat
        'nonvpn-aim_chat_3a': 'AIM chat',
        'nonvpn-aim_chat_3b': 'AIM chat',
        'nonvpn-AIMchat1': 'AIM chat',
        'nonvpn-AIMchat2': 'AIM chat',
        'vpn-aim_chat1a': 'AIM chat',
        'vpn-aim_chat1b': 'AIM chat',

        #  Email
        'nonvpn-email1a': 'Email',
        'nonvpn-email1b': 'Email',
        'nonvpn-email2a': 'Email',
        'nonvpn-email2b': 'Email',
        'vpn-email2a': 'Email',
        'vpn-email2b': 'Email',

        #  Facebook
        'nonvpn-facebook_audio1a': 'Facebook',
        'nonvpn-facebook_audio1b': 'Facebook',
        'nonvpn-facebook_audio2a': 'Facebook',
        'nonvpn-facebook_audio2b': 'Facebook',
        'nonvpn-facebook_audio3': 'Facebook',
        'nonvpn-facebook_audio4': 'Facebook',
        'nonvpn-facebook_chat_4a': 'Facebook',
        'nonvpn-facebook_chat_4b': 'Facebook',
        'nonvpn-facebook_video1a': 'Facebook',
        'nonvpn-facebook_video1b': 'Facebook',
        'nonvpn-facebook_video2a': 'Facebook',
        'nonvpn-facebook_video2b': 'Facebook',
        'nonvpn-facebookchat1': 'Facebook',
        'nonvpn-facebookchat2': 'Facebook',
        'nonvpn-facebookchat3': 'Facebook',
        'vpn-facebook_audio2': 'Facebook',
        'vpn-facebook_chat1a': 'Facebook',
        'vpn-facebook_chat1b': 'Facebook',

        #  FTPS 两个大文件被拆分了 'nonvpn-ftps_down_1a'和'nonvpn-ftps_down_1b': 'FTPS'被分割
        'nonvpn-ftps_down_1a-1': 'FTPS',
        'nonvpn-ftps_down_1a-2': 'FTPS',
        'nonvpn-ftps_down_1a-3': 'FTPS',
        'nonvpn-ftps_down_1a-4': 'FTPS',
        'nonvpn-ftps_down_1a-5': 'FTPS',
        'nonvpn-ftps_down_1a-6': 'FTPS',
        'nonvpn-ftps_down_1b-1': 'FTPS',
        'nonvpn-ftps_down_1b-2': 'FTPS',
        'nonvpn-ftps_down_1b-3': 'FTPS',
        'nonvpn-ftps_down_1b-4': 'FTPS',
        'nonvpn-ftps_up_2a': 'FTPS',
        'nonvpn-ftps_up_2b': 'FTPS',
        'vpn-ftps_A': 'FTPS',
        'vpn-ftps_B': 'FTPS',

        # Gmail
        'nonvpn-gmailchat1': 'Gmail',
        'nonvpn-gmailchat2': 'Gmail',
        'nonvpn-gmailchat3': 'Gmail',

        # Hangouts
        'nonvpn-hangout_chat_4b': 'Hangouts',
        'nonvpn-hangouts_audio1a': 'Hangouts',
        'nonvpn-hangouts_audio1b': 'Hangouts',
        'nonvpn-hangouts_audio2a': 'Hangouts',
        'nonvpn-hangouts_audio2b': 'Hangouts',
        'nonvpn-hangouts_audio3': 'Hangouts',
        'nonvpn-hangouts_audio4': 'Hangouts',
        'nonvpn-hangouts_chat_4a': 'Hangouts',
        'nonvpn-hangouts_video1b': 'Hangouts',
        'nonvpn-hangouts_video2a': 'Hangouts',
        'nonvpn-hangouts_video2b': 'Hangouts',
        'vpn-hangouts_audio1': 'Hangouts',
        'vpn-hangouts_audio2': 'Hangouts',
        'vpn-hangouts_chat1a': 'Hangouts',
        'vpn-hangouts_chat1b': 'Hangouts',

        # ICQ
        'nonvpn-icq_chat_3a': 'ICQ',
        'nonvpn-icq_chat_3b': 'ICQ',
        'nonvpn-ICQchat1': 'ICQ',
        'nonvpn-ICQchat2': 'ICQ',
        'vpn-icq_chat1a': 'ICQ',
        'vpn-icq_chat1b': 'ICQ',

        # Netflix
        'nonvpn-netflix1': 'Netflix',
        'nonvpn-netflix2': 'Netflix',
        'nonvpn-netflix3': 'Netflix',
        'nonvpn-netflix4': 'Netflix',
        'vpn-netflix_A': 'Netflix',

        # SCP 'nonvpn-scp1': 'SCP'  拆分为3个文件
        'nonvpn-scp1-1': 'SCP',
        'nonvpn-scp1-2': 'SCP',
        'nonvpn-scp1-3': 'SCP',
        'nonvpn-scpDown1': 'SCP',
        'nonvpn-scpDown2': 'SCP',
        'nonvpn-scpDown3': 'SCP',
        'nonvpn-scpDown4': 'SCP',
        'nonvpn-scpDown5': 'SCP',
        'nonvpn-scpDown6': 'SCP',
        'nonvpn-scpUp1': 'SCP',
        'nonvpn-scpUp2': 'SCP',
        'nonvpn-scpUp3': 'SCP',
        'nonvpn-scpUp5': 'SCP',
        'nonvpn-scpUp6': 'SCP',

        # SFTP
        'nonvpn-sftp_down_3a': 'SFTP',
        'nonvpn-sftp_down_3b': 'SFTP',
        'nonvpn-sftp_up_2a': 'SFTP',
        'nonvpn-sftp_up_2b': 'SFTP',
        'nonvpn-sftp1': 'SFTP',
        'nonvpn-sftpDown1': 'SFTP',
        'nonvpn-sftpDown2': 'SFTP',
        'nonvpn-sftpUp1': 'SFTP',
        'vpn-sftp_A': 'SFTP',
        'vpn-sftp_B': 'SFTP',

        # Skype
        'nonvpn-skype_audio1a': 'Skype',
        'nonvpn-skype_audio1b': 'Skype',
        'nonvpn-skype_audio2a': 'Skype',
        'nonvpn-skype_audio2b': 'Skype',
        'nonvpn-skype_audio3': 'Skype',
        'nonvpn-skype_audio4': 'Skype',
        'nonvpn-skype_chat1a': 'Skype',
        'nonvpn-skype_chat1b': 'Skype',
        'nonvpn-skype_file1': 'Skype',
        'nonvpn-skype_file2': 'Skype',
        'nonvpn-skype_file3': 'Skype',
        'nonvpn-skype_file4': 'Skype',
        'nonvpn-skype_file5': 'Skype',
        'nonvpn-skype_file6': 'Skype',
        'nonvpn-skype_file7': 'Skype',
        'nonvpn-skype_file8': 'Skype',
        'nonvpn-skype_video1a': 'Skype',
        'nonvpn-skype_video1b': 'Skype',
        'nonvpn-skype_video2a': 'Skype',
        'nonvpn-skype_video2b': 'Skype',
        'vpn-skype_audio1': 'Skype',
        'vpn-skype_audio2': 'Skype',
        'vpn-skype_chat1a': 'Skype',
        'vpn-skype_chat1b': 'Skype',
        'vpn-skype_files1a': 'Skype',
        'vpn-skype_files1b': 'Skype',

        # Spotify
        'nonvpn-spotify1': 'Spotify',
        'nonvpn-spotify2': 'Spotify',
        'nonvpn-spotify3': 'Spotify',
        'nonvpn-spotify4': 'Spotify',
        'vpn-spotify_A': 'Spotify',

        # Torrent
        'vpn-bittorrent': 'Torrent',

        # Voipbuster
        'nonvpn-voipbuster_4a': 'Voipbuster',
        'nonvpn-voipbuster_4b': 'Voipbuster',
        'nonvpn-voipbuster1b': 'Voipbuster',
        'nonvpn-voipbuster2b': 'Voipbuster',
        'nonvpn-voipbuster3b': 'Voipbuster',
        'vpn-voipbuster1a': 'Voipbuster',
        'vpn-voipbuster1b': 'Voipbuster',

        # Vimeo
        'nonvpn-vimeo1': 'Vimeo',
        'nonvpn-vimeo2': 'Vimeo',
        'nonvpn-vimeo3': 'Vimeo',
        'nonvpn-vimeo4': 'Vimeo',
        'vpn-vimeo_A': 'Vimeo',
        'vpn-vimeo_B': 'Vimeo',

        # YouTube
        'nonvpn-youtube1': 'YouTube',
        'nonvpn-youtube2': 'YouTube',
        'nonvpn-youtube3': 'YouTube',
        'nonvpn-youtube4': 'YouTube',
        'nonvpn-youtube5': 'YouTube',
        'nonvpn-youtube6': 'YouTube',
        'nonvpn-youtubeHTML5_1': 'YouTube',
        'vpn-youtube_A': 'YouTube',
    }
    vpn_tra_labels = {
        # 1 Chat
        'nonvpn-aim_chat_3a': 'Chat',
        'nonvpn-aim_chat_3b': 'Chat',
        'nonvpn-AIMchat1': 'Chat',
        'nonvpn-AIMchat2': 'Chat',
        'nonvpn-icq_chat_3a': 'Chat',
        'nonvpn-icq_chat_3b': 'Chat',
        'nonvpn-ICQchat1': 'Chat',
        'nonvpn-ICQchat2': 'Chat',
        'nonvpn-skype_chat1a': 'Chat',
        'nonvpn-skype_chat1b': 'Chat',
        'nonvpn-facebookchat1': 'Chat',
        'nonvpn-facebookchat2': 'Chat',
        'nonvpn-facebookchat3': 'Chat',
        'nonvpn-facebook_chat_4a': 'Chat',
        'nonvpn-facebook_chat_4b': 'Chat',
        'nonvpn-hangouts_chat_4a': 'Chat',
        'nonvpn-hangout_chat_4b': 'Chat',

        # 2 VPN Chat
        'vpn-aim_chat1a': 'VPN Chat',
        'vpn-aim_chat1b': 'VPN Chat',
        'vpn-icq_chat1a': 'VPN Chat',
        'vpn-icq_chat1b': 'VPN Chat',
        'vpn-skype_chat1a': 'VPN Chat',
        'vpn-skype_chat1b': 'VPN Chat',
        'vpn-facebook_chat1a': 'VPN Chat',
        'vpn-facebook_chat1b': 'VPN Chat',
        'vpn-hangouts_chat1a': 'VPN Chat',
        'vpn-hangouts_chat1b': 'VPN Chat',

        # 3 Email
        'nonvpn-email1a': 'Email',
        'nonvpn-email1b': 'Email',
        'nonvpn-email2a': 'Email',
        'nonvpn-email2b': 'Email',
        'nonvpn-gmailchat1': 'Email',
        'nonvpn-gmailchat2': 'Email',
        'nonvpn-gmailchat3': 'Email',

        # 4 VPN Email
        'vpn-email2a': 'VPN Email',
        'vpn-email2b': 'VPN Email',

        # 5 File Transfer
        'nonvpn-skype_file1': 'File Transfer',
        'nonvpn-skype_file2': 'File Transfer',
        'nonvpn-skype_file3': 'File Transfer',
        'nonvpn-skype_file4': 'File Transfer',
        'nonvpn-skype_file5': 'File Transfer',
        'nonvpn-skype_file6': 'File Transfer',
        'nonvpn-skype_file7': 'File Transfer',
        'nonvpn-skype_file8': 'File Transfer',
        # 'nonvpn-ftps_down_1a': 'File Transfer',
        'nonvpn-ftps_down_1a-1': 'File Transfer',
        'nonvpn-ftps_down_1a-2': 'File Transfer',
        'nonvpn-ftps_down_1a-3': 'File Transfer',
        'nonvpn-ftps_down_1a-4': 'File Transfer',
        'nonvpn-ftps_down_1a-5': 'File Transfer',
        'nonvpn-ftps_down_1a-6': 'File Transfer',
        # 'nonvpn-ftps_down_1b': 'File Transfer',
        'nonvpn-ftps_down_1b-1': 'File Transfer',
        'nonvpn-ftps_down_1b-2': 'File Transfer',
        'nonvpn-ftps_down_1b-3': 'File Transfer',
        'nonvpn-ftps_down_1b-4': 'File Transfer',
        'nonvpn-ftps_up_2a': 'File Transfer',
        'nonvpn-ftps_up_2b': 'File Transfer',
        'nonvpn-sftp_down_3a': 'File Transfer',
        'nonvpn-sftp_down_3b': 'File Transfer',
        'nonvpn-sftp_up_2a': 'File Transfer',
        'nonvpn-sftp_up_2b': 'File Transfer',
        'nonvpn-sftp1': 'File Transfer',
        'nonvpn-sftpDown1': 'File Transfer',
        'nonvpn-sftpDown2': 'File Transfer',
        'nonvpn-sftpUp1': 'File Transfer',
        # 'nonvpn-scp1': 'File Transfer',
        'nonvpn-scp1-1': 'File Transfer',
        'nonvpn-scp1-2': 'File Transfer',
        'nonvpn-scp1-3': 'File Transfer',
        'nonvpn-scpDown1': 'File Transfer',
        'nonvpn-scpDown2': 'File Transfer',
        'nonvpn-scpDown3': 'File Transfer',
        'nonvpn-scpDown4': 'File Transfer',
        'nonvpn-scpDown5': 'File Transfer',
        'nonvpn-scpDown6': 'File Transfer',
        'nonvpn-scpUp1': 'File Transfer',
        'nonvpn-scpUp2': 'File Transfer',
        'nonvpn-scpUp3': 'File Transfer',
        'nonvpn-scpUp5': 'File Transfer',
        'nonvpn-scpUp6': 'File Transfer',

        # 6 VPN File Transfer
        'vpn-skype_files1a': 'VPN File Transfer',
        'vpn-skype_files1b': 'VPN File Transfer',
        'vpn-ftps_A': 'VPN File Transfer',
        'vpn-ftps_B': 'VPN File Transfer',
        'vpn-sftp_A': 'VPN File Transfer',
        'vpn-sftp_B': 'VPN File Transfer',

        # 7 Video
        'nonvpn-vimeo1': 'Video',
        'nonvpn-vimeo2': 'Video',
        'nonvpn-vimeo3': 'Video',
        'nonvpn-vimeo4': 'Video',
        'nonvpn-youtube1': 'Video',
        'nonvpn-youtube2': 'Video',
        'nonvpn-youtube3': 'Video',
        'nonvpn-youtube4': 'Video',
        'nonvpn-youtube5': 'Video',
        'nonvpn-youtube6': 'Video',
        'nonvpn-youtubeHTML5_1': 'Video',
        'nonvpn-netflix1': 'Video',
        'nonvpn-netflix2': 'Video',
        'nonvpn-netflix3': 'Video',
        'nonvpn-netflix4': 'Video',
        'nonvpn-facebook_video1a': 'Video',
        'nonvpn-facebook_video1b': 'Video',
        'nonvpn-facebook_video2a': 'Video',
        'nonvpn-facebook_video2b': 'Video',
        'nonvpn-hangouts_video1b': 'Video',
        'nonvpn-hangouts_video2a': 'Video',
        'nonvpn-hangouts_video2b': 'Video',
        'nonvpn-skype_video1a': 'Video',
        'nonvpn-skype_video1b': 'Video',
        'nonvpn-skype_video2a': 'Video',
        'nonvpn-skype_video2b': 'Video',

        # 8 VPN Vidoe
        'vpn-vimeo_A': 'VPN Video',
        'vpn-vimeo_B': 'VPN Video',
        'vpn-youtube_A': 'VPN Video',
        'vpn-netflix_A': 'VPN Video',

        # 9 Audio
        'nonvpn-spotify1': 'Audio',
        'nonvpn-spotify2': 'Audio',
        'nonvpn-spotify3': 'Audio',
        'nonvpn-spotify4': 'Audio',

        # 10 VPN Audio
        'vpn-spotify_A': 'VPN Audio',

        # VPN P2P
        'vpn-bittorrent': 'VPN P2P',

        # 11 VoIP
        'nonvpn-facebook_audio1a': 'VoIP',
        'nonvpn-facebook_audio1b': 'VoIP',
        'nonvpn-facebook_audio2a': 'VoIP',
        'nonvpn-facebook_audio2b': 'VoIP',
        'nonvpn-facebook_audio3': 'VoIP',
        'nonvpn-facebook_audio4': 'VoIP',
        'nonvpn-hangouts_audio1a': 'VoIP',
        'nonvpn-hangouts_audio1b': 'VoIP',
        'nonvpn-hangouts_audio2a': 'VoIP',
        'nonvpn-hangouts_audio2b': 'VoIP',
        'nonvpn-hangouts_audio3': 'VoIP',
        'nonvpn-hangouts_audio4': 'VoIP',
        'nonvpn-skype_audio1a': 'VoIP',
        'nonvpn-skype_audio1b': 'VoIP',
        'nonvpn-skype_audio2a': 'VoIP',
        'nonvpn-skype_audio2b': 'VoIP',
        'nonvpn-skype_audio3': 'VoIP',
        'nonvpn-skype_audio4': 'VoIP',
        'nonvpn-voipbuster_4a': 'VoIP',
        'nonvpn-voipbuster_4b': 'VoIP',
        'nonvpn-voipbuster1b': 'VoIP',
        'nonvpn-voipbuster2b': 'VoIP',
        'nonvpn-voipbuster3b': 'VoIP',

        # 12 VPN VoIP
        'vpn-facebook_audio2': 'VPN VoIP',
        'vpn-hangouts_audio1': 'VPN VoIP',
        'vpn-hangouts_audio2': 'VPN VoIP',
        'vpn-skype_audio1': 'VPN VoIP',
        'vpn-skype_audio2': 'VPN VoIP',
        'vpn-voipbuster1a': 'VPN VoIP',
        'vpn-voipbuster1b': 'VPN VoIP',
    }
    tor_app_labels = {
        # 1 AIM chat
        'nontor-AIM_Chat': 'AIM chat',
        'nontor-aimchat': 'AIM chat',
        'tor-CHAT_aimchatgateway': 'AIM chat',
        'tor-CHAT_gate_AIM_chat': 'AIM chat',

        # 2 Email
        'nontor-Email_IMAP_filetransfer': 'Email',
        'nontor-POP_filetransfer': 'Email',
        'tor-MAIL_gate_Email_IMAP_filetransfer': 'Email',
        'tor-MAIL_gate_POP_filetransfer': 'Email',

        # 3 Thunderbird
        'nontor-Workstation_Thunderbird_Imap': 'Thunderbird',
        'nontor-Workstation_Thunderbird_POP': 'Thunderbird',
        'tor-MAIL_Gateway_Thunderbird_Imap': 'Thunderbird',
        'tor-MAIL_Gateway_Thunderbird_POP': 'Thunderbird',

        # 4 Browser
        'nontor-browsing_ara': 'Browser',
        'nontor-browsing_ara2': 'Browser',
        'nontor-browsing_ger': 'Browser',
        'nontor-browsing': 'Browser',
        'nontor-browsing2-1': 'Browser',
        'nontor-browsing2-2': 'Browser',
        'nontor-browsing2': 'Browser',
        'nontor-SSL_Browsing': 'Browser',
        'nontor-ssl': 'Browser',
        'tor-BROWSING_gate_SSL_Browsing': 'Browser',
        'tor-BROWSING_ssl_browsing_gateway': 'Browser',
        'tor-BROWSING_tor_browsing_ara': 'Browser',
        'tor-BROWSING_tor_browsing_ger': 'Browser',
        'tor-BROWSING_tor_browsing_mam': 'Browser',
        'tor-BROWSING_tor_browsing_mam2': 'Browser',

        # 5 Facebook
        'nontor-facebook_Audio': 'Facebook',
        'nontor-facebook_chat': 'Facebook',
        'nontor-Facebook_Voice_Workstation': 'Facebook',
        'nontor-facebookchat': 'Facebook',
        'tor-CHAT_facebookchatgateway': 'Facebook',
        'tor-CHAT_gate_facebook_chat': 'Facebook',
        'tor-VOIP_Facebook_Voice_Gateway': 'Facebook',
        'tor-VOIP_gate_facebook_Audio': 'Facebook',

        # 6 FTP
        'nontor-FTP_filetransfer': 'FTP',
        'tor-FILE-TRANSFER_gate_FTP_transfer': 'FTP',

        # 7 Hangouts
        'nontor-hangout_chat': 'Hangouts',
        'nontor-Hangout_Audio': 'Hangouts',
        'nontor-Hangouts_voice_Workstation': 'Hangouts',
        'nontor-hangoutschat': 'Hangouts',
        'tor-CHAT_gate_hangout_chat': 'Hangouts',
        'tor-CHAT_hangoutschatgateway': 'Hangouts',
        'tor-VOIP_gate_hangout_audio': 'Hangouts',
        'tor-VOIP_Hangouts_voice_Gateway': 'Hangouts',

        # 8 MultipleSpeed
        'nontor-p2p_multipleSpeed': 'MultipleSpeed',
        'nontor-p2p_multipleSpeed2-1': 'MultipleSpeed',
        'tor-p2p_multipleSpeed2-1': 'MultipleSpeed',
        'tor-P2P_tor_p2p_multipleSpeed': 'MultipleSpeed',

        # 9 Vuze
        'nontor-p2p_vuze': 'Vuze',
        'nontor-p2p_vuze2-1': 'Vuze',
        'tor-P2P_tor_p2p_vuze': 'Vuze',
        'tor-p2p_vuze-2-1': 'Vuze',

        # 10 ICQ
        'nontor-ICQ_Chat': 'ICQ',
        'nontor-icqchat': 'ICQ',
        'tor-CHAT_gate_ICQ_chat': 'ICQ',
        'tor-CHAT_icqchatgateway': 'ICQ',

        # 11 SFTP
        'nontor-SFTP_filetransfer': 'SFTP',
        'tor-FILE-TRANSFER_gate_SFTP_filetransfer': 'SFTP',

        # 12 Skype
        'nontor-Skype_Audio': 'Skype',
        'nontor-skype_chat': 'Skype',
        'nontor-skype_transfer': 'Skype',
        'nontor-Skype_Voice_Workstation': 'Skype',
        'nontor-skypechat': 'Skype',
        'tor-CHAT_gate_skype_chat': 'Skype',
        'tor-CHAT_skypechatgateway': 'Skype',
        'tor-FILE-TRANSFER_tor_skype_transfer': 'Skype',
        'tor-VOIP_gate_Skype_Audio': 'Skype',
        'tor-VOIP_Skype_Voice_Gateway': 'Skype',

        # 13 Spotify
        'nontor-spotify': 'Spotify',
        'nontor-spotify2-1': 'Spotify',
        'nontor-spotify2-2': 'Spotify',
        'nontor-spotify2': 'Spotify',
        'nontor-spotifyAndrew': 'Spotify',
        'tor-AUDIO_spotifygateway': 'Spotify',
        'tor-AUDIO_tor_spotify': 'Spotify',
        'tor-AUDIO_tor_spotify2': 'Spotify',
        'tor-spotify2-1': 'Spotify',
        'tor-spotify2-2': 'Spotify',

        # 14 Vimeo
        'nontor-Vimeo_Workstation': 'Vimeo',
        'tor-VIDEO_Vimeo_Gateway': 'Vimeo',

        # 15 YouTube
        'nontor-Youtube_Flash_Workstation': 'YouTube',
        'nontor-Youtube_HTML5_Workstation': 'YouTube',
        'tor-VIDEO_Youtube_Flash_Gateway': 'YouTube',
        'tor-VIDEO_Youtube_HTML5_Gateway': 'YouTube',
    }
    tor_tra_labels = {
        # 1 Chat
        'nontor-AIM_Chat': 'Chat',
        'nontor-aimchat': 'Chat',
        'nontor-facebook_chat': 'Chat',
        'nontor-facebookchat': 'Chat',
        'nontor-hangout_chat': 'Chat',
        'nontor-hangoutschat': 'Chat',
        'nontor-ICQ_Chat': 'Chat',
        'nontor-icqchat': 'Chat',
        'nontor-skype_chat': 'Chat',
        'nontor-skypechat': 'Chat',

        # 2 Tor Chat
        'tor-CHAT_aimchatgateway': 'Tor Chat',
        'tor-CHAT_gate_AIM_chat': 'Tor Chat',
        'tor-CHAT_gate_hangout_chat': 'Tor Chat',
        'tor-CHAT_hangoutschatgateway': 'Tor Chat',
        'tor-CHAT_facebookchatgateway': 'Tor Chat',
        'tor-CHAT_gate_facebook_chat': 'Tor Chat',
        'tor-CHAT_gate_ICQ_chat': 'Tor Chat',
        'tor-CHAT_icqchatgateway': 'Tor Chat',
        'tor-CHAT_gate_skype_chat': 'Tor Chat',
        'tor-CHAT_skypechatgateway': 'Tor Chat',

        # 3 Email
        'nontor-Email_IMAP_filetransfer': 'Email',
        'nontor-POP_filetransfer': 'Email',
        'nontor-Workstation_Thunderbird_Imap': 'Email',
        'nontor-Workstation_Thunderbird_POP': 'Email',

        # 4 Tor Email
        'tor-MAIL_gate_Email_IMAP_filetransfer': 'Tor Email',
        'tor-MAIL_gate_POP_filetransfer': 'Tor Email',
        'tor-MAIL_Gateway_Thunderbird_Imap': 'Tor Email',
        'tor-MAIL_Gateway_Thunderbird_POP': 'Tor Email',

        # 5 Browsing
        'nontor-browsing_ara': 'Browsing',
        'nontor-browsing_ara2': 'Browsing',
        'nontor-browsing_ger': 'Browsing',
        'nontor-browsing': 'Browsing',
        'nontor-browsing2-1': 'Browsing',
        'nontor-browsing2-2': 'Browsing',
        'nontor-browsing2': 'Browsing',
        'nontor-SSL_Browsing': 'Browsing',
        'nontor-ssl': 'Browsing',

        # 6 Tor Browsing
        'tor-BROWSING_gate_SSL_Browsing': 'Tor Browsing',
        'tor-BROWSING_ssl_browsing_gateway': 'Tor Browsing',
        'tor-BROWSING_tor_browsing_ara': 'Tor Browsing',
        'tor-BROWSING_tor_browsing_ger': 'Tor Browsing',
        'tor-BROWSING_tor_browsing_mam': 'Tor Browsing',
        'tor-BROWSING_tor_browsing_mam2': 'Tor Browsing',

        # 7 File Transfer
        'nontor-FTP_filetransfer': 'File Transfer',
        'nontor-SFTP_filetransfer': 'File Transfer',
        'nontor-skype_transfer': 'File Transfer',

        # 8 Tor File Transfer
        'tor-FILE-TRANSFER_gate_FTP_transfer': 'Tor File Transfer',
        'tor-FILE-TRANSFER_gate_SFTP_filetransfer': 'Tor File Transfer',
        'tor-FILE-TRANSFER_tor_skype_transfer': 'Tor File Transfer',

        # 9 P2P
        'nontor-p2p_multipleSpeed': 'P2P',
        'nontor-p2p_multipleSpeed2-1': 'P2P',
        'nontor-p2p_vuze': 'P2P',
        'nontor-p2p_vuze2-1': 'P2P',

        # 10 Tor P2P
        'tor-p2p_multipleSpeed2-1': 'Tor P2P',
        'tor-P2P_tor_p2p_multipleSpeed': 'Tor P2P',
        'tor-P2P_tor_p2p_vuze': 'Tor P2P',
        'tor-p2p_vuze-2-1': 'Tor P2P',

        # 11 Video
        'nontor-Vimeo_Workstation': 'Video',
        'nontor-Youtube_Flash_Workstation': 'Video',
        'nontor-Youtube_HTML5_Workstation': 'Video',

        # 12 Tor Video
        'tor-VIDEO_Vimeo_Gateway': 'Tor Video',
        'tor-VIDEO_Youtube_Flash_Gateway': 'Tor Video',
        'tor-VIDEO_Youtube_HTML5_Gateway': 'Tor Video',

        # 13 Audio
        'nontor-spotify': 'Audio',
        'nontor-spotify2-1': 'Audio',
        'nontor-spotify2-2': 'Audio',
        'nontor-spotify2': 'Audio',
        'nontor-spotifyAndrew': 'Audio',

        # 14 Tor Audio
        'tor-AUDIO_spotifygateway': 'Tor Audio',
        'tor-AUDIO_tor_spotify': 'Tor Audio',
        'tor-AUDIO_tor_spotify2': 'Tor Audio',
        'tor-spotify2-1': 'Tor Audio',
        'tor-spotify2-2': 'Tor Audio',

        # 15 VoIP
        'nontor-Skype_Audio': 'VoIP',
        'nontor-facebook_Audio': 'VoIP',
        'nontor-Hangout_Audio': 'VoIP',
        'nontor-Skype_Voice_Workstation': 'VoIP',
        'nontor-Facebook_Voice_Workstation': 'VoIP',
        'nontor-Hangouts_voice_Workstation': 'VoIP',

        # 16 Tor VoIP
        'tor-VOIP_Facebook_Voice_Gateway': 'Tor VoIP',
        'tor-VOIP_gate_facebook_Audio': 'Tor VoIP',
        'tor-VOIP_gate_hangout_audio': 'Tor VoIP',
        'tor-VOIP_Hangouts_voice_Gateway': 'Tor VoIP',
        'tor-VOIP_gate_Skype_Audio': 'Tor VoIP',
        'tor-VOIP_Skype_Voice_Gateway': 'Tor VoIP',
    }
    ustc_app_labels = {
        'Benign-BitTorrent': 'BitTorrent',
        'Benign-Facetime': 'Facetime',
        'Benign-FTP': 'FTP',
        'Benign-Gmail': 'Gmail',
        'Benign-MySQL': 'MySQL',
        'Benign-Outlook': 'Outlook',
        'Benign-Skype': 'Skype',
        'Benign-SMB-1': 'SMB',
        'Benign-SMB-2': 'SMB',
        'Benign-Weibo-1': 'WeiBo',
        'Benign-Weibo-2': 'WeiBo',
        'Benign-Weibo-3': 'WeiBo',
        'Benign-Weibo-4': 'WeiBo',
        'Benign-WorldOfWarcraft': 'WorldOfWarcraft',
        'Malware-Cridex': 'Cridex',
        'Malware-Geodo': 'Geodo',
        'Malware-Htbot': 'Htbot',
        'Malware-Miuref': 'Miuref',
        'Malware-Neris': 'Neris',
        'Malware-Nsis-ay': 'Nsis-ay',
        'Malware-Shifu': 'Shifu',
        'Malware-Tinba': 'Tinba',
        'Malware-Virut': 'Virut',
        'Malware-Zeus': 'Zeus',
    }

    mypcap_app_labels = {
        'QQ': 'QQ',
        '微信': 'WeChat',
        'QQ音乐': 'QQMusic',
        '网易云音乐': 'CloudMusic',
        '爱奇艺': 'IQIYI',
        '腾讯视频': 'TencentVideo',
        '谷歌浏览器微博': 'ChromeDouBan',
        '谷歌浏览器豆瓣': 'ChromeWeiBo',
        '穿越火线': 'CrossFire',
        '英雄联盟': 'LOL',
    }

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

    root_path = r"E:\datasets\data"
    data_path = os.path.join(root_path, 'DeepClear')
    image_path = os.path.join(root_path, 'SessionVideo')
    datasets = [
        'ISCXVPN2016',
    #     'USTC-TF2016',
    #     'ISCXTor2016',
    #     'MyPcap',
    #       'Abuffer',
    ]
    # 生成ISCXVPN2016数据集application部分
    data_src = os.path.join(data_path, 'ISCXVPN2016')
    data_dst = os.path.join(image_path, 'ISCXVPN2016', 'AppZeroPort')

    ptsw = PcapToSessionWorker(data_src, data_dst, zero_port=True, labels=vpn_app_labels)
    ptsw.work()

    # data_src = os.path.join(data_path, 'ISCXTor2016')
    # data_dst = os.path.join(image_path, 'ISCXTor2016', 'traffic')
    #
    # ptsw = PcapToSessionWorker(data_src, data_dst, tor_tra_labels)
    # ptsw.work()

    # data_src = os.path.join(data_path, 'USTC-TF2016')
    # data_dst = os.path.join(image_path, 'USTC-TF2016', 'application')
    #
    # ptsw = PcapToSessionWorker(data_src, data_dst, ustc_app_labels)
    # ptsw.work()






