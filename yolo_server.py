# https://www.jb51.net/article/167099.htm

import pickle
import socketserver
import struct
import sys
import time
import warnings

import torch

from models import Darknet
from utils.utils import non_max_suppression

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# ip_port = ("192.168.1.102", 9999)


# ip_port = ("127.0.0.1", 9999)
ip_port = ("192.168.3.6", 9999)


# 1、 加载模型
def load_model(cfg="cfg/yolov4.cfg", img_size=416, weights='weights/yolov4/yolo-v4-best-step1.pt',
               device="cuda" if torch.cuda.is_available() else "cpu"):
    heavy_weight_model = Darknet(cfg, img_size)
    heavy_weight_model.load_state_dict(torch.load(weights, map_location=device)["model"])
    # Eval mode
    heavy_weight_model.to(device).eval()
    return heavy_weight_model


class YOLOServer(socketserver.BaseRequestHandler):

    def handle(self):
        print("{}, connected to the server".format(self.client_address))
        heavy_weight_model = load_model()
        while True:
            try:
                # 接收数据并解析，接收数据的时候先接收数据的长度，在接收数据
                # -----------------------------------------------------------------
                # Receive Header
                recv_header = self.request.recv(4)
                if not recv_header:
                    break
                recv_size = struct.unpack('i', recv_header)

                # Receive data
                recv_data = b""
                while sys.getsizeof(recv_data) < recv_size[0]:
                    recv_data += self.request.recv(recv_size[0])
                data = pickle.loads(recv_data)
                t_recv_time = time.time()
                # -----------------------------------------------------------------

                # 对接收的数据进行大模型处理
                # -----------------------------------------------------------------
                inference_time_start = time.time()
                with torch.no_grad():
                    pred, _ = heavy_weight_model(data)

                nms_time = time.time()

                results = non_max_suppression(pred, 0.3, 0.5)
                inference_time_end = time.time()
                # -----------------------------------------------------------------

                # 处理结果返回客户端
                # -----------------------------------------------------------------
                t_send_timestamp = time.time()
                results = pickle.dumps(results, protocol=0)
                size = sys.getsizeof(results)
                header = struct.pack("i", size)
                self.request.sendall(header)
                self.request.sendall(results)
                # -----------------------------------------------------------------
                print(
                    "recv data timestamp: {}, inference time: {}, nms_time:{}, send result to client timestamp: {}".format(
                        t_recv_time, nms_time - inference_time_start, inference_time_end - nms_time,
                        t_send_timestamp))
            except ConnectionResetError as e:
                print('出现错误', e)
                break
        self.request.close()
        print("{} exited the connection".format(self.client_address))


if __name__ == '__main__':
    server_example = socketserver.ThreadingTCPServer(ip_port, YOLOServer)
    print("Starting YOLO service......")
    server_example.serve_forever()
