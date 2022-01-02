import pickle
import time
import warnings

import torch
import zmq

from models import Darknet
from utils.utils import non_max_suppression

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

context = zmq.Context()
# context.setsockopt(zmq.SNDHWM, 1)
# context.setsockopt(zmq.RCVHWM, 1)
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
device = "cuda" if torch.cuda.is_available() else "cpu"


class ReConvNetwork(torch.nn.Module):
    def __init__(self):
        super(ReConvNetwork, self).__init__()
        self.upstream = torch.nn.UpsamplingBilinear2d(size=(416, 416))

    def forward(self, x):
        x = self.upstream(x)
        return x


renet = ReConvNetwork().to(device)


# socket.bind("tcp://192.168.1.104:5555")

# pool = ThreadPoolExecutor(max_workers=4)
#
#
# def link_handle(pred, conn):
#     t_nms_start = time.time()
#     results = non_max_suppression(pred, 0.3, 0.5)
#     t_nms_end = time.time()
#
#     #  Send reply back to client
#     results = pickle.dumps(results, protocol=0)
#     conn.send(results)
#     print("nms-processing time: {},".format(t_process_end - t_process_start, t_nms_end - t_nms_start))

#

# def load_heavy_model(cfg="cfg/yolov4.cfg", img_size=416, weights='weights/yolov4_voc/yolo-v4-best-step1.pt', device="cuda" if torch.cuda.is_available() else "cpu"):
# def load_heavy_model(cfg="cfg/yolov4.cfg", img_size=416, weights='weights/yolov4_coco/best_1.pt', device="cuda" if torch.cuda.is_available() else "cpu"):
def load_heavy_model(cfg="cfg/yolov3-spp.cfg", img_size=416, weights='weights/yolov3-spp/best-step1.pt', device="cuda" if torch.cuda.is_available() else "cpu"):
    heavy_weight_model = Darknet(cfg, img_size)
    heavy_weight_model.load_state_dict(torch.load(weights, map_location=device)["model"])
    # Eval mode
    heavy_weight_model.to(device).eval()
    return heavy_weight_model


if __name__ == '__main__':
    heavy_weight_model = load_heavy_model()

    while True:
        #  Wait for next request from client
        data = socket.recv()
        data = pickle.loads(data)
        data = renet(data)

        # Do some 'work'对接收的数据进行大模型处理
        t_process_start = time.time()
        with torch.no_grad():
            pred, _ = heavy_weight_model(data)
        t_process_end = time.time()

        t_nms_start = time.time()
        # results = nms(pred, 0.5, 0.5)
        results = non_max_suppression(pred, 0.3, 0.5)
        t_nms_end = time.time()

        #  Send reply back to client
        results = pickle.dumps(results, protocol=0)
        socket.send(results)

        print("heavy-model inference time: {}, nms-processing time: {},".format(t_process_end - t_process_start, t_nms_end - t_nms_start))
        # print(t4)
