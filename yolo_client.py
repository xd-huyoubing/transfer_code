import os
import pickle
import random
import shutil
import socket
import struct
import sys
import time

import cv2
import numpy as np
import torch

from models import Darknet
from utils.datasets import letterbox
from utils.parse_config import parse_data_cfg
from utils.utils import load_classes, non_max_suppression, scale_coords, plot_one_box

server_info = ("192.168.1.104", 9999)
# server_info = ("127.0.0.1", 9999)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(server_info)

device = "cuda" if torch.cuda.is_available() else "cpu"
classes = load_classes(parse_data_cfg("data/voc.data")["names"])
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]


def transfer_image(img, img_size=416, half=False):
    img = letterbox(img, new_shape=img_size)[0]

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(
        img, dtype=np.float16 if half else np.float32
    )  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def load_light_model(cfg="cfg/prune_0.8_keep_0.1_20_shortcut_yolov4.cfg", img_size=416, weights='weights/best.pt'):
    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    # Eval mode
    model.to(device).eval()

    return model


light_model = load_light_model()


def calc_area(coordinates, w, h):
    x1, y1, x2, y2 = np.array(coordinates.cpu()[0])
    return (x2 - x1) * (y2 - y1) / (w * h)


def get_light_model_detect_feature(detect_results, img, im0):
    if detect_results[0] is None:
        return np.array([[0, 0, 0, 0]])
    features = []
    w, h = im0.shape[0], im0.shape[1]
    for i, det in enumerate(detect_results):  # detections per image
        # s = "%gx%g " % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            features.append(calc_area(det[:, :4], w, h))
    features.sort()
    return np.array([[features[0], features[-1], len(features) / 20, w * h / 250000]])


# def read_images(path="data/samples", out="output1", discriminator_path="./edge-cloud/models/knn.pth"):
def read_images(path="/home/youbing/workspace/yolo_datasets/nano/images", out="output", discriminator_path="./edge-cloud/models/svm.pth"):
    # def read_images(path="/home/nano/Desktop/yolo_datasets/images", out="output"):
    """
    Args:
        path: path代表的是文件夹，下面有很多图片，我们一张一张的读取图片进行处理
    Returns: 返回检测好的图片结果
    """
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    images = os.listdir(path)
    total_num = len(images)
    total_times = []
    t0 = time.time()
    discriminator = torch.load(discriminator_path)
    server_count = 0
    for idx, image in enumerate(images):
        image_path = os.path.join(path, image)

        t = time.time()
        im0 = cv2.imread(image_path)
        img = transfer_image(im0)
        # print(sys.getsizeof(im0))
        # simple-case 在本地的Jetson Nano上处理
        with torch.no_grad():
            pred, _ = light_model(img)
        detect_results = non_max_suppression(pred, 0.3, 0.5)
        detect_features = get_light_model_detect_feature(detect_results, img, im0)
        discriminator_out = discriminator.predict(detect_features)
        # discriminator_out = discriminator(pred)
        # _, pp = torch.max(discriminator_out.data, dim=1)
        # TODO  这里嵌入难例判别器的代码，如果是难例，则由在nano上由小模型处理，否则通过网络发送到服务器由大模型处理
        if discriminator_out:  # hard-case  发送到服务器进行处理
            server_count += 1
            # 发送客户端数据到服务器
            # -----------------------------------------------------------------
            data = pickle.dumps(img, protocol=0)
            size = sys.getsizeof(data)
            header = struct.pack("i", size)
            # send data
            client_socket.sendall(header)
            client_socket.sendall(data)
            # -----------------------------------------------------------------
            print(image + " send to server detect")
            # 接收服务的计算好的数据
            # -----------------------------------------------------------------
            # Receive header
            recv_header = client_socket.recv(4)
            recv_size = struct.unpack('i', recv_header)
            # Receive data
            recv_data = b""
            while sys.getsizeof(recv_data) < recv_size[0]:
                recv_data += client_socket.recv(recv_size[0])
            detect_results = pickle.loads(recv_data)
            # -----------------------------------------------------------------

        single_time = time.time() - t
        # print("[%d/%d] %s Done. (%.5fs)" % (total_num, idx + 1, image_path, single_time))
        total_times.append(single_time)
        for i, det in enumerate(detect_results):  # detections per image
            # s = "%gx%g " % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                if discriminator_out:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     # s += "%g %ss, " % (n, classes[int(c)])  # add to string
                # Write results
                for *xyxy, conf, _, cls in det:
                    # if save_txt:  # Write to file
                    #     with open(save_path + ".txt", "a") as file:
                    #         file.write(("%g " * 6 + "\n") % (*xyxy, cls, conf))

                    if True:  # Add bbox to image
                        label = "%s %.3f" % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
            # Stream results
            # if True:
            #     cv2.imshow("res", im0)
            cv2.imwrite(os.path.join(out, image), im0)

    print("local detect: {}, server detect: {}".format(total_num - server_count, server_count))
    print("Done. (%.5fs)" % (time.time() - t0))
    print("Done. (%.5fs)" % (np.mean(total_times)))


if __name__ == '__main__':
    read_images()
    client_socket.close()
