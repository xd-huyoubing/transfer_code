import multiprocessing
import os
import pickle
import random
import shutil
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2
import numpy as np
import torch
import zmq
from torchvision import transforms
from utils.datasets import letterbox
from utils.parse_config import parse_data_cfg
from utils.utils import scale_coords, load_classes

classes = load_classes(parse_data_cfg("data/voc.data")["names"])
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

context = zmq.Context()
# context.setsockopt(zmq.SNDHWM, 51200)
# context.setsockopt(zmq.RCVHWM, 51200)
#  Socket to talk to server
# print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)

socket.connect("tcp://127.0.0.1:5555")
# socket.connect("tcp://192.168.3.75:5555")

device = "cuda" if torch.cuda.is_available() else "cpu"

out = "output"
pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

tt = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])


def transfer_image(img, img_size=416, half=False):
    # img = letterbox(img, new_shape=img_size)[0]
    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img.to(device)


class DemoNetwork(torch.nn.Module):
    def __init__(self):
        super(DemoNetwork, self).__init__()
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.max_pooling(x)
        return x


net = DemoNetwork().to(device)


def transfer_image_pil(img, img_size=416, half=False):
    # img = letterbox(img, new_shape=img_size)[0]
    # Normalize RGB
    img = tt(img).to(device)

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    img = net(img)
    return img


def link_handle(im0, img, detect_results):
    for i, det in enumerate(detect_results):  # detections per image
        # s = "%gx%g " % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


def read_images(path="/home/youbing/workspace/yolo_datasets/nano/images", out="output", discriminator_path="./edge-cloud/models/svm.pth"):
    # def read_images(path="./data/samples/", out="output",
    #                 discriminator_path="./edge-cloud/models/svm.pth"):
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    images = os.listdir(path)
    total_times = []
    pre_processing_times = []
    scale_coords_times = []

    for idx, image in enumerate(images):
        image_path = os.path.join(path, image)
        # im0 = cv2.imread(image_path)
        #
        # _, img = cv2.imencode(".jpg", im0, encode_param)

        im0 = Image.open(image_path).convert("RGB")

        # pre-processing time: t1 - t0
        t0 = time.time()
        img = transfer_image_pil(im0)
        data = pickle.dumps(img, protocol=0)
        t1 = time.time()

        # send image to server time(transpose time) : t3 - t2
        socket.send(data)

        #  Get the reply time. : t5 - t4
        message = socket.recv()
        detect_results = pickle.loads(message)

        # print("Received reply %s [ %s ]" % (request, message))
        t5 = time.time()
        pool.submit(link_handle, im0, img, detect_results)

        post_processing_end = time.time()
        single_time = post_processing_end - t0
        total_times.append(single_time)
        pre_processing_times.append(t1 - t0)
        scale_coords_times.append(post_processing_end - t5)

        print("Total time for each picture: {}, pre-processing time: {},scale_coords time: {}".format(single_time, t1 - t0, post_processing_end - t5))
    # print("Total time for each picture: {}, pre-processing time: {},scale_coords time: {}".format(np.mean(total_times), np.mean(pre_processing_times),
    #                                                                                               np.mean(scale_coords_times)))


# --cfg cfg/coco_light_yolov4.cfg --data data/coco.data  --weights weights/yolov4_coco/best_458.pt --source /media/tx2/storage_dev/detection_dataset/val2017
if __name__ == '__main__':
    read_images()
