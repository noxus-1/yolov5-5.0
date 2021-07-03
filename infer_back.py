# from utils.datasets import *
# from utils.utils import *
import math
import os
import random
import shutil
import time
import cv2
import numpy as np
import torch
import torchvision
import glob
import torch
import copy
import json
import tqdm
import cv2
import os
import glob
import time


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def new_letterbox(image,size = 640):
    if image.shape[0] > image.shape[1]:
        w,h = int(image.shape[1]*size / image.shape[0]),size
        pad = ((0,0),(0,size - w),(0,0))
    else:
        w, h =  size , int(image.shape[0] * size / image.shape[1]),
        pad = ((0,size - h), (0, 0), (0, 0))
    image = cv2.resize(image,(w,h))
    image = np.pad(image,pad)

    return image,w,h
def letterbox(img):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    new_shape = 640
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # if not scaleup:  # only scale down, do not scale up (for better test mAP)
    #     r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
    return img, ratio, (dw, dh)
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
def non_max_suppression(prediction):

    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    conf_thres = 0.01
    iou_thres = 0.5
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        # i, j = (x[:, 5:] > conf_thres).nonzero().t()
        # x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        agnostic = False
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        # pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        pad = (0,0)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
class yolo5det:
    def __init__(self, weight='',imgsz=640,out='',device='0'):
        self.out = out
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        device = torch.device('cuda:0')
        # if os.path.exists(out):
        #     shutil.rmtree(out)  # delete output folder
        # os.makedirs(out)  # make new output folder
        self.model = torch.load(weight, map_location=device)['model'].float().fuse().eval()  # load FP32 model
        self.imgsz = imgsz
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names #classes
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))] # 随机生产框的颜色
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=device)  # init img
        _ = self.model(img) if device.type != 'cpu' else None  # run once
        self.device = device
    def plot_img(self,img,pred):
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)], line_thickness=2)
        return img
    @torch.no_grad()
    def detect(self,img,cla=(0,1,2,3,4)):
        out = self.out
        img0 = img.copy()
        h, w, _ = img0.shape
        # img = letterbox(img)[0]
        img,w,h = new_letterbox(img)
        # cv2.imwrite(self.out + f'resize{w}.jpg', img)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        #run
        pred = self.model(img, augment=False)
        # pred = self.model(img)
        # import pdb
        # pdb.set_trace()
        pred = pred[0]
        # Apply NMS
        _pred = non_max_suppression(pred)
        for i, det in enumerate(_pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                det[:, :4] = scale_coords([h,w], det[:, :4], img0.shape).round()
                det = det[(det[:, 5:6] == torch.tensor(cla, device=det.device)).any(1)]
        if out != '':
            os.makedirs(out,exist_ok=True)  # make new output folder
            for i, det in enumerate(_pred):  # detections per image
                if det is not None and len(det):
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=2)
            cv2.imwrite(self.out+f'debug{w}.jpg',img0)
        return _pred

if __name__ == '__main__':
    weight = "/home/linwang/pyproject/yolov5-5.0/runs/train/exp4/weights/best.pt"
    out = "/home/linwang/pyproject/yolov5-5.0/runs/detect/infer/"
    labelsave = "/home/linwang/pyproject/yolov5-5.0/ourdatas/test1000/label_yolo55x/"
    os.makedirs(out, exist_ok=True)
    device = '1'
    dir = "/home/linwang/pyproject/yolov5-5.0/ourdatas/test1000/img/"
    name = dir.split(os.sep)[-2]
    yolo5 = yolo5det(weight, out='', device=device )
    # import pdb
    # pdb.set_trace()
    json_list = []
    for imgpath in glob.glob(dir + '*.jpg'):
        # img = cv2.imread("/home/linwang/pyproject/yolov5s_bdd100k_backup/yolov5s_bdd100k-master/inference/images/street.jpg")
        imgname = imgpath.split(os.sep)[-1]
        img = cv2.imread(imgpath)
        img0 = img.copy()
        pred = yolo5.detect(img)
        img_out = yolo5.plot_img(img0,pred)
        cv2.imwrite(out + imgname, img0)
        if labelsave:
            target = pred[0]
            target = target.tolist()
            annot_dict = [('input', out + imgname), ('target', target)]
            dict1 = dict(annot_dict)
            json_list.append(dict1)
        print(len(pred[0]))
        # time.sleep(1)
    if labelsave:
        label_path = labelsave + str(name) + '.json'
        with open(label_path, 'w', encoding='utf-8') as f:
            for sample in json_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


