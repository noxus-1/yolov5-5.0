import pycuda.driver as cuda
# import tensorrt as trt
import numpy as np
import torch
import cv2
import os
import random
import glob
import tqdm
import json
from toolclass import walkfile


def xywh2xyxy(x):  # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def iou(bbox1, bbox2):
    assert bbox1.dim() == bbox2.dim()
    dim = bbox1.dim()

    bbox1_x1, bbox1_x2 = bbox1[..., 0], bbox1[..., 2]
    bbox1_y1, bbox1_y2 = bbox1[..., 1], bbox1[..., 3]
    bbox2_x1, bbox2_x2 = bbox2[..., 0], bbox2[..., 2]
    bbox2_y1, bbox2_y2 = bbox2[..., 1], bbox2[..., 3]
    bbox1_x1, bbox1_x2 = bbox1_x1.unsqueeze(dim-1), bbox1_x2.unsqueeze(dim-1)
    bbox1_y1, bbox1_y2 = bbox1_y1.unsqueeze(dim-1), bbox1_y2.unsqueeze(dim-1)
    bbox2_x1, bbox2_x2 = bbox2_x1.unsqueeze(dim-2), bbox2_x2.unsqueeze(dim-2)
    bbox2_y1, bbox2_y2 = bbox2_y1.unsqueeze(dim-2), bbox2_y2.unsqueeze(dim-2)

    x1 = torch.max(bbox1_x1, bbox2_x1)
    y1 = torch.max(bbox1_y1, bbox2_y1)
    x2 = torch.min(bbox1_x2, bbox2_x2)
    y2 = torch.min(bbox1_y2, bbox2_y2)
    inter_area = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)  # discrete value: 14~16 are 3 pixels
    bbox1_area = (bbox1_x2 - bbox1_x1 + 1) * (bbox1_y2 - bbox1_y1 + 1)
    bbox2_area = (bbox2_x2 - bbox2_x1 + 1) * (bbox2_y2 - bbox2_y1 + 1)
    union_area = bbox1_area + bbox2_area - inter_area
    iou = inter_area / (union_area + 1e-16)
    return iou


def nms(boxes, scores, iou_thres):
    indices = scores.sort(descending=True)[1]
    boxes = boxes[indices]
    output = []
    while boxes.size(0):
        output.append(int(indices[0]))
        invalid = iou(boxes[0: 1, :4], boxes[:, :4]) > iou_thres
        invalid = invalid.squeeze(0)
        boxes = boxes[~invalid]
        indices = indices[~invalid]
    return output


class HostDeviceMem(object):

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(eslf):
        return self.__str__()


class YoloPredictor:

    def __init__(self, model_path, gpus, use_trt=False ,imgsize = 640):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        self.image_size = imgsize
        self.use_trt = use_trt
        if not use_trt:
            self.model = torch.load(model_path, map_location="cuda:0")['model'].float().fuse().eval()
            self.model(torch.zeros(1, 3, self.image_size, self.image_size).cuda(0))
        else:
            import pycuda.autoinit
            self.model = self.load_engine(model_path)
            self.context = self.model.create_execution_context()
            self.stream = cuda.Stream()
            assert self.model.max_batch_size == 1
            self.inputs, self.outputs, self.bindings, self.output_shapes = self.allocate_buffers()
            self.anchors = {
                (20, 20): torch.tensor([[116., 90.], [156., 198.], [373., 326.]]).view(1, 3, 1, 1, 2).float(),
                (40, 40): torch.tensor([[30., 61.], [62., 45.], [59., 119.]]).view(1, 3, 1, 1, 2).float(),
                (80, 80): torch.tensor([[10., 13.], [16., 30.], [33., 23.]]).view(1, 3, 1, 1, 2).float(),
            }
            self.grid = {}
            for key in self.anchors:
                yv, xv = torch.meshgrid([torch.arange(key[0]), torch.arange(key[1])])
                self.grid[key] = torch.stack((xv, yv), 2).view((1, 1, key[0], key[1], 2)).float()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # classes
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]  # 随机生产框的颜色

    def load_engine(self, model_path):
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as fid, trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(fid.read())
            return engine

    def allocate_buffers(self):
        inputs, outputs, bindings, output_shapes = [], [], [], []
        for (idx, binding) in enumerate(self.model):
            shape = self.model.get_binding_shape(binding)
            shape = tuple([1] + list(shape[1:]))
            size = trt.volume(shape)
            dtype = trt.nptype(self.model.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.model.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                output_shapes.append(shape)
        return inputs, outputs, bindings, output_shapes

    def preprocess(self, image, size=640):
        if image.shape[0] > image.shape[1]:
            width, height = int(image.shape[1] * size / image.shape[0]), size
            pad = ((0, 0), (0, size - width), (0, 0))
        else:
            width, height = size, int(image.shape[0] * size / image.shape[1])
            pad = ((0, size - height), (0, 0), (0, 0))
        image = cv2.resize(image, (width, height))
        image = np.pad(image, pad)

        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = image / 255.
        return np.ascontiguousarray(image)

    def postprocess(self, prediction, conf_thres=0.01, iou_thres=0.5):  # NOTE: wanted: 0: pedestrian, 1~6: vehicle
        """detections with shape: nx6 (x1, y1, x2, y2, conf, cls)."""
        # import pdb
        # pdb.set_trace()
        nc = prediction[0].shape[1] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        # xc = torch.logical_and(prediction[..., 5:].max(dim=-1)[1] < 7, xc)  # NOTE for bdd
        xc = torch.logical_and(prediction[..., 5:].max(dim=-1)[1] < 8, xc)  # NOTE for coco
        output = [None] * prediction.size(0)
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]
            if not x.size(0):
                continue

            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = xywh2xyxy(x[:, :4])  # to xyxy
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            x[:, 5: 6] = (x[:, 5: 6] > 0).float()  # NOTE: person = 0 all veh = 1
            if not x.size(0):
                continue

            boxes, scores = x[:, :4] + x[:, 5: 6] * 4096, x[:, 4]
            i = nms(boxes, scores, iou_thres)
            output[xi] = x[i]
            # output[xi][:, 5: 6] = (output[xi][:, 5: 6] > 0).float()  # NOTE:  person = 0 all veh = 1

        return output

    def detect(self, image,conf_thres = 0.01 , ispyramid = False):
        if ispyramid:
            pred_pyramid = self.pyramid(image,cutlist = ispyramid ,conf_thres = conf_thres)
        input = self.preprocess(image, self.image_size )

        if not self.use_trt:
            input = torch.tensor(input).float().unsqueeze(dim=0).cuda(0)
            pred = self.model(input, augment=False)[0]
        else:
            input = np.float32(np.expand_dims(input, axis=0))
            self.inputs[0].host = input
            [cuda.memcpy_htod_async(input.device, input.host, self.stream) for input in self.inputs]
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            [cuda.memcpy_dtoh_async(output.host, output.device, self.stream) for output in self.outputs]
            self.stream.synchronize()
            pred = []
            for (shape, output) in zip(self.output_shapes, self.outputs):  # can not do post-processing on cuda due to tensorrt used before
                output = torch.from_numpy(output.host.reshape(shape)).float().sigmoid()
                output[..., 0: 2] = (output[..., 0: 2] * 2. - 0.5 + self.grid[tuple(shape[2: 4])]) * self.image_size / shape[2]
                output[..., 2: 4] = (output[..., 2: 4] * 2) ** 2 * self.anchors[tuple(shape[2: 4])]
                pred.append(output.view(shape[0], -1, shape[-1]))
            pred = torch.cat(pred, 1)

        pred = self.postprocess(pred,conf_thres = conf_thres)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = det[:, :4] / self.image_size * max(image.shape[:2])
                det[:, 0] = det[:, 0].clamp_(0, image.shape[1])
                det[:, 1] = det[:, 1].clamp_(0, image.shape[0])
                det[:, 2] = det[:, 2].clamp_(0, image.shape[1])
                det[:, 3] = det[:, 3].clamp_(0, image.shape[0])
                # image_vis = image.copy()
                # self.plot_img(image_vis,det)
                # os.makedirs("badcase", exist_ok=True)
                # savepath = os.path.join("badcase", f"{dataname}.png")
                # cv2.imwrite(f"result_{i}.jpg", image_vis)
        if ispyramid:
            pred[0] = torch.cat((pred[0],pred_pyramid[0]))
            import copy
            output = copy.deepcopy(pred)
            for xi, x in enumerate(pred):  # image index, image inference
                if not x.size(0):
                    continue
                boxes, scores = x[:, :4] + x[:, 5: 6] * 4096, x[:, 4]
                i = nms(boxes, scores, 0.5)
                output[xi] = x[i]
                pred = output
        return pred

    def pyramid(self, img,cutlist = 'highway',conf_thres = 0.01):
        image = img.copy()
        if cutlist == 'highway':
            h,w,_ = image.shape
            image = image[0:int(h*1/2),int(w*1/4):int(w*3/4)]
        else:
            raise ValueError("Check cutlist type")

        input = self.preprocess(image, self.image_size)

        if not self.use_trt:
            input = torch.tensor(input).float().unsqueeze(dim=0).cuda(0)
            pred = self.model(input, augment=False)[0]
        else:
            input = np.float32(np.expand_dims(input, axis=0))
            self.inputs[0].host = input
            [cuda.memcpy_htod_async(input.device, input.host, self.stream) for input in self.inputs]
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            [cuda.memcpy_dtoh_async(output.host, output.device, self.stream) for output in self.outputs]
            self.stream.synchronize()
            pred = []
            for (shape, output) in zip(self.output_shapes,
                                       self.outputs):  # can not do post-processing on cuda due to tensorrt used before
                output = torch.from_numpy(output.host.reshape(shape)).float().sigmoid()
                output[..., 0: 2] = (output[..., 0: 2] * 2. - 0.5 + self.grid[tuple(shape[2: 4])]) * self.image_size / \
                                    shape[2]
                output[..., 2: 4] = (output[..., 2: 4] * 2) ** 2 * self.anchors[tuple(shape[2: 4])]
                pred.append(output.view(shape[0], -1, shape[-1]))
            pred = torch.cat(pred, 1)

        pred = self.postprocess(pred, conf_thres=conf_thres)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = det[:, :4] / self.image_size * max(image.shape[:2])
                det[:, 0] = det[:, 0].clamp_(0, image.shape[1]) + int(w*1/4)  # for cut
                det[:, 1] = det[:, 1].clamp_(0, image.shape[0])
                det[:, 2] = det[:, 2].clamp_(0, image.shape[1]) + int(w*1/4)
                det[:, 3] = det[:, 3].clamp_(0, image.shape[0])
        return pred
    def plot_img(self,img,det):
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)], line_thickness=1)
                    # plot_one_box(xyxy, img, color=self.colors[int(cls)], line_thickness=1)
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
if __name__ == '__main__':
    '''
    predictor = YoloPredictor("/home/linwang/pyproject/yolov5-5.0/runs/train/exp4/weights/best.pt", gpus="1")
    # predictor = YoloPredictor("./runs/exp2_yolov5x_bdd/best_b1.trt", gpus="0", use_trt=True)
    image = cv2.imread("/home/linwang/pyproject/yolov5-5.0/data/images/bus.jpg")
    # imagepath = "/home/linwang/pyproject/yolov5s_bdd100k_backup/yolov5s_bdd100k-master/ourdatas/ourvideo_imgs/image/B001（1200W） 00_26_04-00_28_25/"
    # imagepath =  "/home/linwang/pyproject/yolov5s_bdd100k_backup/yolov5s_bdd100k-master/ourdatas/ourvideo_imgs/image/B001（1200W） 00_41_06-00_43_12/"
    imagepath = "/home/linwang/pyproject/yolov5s_bdd100k_backup/yolov5s_bdd100k-master/ourdatas/ourvideo_imgs/image/A001(200W) 00_07_41-00_08_50/"
    images = glob.glob(imagepath+'*.jpg')
    for img in tqdm.tqdm(images):
        name = img.split(os.sep)[-1]
        if ('1200w' in name) or ('1200W' in name) or ('200W'in name ) or ('200w' in name):
            image = cv2.imread(img)
            for step in range(20):
                conf = 0.02 + step/50
                output = predictor.detect(image,conf_thres = conf)
                for i, det in enumerate(output):  # detections per image
                    if det is not None and len(det):
                        image_vis = image.copy()
                        predictor.plot_img(image_vis, det)
                        os.makedirs(f"conf/1200conf{conf}", exist_ok=True)
                        os.makedirs(f"conf/200conf{conf}", exist_ok=True)
                        if  ('1200w'in name) or ('1200W' in name):
                            savepath = os.path.join(f"conf/1200conf{conf}", name)
                            cv2.imwrite(savepath, image_vis)
                        elif ('200W'in name )or ('200w' in name) :
                            savepath = os.path.join(f"conf/200conf{conf}", name)
                            cv2.imwrite(savepath, image_vis)
                        else:
                            continue
    '''
    weight = "/home/linwang/pyproject/yolov5/runs/train/exp4/weights/best.pt"
    out = "/home/linwang/pyproject/yolov5/runs/detect/infer_615/"
    labelsave = '/home/linwang/pyproject/yolov5/'
    os.makedirs(out, exist_ok=True)
    device = '1'
    dir = '/home/linwang/pyproject/yolov5/data/test/'
    name = dir.split(os.sep)[-2]
    yolo5 = YoloPredictor(weight, gpus="1",imgsize = 640)
    json_list = []
    for imgpath in tqdm.tqdm(walkfile(dir)):
            # img = cv2.imread("/home/linwang/pyproject/yolov5s_bdd100k_backup/yolov5s_bdd100k-master/inference/images/street.jpg")
            imgname = imgpath.split(os.sep)[-1]
            img = cv2.imread(imgpath)
            img0 = img.copy()
            pred = yolo5.detect(img , conf_thres = 0.1 , ispyramid = 'highway' )
            img_out = yolo5.plot_img(img0, pred[0])
            cv2.imwrite(out + imgname, img0)
            if labelsave:
                target = pred[0]
                target = target.tolist()      # 取决于要不要score
                # target = target.int().tolist()
                annot_dict = [('input', imgpath), ('target', target)]
                dict1 = dict(annot_dict)
                json_list.append(dict1)
            # print(len(pred[0]))
            # time.sleep(1)
    if labelsave:
            label_path = labelsave + str(name) + '.json'
            with open(label_path, 'w', encoding='utf-8') as f:
                for sample in json_list:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")