import numpy as np
import torch
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

NMS_THRESH = 0.5


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
    inter_area = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)
    bbox1_area = (bbox1_x2 - bbox1_x1 + 1) * (bbox1_y2 - bbox1_y1 + 1)
    bbox2_area = (bbox2_x2 - bbox2_x1 + 1) * (bbox2_y2 - bbox2_y1 + 1)
    union_area = bbox1_area + bbox2_area - inter_area
    iou = inter_area / (union_area + 1e-16)
    return iou


def nms(boxes, scores):
    indices = scores.sort(descending=True)[1]
    boxes = boxes[indices]
    output = []
    while boxes.size(0):
        output.append(int(indices[0]))
        invalid = iou(boxes[0: 1, :4], boxes[:, :4]) > NMS_THRESH
        invalid = invalid.squeeze(0)
        boxes = boxes[~invalid]
        indices = indices[~invalid]
    return output


def load(jsonpath, conf_thresh):
    samples = {}
    with open(jsonpath, "r") as fid:
        for line in fid:
            data = json.loads(line)
            if isinstance(data,str):
                data = eval(data)
            # import pdb
            # pdb.set_trace()
            samples[data["input"]] = []
            for bbox in data["target"]:  # [x1, y1, x2, y2, score, label]
                if bbox[-2] > conf_thresh:
                    samples[data["input"]].append(bbox)
    return samples


def merge(configs):
    samples = [load(jsonpath, conf_thresh) for (jsonpath, conf_thresh) in configs]
    with open("merged_label.json", "w") as fid:
        for key in samples[0].keys():
            x = torch.tensor(samples[0][key]).float().cuda(0)
            for i in range(1, len(samples)):
                x = torch.cat([x, torch.tensor(samples[i][key]).float().cuda(0)], dim=0)
            boxes, scores = x[:, :4] + x[:, 5: 6] * 4096, x[:, 4]
            indices = nms(boxes, scores)
            result = x[indices].data.cpu().numpy().tolist()
            fid.write(json.dumps({"input": key, "target": result}) + "\n")


if __name__ == "__main__":
    configs = [  # (jsonpath, conf_thresh)
        ("/home/linwang/pyproject/yolov5-5.0/ourdatas/testgt/label/img.json", 0.1),

        ("/home/linwang/pyproject/yolov5-5.0/ourdatas/testgt/label/cmu_img.json", 0.1),
    ]
    merge(configs)
