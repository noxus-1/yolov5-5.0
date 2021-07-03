# from utils.utils import compute_ap
from infer import iou as get_iou
from infer import YoloPredictor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import tqdm
import copy
import json
import cv2
import os
import time

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap
def ap_per_class(tp, conf, pred_cls, target_cls):
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    recalls, precisions = {}, {}
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            recalls[c] = list(recall[:, 0])  # recall at IoU=0.5
            precisions[c] = list(precision[:, 0])  # precision at IoU=0.5
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r,  f1, unique_classes.astype('int32'), ap,recalls, precisions
class Benchmark(object):

    def __init__(self, gpus="0"):
        super(Benchmark, self).__init__()
        self.gpus = gpus.split(",")

    def load_data(self, jsonpath):
        with open(jsonpath, "r") as fid:
            samples = [json.loads(line) for line in fid]
        return samples

    def compute_stats(self, preds, targets):
        iouv = torch.linspace(0.5, 0.95, 10).to(preds.device)  # iou vector for mAP@0.5:0.95
        if preds is None:
            return torch.zeros(0, iouv.numel(), dtype=torch.bool), torch.Tensor(), torch.Tensor(), targets[:, 4].tolist()

        preds = preds.float()
        correct = torch.zeros(preds.size(0), iouv.numel(), dtype=torch.bool).to(preds.device)
        detected = []  # detected targets
        for cls in torch.unique(targets[:, 4]):  # per class
            ti = torch.nonzero(cls == targets[:, 4]).view(-1)  # prediction indices
            pi = torch.nonzero(cls == preds[:, 5]).view(-1)  # target indices
            if pi.shape[0]:
                ious, i = get_iou(preds[pi, :4], targets[:, :4][ti]).max(1)
                for j in torch.nonzero(ious > iouv[0]):
                    d = ti[i[j]]  # detected target
                    if d not in detected:
                        detected.append(d)
                        correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                        if len(detected) == targets.size(0):  # all targets already located in image
                            break

        return correct.cpu(), preds[:, 4].cpu(), preds[:, 5].cpu(), targets[:, 4].tolist()

    @torch.no_grad()
    def _forward(self, config):
        index, config = config
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpus[index % len(self.gpus)])
        if config["model_type"] == "yolov5":
            model = YoloPredictor(config["model_path"], gpus=str(self.gpus[index % len(self.gpus)]),imgsize = config["imgsize"])
        elif config["model_type"] == "yolov5_tensorrt":
            model = YoloPredictor(config["model_path"], gpus=str(self.gpus[index % len(self.gpus)]), use_trt=True)
        else:
            raise ValueError("Check model type")

        metrics = {}
        for key, jsonpath in config["data"].items():
            samples = self.load_data(jsonpath)
            pbar = tqdm.tqdm(samples, position=index + 1)

            stats = []
            runtime = 0.0
            for i , sample in enumerate(pbar):
                target = torch.tensor(sample["target"]).float()  # num_bbox * [x, y, x, y, class]
                target = target[target[:, 4] < 7]  # NOTE: 0: pedestrian; 1~6: vehicle
                target[:, 4] = (target[:, 4] > 0).float()  # NOTE
                if target.size(0) > 0:
                    start = time.time()
                    pred = model.detect(cv2.imread(sample["input"]),ispyramid = config["trick"])[0]  # num_bbox * [x, y, x, y, score, class]
                    end = time.time()
                    if i > 1:
                        runtime += (end - start)
                    stats.append(self.compute_stats(pred, target.to(pred.device)))

            stats = [np.concatenate(x, 0) for x in zip(*stats)]
            ap,recalls, precisions= ap_per_class(*stats)[-3:]
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            map50, map =  ap50.mean(), ap.mean()
            metrics[key] = {"recall": recalls, "precision": precisions , 'mAP': (map50, map)}
            print(config["model_name"],' : ', (map50, map),(runtime,runtime/1134) )
        return metrics

    def forward(self, configs, in_pool, num_process=4):
        with torch.multiprocessing.Pool(num_process) as pool:
            metrics = list(tqdm.tqdm(pool.imap(self._forward, configs[:in_pool]), total=len(configs[:in_pool]), position=0))
        for config in configs[in_pool:]:
            metrics.append(self._forward(config))
        # import pdb
        # pdb.set_trace()
        for dataname in metrics[0].keys():
            with open(f'benchdata_{dataname}.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(metrics[0][dataname], ensure_ascii=False))
            plt.figure(figsize=(12, 8))
            plt.clf()
            for i, metric in enumerate(metrics):
                for classid in metric[dataname]["recall"].keys():
                    classname = "pedestrian" if classid == 0 else "vehicle"
                    color = "#" + "".join([random.choice('0123456789ABCDEF') for j in range(6)])
                    model_name = configs[i][1]["model_name"]
                    # print(configs[i][1]["model_name"] ,' : ',metric[dataname]["mAP"])
                    # label = f"{model_name}-{classname}"
                    label = f"{model_name}"
                    plt.plot(metric[dataname]["precision"][classid], metric[dataname]["recall"][classid], color=color, label=label)
            plt.xlabel("Precision")
            plt.ylabel("Recall")
            plt.xticks(torch.linspace(0, 1, 11).numpy())
            plt.yticks(torch.linspace(0, 1, 11).numpy())
            plt.grid()
            plt.legend(loc="best")
            os.makedirs("BENCHMARK", exist_ok=True)
            savepath = os.path.join("BENCHMARK", f"{dataname}.png")
            plt.savefig(savepath)
            plt.close()

if __name__ == "__main__":
    config_template = {
        "model_type": None,
        "model_name": None,
        "model_path": None,
        'trick': False,
        'imgsize' : 640,
        'conf_thres':0.01,
        "data": {
            # "bdd100k": "/home/linwang/pyproject/yolo5s_bdd100k/bdd100k/labels/newlabel.json",
            # 'debug' : "/home/linwang/pyproject/yolo5s_bdd100k/bdd100k/labels/debug.json",
            "ourdata": "/home/linwang/pyproject/yolov5-5.0/merged_label.json",
        }
    }
    models = [  # model_type, model_name, model_path,trick,imgsize
        ("yolov5", "yolo3325_pyramid", "/home/linwang/pyproject/yolov5-5.0/runs/train/3325/weights/best.pt",'highway',640),
        ("yolov5", "yolo3325", "/home/linwang/pyproject/yolov5-5.0/runs/train/3325/weights/best.pt", False, 640),
        ("yolov5", "yolo3325_1280", "/home/linwang/pyproject/yolov5-5.0/runs/train/3325/weights/best.pt", False, 1280),
        (
        "yolov5", "yolo3350_pyramid", "/home/linwang/pyproject/yolov5-5.0/runs/train/33503/weights/best.pt", 'highway', 640),
        ("yolov5", "yolo3350", "/home/linwang/pyproject/yolov5-5.0/runs/train/33503/weights/best.pt", False, 640),
        ("yolov5", "yolo3350_1280", "/home/linwang/pyproject/yolov5-5.0/runs/train/33503/weights/best.pt", False, 1280),
        (
        "yolov5", "yolo6775_pyramid", "/home/linwang/pyproject/yolov5-5.0/runs/train/6775/weights/best.pt", 'highway', 640),
        ("yolov5", "yolo6775", "/home/linwang/pyproject/yolov5-5.0/runs/train/6775/weights/best.pt", False, 640),
        ("yolov5", "yolo6775_1280", "/home/linwang/pyproject/yolov5-5.0/runs/train/6775/weights/best.pt", False, 1280),
        # ("cmu", "cmu", "./runs/exp2_yolov5x_bdd/best_int8_b1.trt"),
    ]

    configs = []
    for i in range(len(models)):
        config = copy.deepcopy(config_template)
        config["model_type"] = models[i][0]
        config["model_name"] = models[i][1]
        config["model_path"] = models[i][2]
        config["trick"] = models[i][3]
        config["imgsize"] = models[i][4]
        configs.append([i, config])

    benchmark = Benchmark(gpus="0,1,2,3,4,5")
    benchmark.forward(configs, in_pool=12, num_process=12)
    # benchmark._forward(configs[3])
