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

def plot_bench(metrics):
    '''
     metrics[0][key] = {"recall": recalls, "precision": precisions}
    '''
    plt.figure(figsize=(12, 8))
    plt.clf()
    # import pdb
    # pdb.set_trace()
    for dataname in metrics[0].keys():
        for i, metric in enumerate(metrics):
            for classid in metric[f'{i}']["recall"].keys():
                classname = "pedestrian" if classid == '0.0' else "vehicle"
                color = "#" + "".join([random.choice('0123456789ABCDEF') for j in range(6)])
                model_name = 'yolov5_old' if dataname == '0' else 'yolov5_new'
                label = f"{model_name}-{classname}"
                plt.plot(metric[dataname]["precision"][classid], metric[dataname]["recall"][classid], color=color,
                         label=label)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.xticks(torch.linspace(0, 1, 11).numpy())
    plt.yticks(torch.linspace(0, 1, 11).numpy())
    plt.grid()
    plt.legend(loc="best")
    os.makedirs("BENCHMARK", exist_ok=True)
    savepath = os.path.join("BENCHMARK", "benchmark529.png")
    plt.savefig(savepath)
    plt.close()

if __name__ == '__main__':
    data = {}
    jsonpath = "/home/linwang/pyproject/yolov5s_bdd100k_backup/yolov5s_bdd100k-master/benchdata_bdd100k.json"

    f = open(jsonpath)
    info = json.load(f)
    data['0'] = info
    jsonpath1 = "/home/linwang/pyproject/yolov5-5.0/benchdata_bdd100k.json"
    f1 = open(jsonpath1)
    info1 = json.load(f1)
    data['1'] = info1
    datas = []
    datas.append(data)
    plot_bench(datas)
