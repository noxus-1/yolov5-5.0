import glob
import multiprocessing
import random
import torch
import json
import tqdm
import cv2,os
import xml.etree.ElementTree as ET
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent.parent.parent))
# 导入minidom
from xml.dom import minidom
import requests, cv2, math
from urllib import request
from socket import *
import numpy as np
import time
import shutil

class VMCI(object):
    """通信类"""

    def Sendtojava(self, request_url, json_data):
        """调用api"""
        try:
            req_url = request_url
            r = requests.post(url=req_url, json=json_data)
            print('数据是否发送： ', r.text)
        except request.HTTPError:
            # print("there is an error")
            pass  # 跳过错误，不进行处理，直接继续执行

    def Clienttcp(self, HOST, PORT, BUFSIZ, Senddata):
        """TCP 客户机"""
        ADDR = (HOST, PORT)
        tcpCliSock = socket(AF_INET, SOCK_STREAM)
        tcpCliSock.connect(ADDR)
        tcpCliSock.send(Senddata.encode())
        Img_src = tcpCliSock.recv(BUFSIZ).decode()
        # if not Img_src:
        # continue
        # time.sleep(1)
        tcpCliSock.close()
        print('\n')

    def writetxt(seif, bad, m):  # 写入txt接口到f：\zzh.txt
        if bad[0] == '正常片':
            s = 0
        else:
            s = 1
        t = m
        try:
            try:
                txt = open('E:\\zzh.txt', 'w+')
                n = str(t) + '\n' + str(s)
                txt.write(n)
                txt.close()
                print('xieru:', m, n)
            except:
                time.sleep(0.05)
                txt = open('E:\\zzh.txt', 'w+')
                n = str(t) + '\n' + str(s)
                txt.write(n)
                txt.close()
                print('xieru:', m, n)
        except:
            time.sleep(0.05)
            txt = open('E:\\zzh.txt', 'w+')
            n = str(t) + '\n' + str(s)
            txt.write(n)
            txt.close()
            print('xieru:', m, n)

    def writehalm(self, bad, m):  # halm软件写入脚本
        if bad[0] == '正常片':
            s = 0
        else:
            s = 1
        # print(m)
        name = m.split('.')[0]
        realname = name.split('_')
        try:
            try:
                txt = open('E:\\zzh.txt', 'w+')
                n = str(realname[1]) + '\n' + str(realname[2]) + '\n' + str(realname[3]) + '\n' + str(s)
                txt.write(n)
                txt.close()
                print('xieru:', m, n)
            except:
                time.sleep(0.05)
                txt = open('E:\\zzh.txt', 'w+')
                n = str(realname[1]) + '\n' + str(realname[2]) + '\n' + str(realname[3]) + '\n' + str(s)
                txt.write(n)
                txt.close()
                print('xieru:', m, n)
        except:
            time.sleep(0.05)
            txt = open('E:\\zzh.txt', 'w+')
            n = str(realname[1]) + '\n' + str(realname[2]) + '\n' + str(realname[3]) + '\n' + str(s)
            txt.write(n)
            txt.close()
            print('xieru:', m, n)

class writexml(object):
    """预测的标签数据重新写入xml"""

    def writetoxml(self, gz, save_path):
        """
        :param gz: ['10943', 520, 520, 1, ['断栅', 194, 13, 263, 27], ['断栅', 69, 166, 138, 194], ['断栅', 374, 41, 429, 69]]
        :return: 新生成的xml
        """
        dom = minidom.Document()  # 1.创建DOM树对象

        root_node = dom.createElement('annotation')  # 2.创建根节点。每次都要用DOM对象来创建任何节点。

        root_node.setAttribute('verified', 'no')  # 3.设置根节点的属性

        dom.appendChild(root_node)  # 4.用DOM对象添加根节点

        # 用DOM对象创建元素子节点
        # folder
        folder_node = dom.createElement('folder')
        root_node.appendChild(folder_node)  # 用父节点对象添加元素子节点
        folder_text = dom.createTextNode('预测图片做标签')  # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        folder_node.appendChild(folder_text)  # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点

        # filename
        filename_node = dom.createElement('filename')
        root_node.appendChild(filename_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        filename_text = dom.createTextNode(str(gz[0]))  # gz[0]
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        filename_node.appendChild(filename_text)

        # path
        path_node = dom.createElement('path')
        root_node.appendChild(path_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        path_text = dom.createTextNode('默认路径')
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        path_node.appendChild(path_text)

        # source
        source_node = dom.createElement('source')
        root_node.appendChild(source_node)
        # database
        database_node = dom.createElement('database')
        source_node.appendChild(database_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        database_text = dom.createTextNode('Unknown')
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        database_node.appendChild(database_text)

        # size
        size_node = dom.createElement('size')
        root_node.appendChild(size_node)
        # width
        width_node = dom.createElement('width')
        size_node.appendChild(width_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        width_text = dom.createTextNode(str(gz[1]))  # gz[1]
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        width_node.appendChild(width_text)
        # height
        height_node = dom.createElement('height')
        size_node.appendChild(height_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        height_text = dom.createTextNode(str(gz[2]))  # gz[2]
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        height_node.appendChild(height_text)
        # depth
        depth_node = dom.createElement('depth')
        size_node.appendChild(depth_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        depth_text = dom.createTextNode(str(gz[3]))  # gz[3]
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        depth_node.appendChild(depth_text)

        # segmented
        segmented_node = dom.createElement('segmented')
        root_node.appendChild(segmented_node)
        # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
        segmented_text = dom.createTextNode('0')
        # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
        segmented_node.appendChild(segmented_text)

        length = len(gz)  # 求数据输入长度
        for i in range(length):
            if i < 4:  # 从第5个开始
                continue
            # object
            object_node = dom.createElement('object')
            root_node.appendChild(object_node)
            # name
            name_node = dom.createElement('name')
            object_node.appendChild(name_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            name_text = dom.createTextNode(str(gz[i][0]))  # gz[i][0]
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            name_node.appendChild(name_text)
            # pose
            pose_node = dom.createElement('pose')
            object_node.appendChild(pose_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            pose_text = dom.createTextNode('Unspecified')
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            pose_node.appendChild(pose_text)
            # truncated
            truncated_node = dom.createElement('truncated')
            object_node.appendChild(truncated_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            truncated_text = dom.createTextNode('0')
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            truncated_node.appendChild(truncated_text)
            # Difficult
            Difficult_node = dom.createElement('Difficult')
            object_node.appendChild(Difficult_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            Difficult_text = dom.createTextNode('0')
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            Difficult_node.appendChild(Difficult_text)
            # bndbox
            bnd_node = dom.createElement('bndbox')
            object_node.appendChild(bnd_node)
            # xmin
            xmin_node = dom.createElement('xmin')
            bnd_node.appendChild(xmin_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            xmin_text = dom.createTextNode(str(gz[i][1]))  # gz[i][1]
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            xmin_node.appendChild(xmin_text)
            # ymin
            ymin_node = dom.createElement('ymin')
            bnd_node.appendChild(ymin_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            ymin_text = dom.createTextNode(str(gz[i][2]))  # gz[i][2]
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            ymin_node.appendChild(ymin_text)
            # xmax
            xmax_node = dom.createElement('xmax')
            bnd_node.appendChild(xmax_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            xmax_text = dom.createTextNode(str(gz[i][3]))  # gz[i][3]
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            xmax_node.appendChild(xmax_text)
            # ymax
            ymax_node = dom.createElement('ymax')
            bnd_node.appendChild(ymax_node)
            # 用DOM创建文本节点，把文本节点（文字内容）看成子节点
            ymax_text = dom.createTextNode(str(gz[i][4]))  # gz[i][4]
            # 用添加了文本的节点对象（看成文本节点的父节点）添加文本节点
            ymax_node.appendChild(ymax_text)

        # 每一个结点对象（包括dom对象本身）都有输出XML内容的方法，如：toxml()--字符串, toprettyxml()--美化树形格式。

        try:
            with open(save_path + '%s.xml' % (str(gz[0])), 'w', encoding='UTF-8') as fh:
                # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
                # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
                dom.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')
                print('写入xml OK!')
        except Exception as err:
            print('错误信息：{0}'.format(err))

class Judge_EL(object):

    def ELsummary(self, jsonObj):

        """接收{"黑斑黑点": "1", "水印": "1", "划痕": "10"}，返回根据规则判断的列表"""
        ELclassify = []
        for key in jsonObj.keys():
            if key == "同心圆发黑":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            # elif key == "黑边":
            # if int(jsonObj[key]) > 0:
            # ELclassify.append(key)
            elif key == "隐裂":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            elif key == "断栅":
                if int(jsonObj[key]) > 6:
                    ELclassify.append(key)
            elif key == "连续断栅":
                if int(jsonObj[key]) > 0:
                    ELclassify.append("断栅")
            elif key == "雾状发黑":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            elif key == "大划痕":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            # elif key == "划痕":
            #    if int(jsonObj[key]) > 2:
            #        ELclassify.append(key)
            elif key == "小划痕":
                if int(jsonObj[key]) > 7:
                    ELclassify.append(key)
            # elif key == "吸球印吸盘印":
            #    if int(jsonObj[key]) > 0:
            #        ELclassify.append(key)
            elif key == "气流片":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            elif key == "水印手指印":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
            elif key == "皮带印滚轮印":
                if int(jsonObj[key]) > 0:
                    ELclassify.append(key)
        return ELclassify

    def transform_form(self, thresh, transform_type, scale):
        # 输入图片和模式，输出图片
        """开、闭运算操作"""
        kern = int((scale + 0.2) * 3)
        if transform_type == 'closing':  # 闭运算：先膨胀再腐蚀
            dilation_kernel = np.ones((kern, kern), np.uint8)  # 增加白色区域
            dilation = cv2.dilate(thresh, dilation_kernel, iterations=1)
            erode_kernel = np.ones((kern, kern), np.uint8)
            closing_image = cv2.erode(dilation, erode_kernel, iterations=1)  # 腐蚀 增加黑色区域
            return closing_image
        elif transform_type == 'opening':  # 开运算；先腐蚀再膨胀
            erode_kernel = np.ones((kern, kern), np.uint8)
            # erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kern, kern))
            erode = cv2.erode(thresh, erode_kernel, iterations=1)  # 腐蚀 增加黑色区域
            dilation_kernel = np.ones((kern, kern), np.uint8)
            # dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kern, kern))
            opening_image = cv2.dilate(erode, dilation_kernel, iterations=2)
            return opening_image

    def TFhbhd(self, img, zone, dingdian):
        """
        :param img: 图片，直接读取的图片，未经过灰度转化，如果输入灰度图，下面的灰度转化语句需要注释掉
        :param zone: 数组，四通道[[xmin,xmax,ymin,ymax]]
        :return: 'y'指判断黑斑黑点，'n'判断非黑斑黑点
        """

        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 求输入图片背景灰度

        imgraycopy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        xd1 = dingdian[0][0]
        yd1 = dingdian[0][1]
        xd2 = dingdian[1][0]
        yd2 = dingdian[1][1]
        xd3 = dingdian[2][0]
        yd3 = dingdian[2][1]
        xd4 = dingdian[3][0]
        yd4 = dingdian[3][1]
        xd5 = dingdian[4][0]
        yd5 = dingdian[4][1]
        xd6 = dingdian[5][0]
        yd6 = dingdian[5][1]
        xd7 = dingdian[6][0]
        yd7 = dingdian[6][1]
        xd8 = dingdian[7][0]
        yd8 = dingdian[7][1]
        k1 = (yd2 - yd1) / (xd2 - xd1)
        k2 = (yd3 - yd2) / (xd3 - xd2)
        k3 = (yd4 - yd3) / (xd4 - xd3)
        k4 = (xd5 - xd4) / (yd5 - yd4)
        k5 = (yd6 - yd5) / (xd6 - xd5)
        k6 = (yd7 - yd6) / (xd7 - xd6)
        k7 = (yd8 - yd7) / (xd8 - xd7)
        k8 = (xd1 - xd8) / (yd1 - yd8)
        '''
        imgray1 = abs(imgraycopy - 127)
        w2, h2 = imgray.shape
        overallarea, overallgray = 0, 0
        scale = w2 / 520
        for x in range(0, w2, 2):
            for y in range(0, h2, 2):
                if imgray1[x][y] < 80:
                    overallarea += 1
                    overallgray += imgray[x][y]
        overallagray = overallgray / (overallarea + 1)
        print(scale, overallagray)
        # 根据求得的背景灰度agray，将图片整体灰度调整，规避灰度越界，标准灰度163
        for x1 in range(w2):
            #if max(imgray[x1]) > 255:
            for y1 in range(h2):
                    if imgray[x1][y1] * (163 / overallagray) > 255:
                        imgray[x1][y1] = 255
                    else:
                        imgray[x1][y1] * (163 / overallagray)
        '''
        w2, h2 = imgray.shape
        overallarea, overallgray = 0, 0
        scale = w2 / 520
        # 根据调整后的图片和标记区间zone（数组），求每个区间的属性并计算属于那种黑斑黑点
        qiandian, shendian, qianban, shenban, median = 0, 0, 0, 0, 0
        for p in range(len(zone)):
            image = imgray[zone[p][1]:zone[p][3], zone[p][0]:zone[p][2]]
            w, h = image.shape
            # image_show('image', image)
            if scale > 2:
                w1 = 33
            else:
                w1 = 19
            th = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, w1,
                                       3)  # 自适应均值阈值并黑白反转
            th1 = th[zone[p][1]:zone[p][3], zone[p][0]:zone[p][2]]
            if zone[p][0] > scale * 400 or zone[p][0] < scale * 100 or zone[p][1] > scale * 400 or zone[p][
                1] < scale * 100:
                thres = 88
                threshold = 80
            else:
                thres = 93
                threshold = 85
            ret, th2 = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
            # image_show('th', th)
            image1 = self.transform_form(th1, 'opening', scale)
            image2 = self.transform_form(th2, 'opening', scale)
            # show('open', image1)
            # image_show('guding', image2)

            partarea1, partgray1, partarea2, partgray2 = 0, 0, 0, 0
            for x in range(w):
                for y in range(h):
                    ys = zone[p][0] + y
                    xs = zone[p][1] + x
                    if image1[x][y] == 255 and 160 > image[x][y] > 20:
                        if 0 < ys < 43 or 72 < ys < 152 or 169 < ys < 253 or 268 < ys < 355 or 372 < ys < 453 or 476 < ys < 520:
                            if k1 * (xs - xd1) + yd1 <= ys <= k5 * (xs - xd5) + yd5 and k2 * (
                                    xs - xd2) + yd2 <= ys <= k6 * (xs - xd6) + yd6 and k3 * (
                                    xs - xd3) + yd3 <= ys <= k7 * (xs - xd7) + yd7 and k8 * (
                                    ys - yd8) + xd8 <= xs <= k4 * (ys - yd4) + xd4:
                                partarea1 += 1
                                partgray1 += image[x][y]
            partagray = partgray1 / (partarea1 + 1)
            for x2 in range(w):
                for y2 in range(h):
                    ys = zone[p][0] + y2
                    xs = zone[p][1] + x2
                    if image2[x2][y2] == 255 and 160 > image[x2][y2] > 20:
                        if 0 < ys < 43 or 72 < ys < 152 or 169 < ys < 253 or 268 < ys < 355 or 372 < ys < 453 or 476 < ys < 520:
                            if k1 * (xs - xd1) + yd1 <= ys <= k5 * (xs - xd5) + yd5 and k2 * (
                                    xs - xd2) + yd2 <= ys <= k6 * (xs - xd6) + yd6 and k3 * (
                                    xs - xd3) + yd3 <= ys <= k7 * (xs - xd7) + yd7 and k8 * (
                                    ys - yd8) + xd8 <= xs <= k4 * (ys - yd4) + xd4:
                                partarea2 += 1
                                partgray2 += image[x2][y2]
            partagray2 = partgray2 / (partarea2 + 1)
            if partarea2 - partarea1 > scale * 150:
                partarea1 = (2 * partarea2 + partarea1) / 3
                partagray = partagray2

            print(partagray, partarea1)
            if partagray > thres and partarea1 < scale * 210:
                qiandian += 1
            elif partagray > thres + 15 and partarea1 > scale * 210:
                qianban += 1
            elif partagray < thres and partarea1 < scale * 210 and partarea1 > scale * 40:
                shendian += 1
            elif partagray < thres and partarea1 < scale * 210 and partarea1 < scale * 40:
                qiandian += 1
            elif partagray < thres + 15 and partarea1 > scale * 210:
                shenban += 1
            if partarea1 > scale * 350:
                shenban += 1
            if 150 < partarea1 < 210 and partagray < thres + 30:
                median += 1
                median = int(median / 2)
        if qiandian > 7 or qianban + shendian + median >= 2 or shenban > 0 or len(zone) > 7:
            print("浅点", qiandian, '浅斑', qianban, '深斑', shenban, '深点', shendian)
            return '黑斑黑点'
        else:
            return '未知'

    def yinlie(self, zone):
        '''

        :param zone:  [x,y,x,y,p,w,h]
        :return:
        '''
        final = []
        for k in zone:
            total_center = [k[6] / 2, k[5] / 2]
            center = [(k[2] + k[0]) / 2, (k[3] + k[1]) / 2]
            dis = np.sqrt((total_center[0] - center[0]) ** 2 + (total_center[1] - center[1]) ** 2)  # 欧式距离
            x1, y1, x2, y2, x3, y3 = total_center[0], total_center[1], k[6] / 2, 0, center[0], center[1]

            # 计算三条边长
            a = math.sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3))
            b = math.sqrt((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3))
            c = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

            # 利用余弦定理计算三个角的角度
            A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
            # B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
            # C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
            # 输出三个角的角度
            # print("There three angles are", round(A, 2), round(B, 2), round(C, 2))
            if (42 < A < 48 or 132 < A < 138) and dis > (k[5] * 0.5 + 40):  # 角度限定,距离限定
                if k[4] > 0.82:
                    # if k[4] > 0.82:  # 概率限定
                    final.append(k)
                else:
                    continue
            else:
                final.append(k)
        return final

class histogram(object):
    def histogram(self, dst, clahe_dir, file):
        """
        区域直方图均衡
        :param dst: 原始图片路径
        :param clahe_dir: 处理后的图片保存目录
        :return: 处理后的图片路径
        """
        img = cv2.imread(dst, 0)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(25, 25))
        cl1 = clahe.apply(img)
        clahe_el_path = os.path.join(clahe_dir, file)
        cv2.imwrite(clahe_el_path, cl1)  # 把bmp\jpg格式经过处理后都转化成jpg格式
        return clahe_el_path

def videow(img_vid_path):
    '''
    opencv_机器学习-图片合成视频
    实现步骤:
    1.加载视频
    2.读取视频的Info信息
    3.通过parse方法完成数据的解析拿到单帧视频
    4.imshow，imwrite展示和保存
    5.本地保存格式：image+数字。jpg        以第一张size为模板
    '''
    img = cv2.imread(os.path.join(img_vid_path, 'vehicle_0000385.jpg'))
    # 获取当前图片的信息
    imgInfo = img.shape
    size = (imgInfo[1], imgInfo[0])
    # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
    videowrite = cv2.VideoWriter(os.path.join(img_vid_path + 'test.mp4'), -1, 30, size)
    for i in range(385, 415):
        fileName = 'vehicle_0000' + str(i) + '.jpg'
        img = cv2.imread(os.path.join(img_vid_path, fileName))
        # 写入参数，参数是图片编码之前的数据
        videowrite.write(img)
    print('end!')
def video2img(videopath,gap=25):
    cap = cv2.VideoCapture(videopath)  # 文件名及格式
    count = 0
    index = 1
    file = videopath.split(os.sep)[-1]
    dir = videopath.replace(file,'')
    filename = file.split('.')[-2]
    # os.makedirs(f'{dir}/{filename}', exist_ok=True)
    ret = True
    while ret:
        # capture frame-by-frame
        ret, frame = cap.read()
        if count % gap == 0:
            # namenum =str(count).zfill(6)
            # savepath = f'{dir}/{filename}/{filename}_{count}.jpg'
            os.makedirs('/home/linwang/pyproject/yolov5-5.0/ourdatas/testset/img/1200/' + f'{filename}/', exist_ok=True)
            savepath = '/home/linwang/pyproject/yolov5-5.0/ourdatas/testset/img/1200/' + f'{filename}/{filename}_{count}.jpg'
            cv2.imwrite(savepath,frame)
            index+=1
        count += 1
        # if index > 10:
        #     break
    # when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()
"""图片展示类"""

def image_show(barname, image):
    """
    :param barname: 图片显示的名称
    :param image: 输入
    :return:
    """
    cv2.namedWindow(barname, cv2.WINDOW_NORMAL)
    cv2.imshow(barname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def json_write(img_path,xml_path,label_path):
    annots = [os.path.join(xml_path, s) for s in os.listdir(xml_path)]  # 训练样本的xml路径
    json_list = []
    for annot in annots:
        """依次解析XML文件"""
        filename = annot.split('/')[-1]
        x = filename.split('.')[-1]
        filename_img = filename.replace(f'.{x}', '')
        #filename_img = filename.split('.')[-2]
        file_path = img_path + '/' + filename_img +'.jpg'
        print(file_path)
        et = ET.parse(annot)
        element = et.getroot()
        element_objs = element.findall('object')
        target_list = []
        for element_obj in element_objs:
            reframe = []
            #class_name = element_obj.find('name').text
            #if class_name == '小车':
            #    first = 0
            #else:
            #    first = 1
            reframe.append(int(element_obj.find('bndbox').find('xmin').text))
            reframe.append(int(element_obj.find('bndbox').find('ymin').text))
            reframe.append(int(element_obj.find('bndbox').find('xmax').text))
            reframe.append(int(element_obj.find('bndbox').find('ymax').text))
            trframe = xyxy2xywh(reframe,0)
            target_list.append(trframe)
        annot_dict = [('input', file_path), ('target', target_list)]
        dict1 = dict(annot_dict)
        json_list.append(dict1)
    #json1 = json.dumps(json_list)
    #print(json1)
    #with open('D:/pythonprojects/TrainSet/detection/v2.3train_test/label_test.json', 'w') as f:
    #    json.dump(json1, f)
    with open(label_path,'w',encoding='utf-8') as f:
        for sample in json_list:
            f.write(json.dumps(sample,ensure_ascii=False) + "\n")

def json_write_bdd_old(img_path,json_old,label_path):
    annots = [os.path.join(json_old, s) for s in os.listdir(json_old)]  # 训练样本的xml路径
    json_list = []
    for annot in annots:
        """依次解析json文件"""
        filename = annot.split('/')[-1]
        filename_img = filename.split('.')[-2]
        file_path = img_path + '/' + filename_img +'.jpg'
        print('annot', annot , 'file_path',file_path)
        bboxs_bdd = json_bdd(annot)
        #print(bboxs_bdd)
        if bboxs_bdd:
            annot_dict = [('input', file_path), ('target', bboxs_bdd)]
            dict1 = dict(annot_dict)
            json_list.append(dict1)
    with open(label_path, 'w', encoding='utf-8') as f:
            for sample in json_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
categorys = ['car', 'bus', 'truck']

def json_write_bdd(img_path,img_path_new, json_old, label_path):
    '''
    name -=[['daytime', 'night', 'dawn/dusk', 'undefined']
    :param img_path:
    :param json_old:
    :param label_path:
    :return:
    '''
    annots = [os.path.join(json_old, s) for s in os.listdir(json_old)]  # 训练样本的xml路径
    json_daytime_list = []
    json_night_list = []
    json_dawn_list = []
    for annot in annots:
        """依次解析json文件"""
        filename = annot.split('/')[-1]
        filename_img = filename.split('.')[-2]
        file_path = img_path + '/' + filename_img + '.jpg'
        #print('annot', annot, 'file_path', file_path)
        bboxs_bdd,name = json_bdd(annot)
        # print(bboxs_bdd)
        if bboxs_bdd:
            annot_dict = [('input', file_path), ('target', bboxs_bdd)]
            dict1 = dict(annot_dict)
        if name == 'daytime' or name == 'undefined':
            json_daytime_list.append(dict1)
            newfile_path = img_path_new + '/' + 'daytime' + '/' + filename_img + '.jpg'
            shutil.copyfile(file_path,newfile_path)
        elif name == 'night':
            json_night_list.append(dict1)
            newfile_path = img_path_new + '/' + 'night' + '/' + filename_img + '.jpg'
            shutil.copyfile(file_path, newfile_path)
        else:
            json_dawn_list.append(dict1)
            newfile_path = img_path_new + '/' + 'dawn' + '/' + filename_img + '.jpg'
            shutil.copyfile(file_path, newfile_path)
    with open(label_path + '/' + 'label_bdd_daytime.json', 'w', encoding='utf-8') as f:
        for sample in json_daytime_list:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(label_path + '/' + 'label_bdd_night.json', 'w', encoding='utf-8') as f2:
        for sample in json_night_list:
            f2.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(label_path + '/' + 'label_bdd_dawn.json', 'w', encoding='utf-8') as f3:
        for sample in json_dawn_list:
            f3.write(json.dumps(sample, ensure_ascii=False) + "\n")

def json_bdd(jsonFile):
    '''
      params:
        jsonFile -- BDD00K数据集的一个json标签文件
      return:
        返回一个列表的列表，，
        形如：[[325, 342, 376, 384, 'car'], [245, 333, 336, 389, 'car']]
    '''
    objs = []
    obj = []
    f = open(jsonFile)
    info = json.load(f)
    objects = info['frames'][0]['objects']
    categorys = ['car', 'bus', 'truck']
    name = info['attributes']['timeofday']
    for i in objects:
        if(i['category'] in categorys):
            obj.append(int(i['box2d']['x1']))
            obj.append(int(i['box2d']['y1']))
            obj.append(int(i['box2d']['x2']))
            obj.append(int(i['box2d']['y2']))
            obj_tran = trans(obj , 0)
            objs.append(obj_tran)
            obj = []
    #print("objs",objs)
    return objs,name

def check_json(json_label):
    filelist = []
    bboxlist = []
    errorlist = []
    f = open(json_label,'r')
    for i,line in enumerate(f):
        info = json.loads(line)
        # import pdb
        # pdb.set_trace()
        if isinstance(info, str):
            info = eval(info)
        if os.path.isfile(info['input']):
            name = info['input'].split('/')[-1]
            img = cv2.imread(info['input'])
            for sample in info['target']:
                cv2.rectangle(img,(int(sample[0]),int(sample[1])),(int(sample[2]),int(sample[3])),(255,0,0),1)
            cv2.imwrite("/home/linwang/pyproject/yolov5-5.0/runs/detect/gt_out/"+name,img)
        else:
            raise ValueError("is not file")



    '''
    objects = info['frames'][0]['objects']
    categorys = ['car', 'bus', 'truck']
    for i in objects:
        if (i['category'] in categorys):
            obj.append(int(i['box2d']['x1']))
            obj.append(int(i['box2d']['y1']))
            obj.append(int(i['box2d']['x2']))
            obj.append(int(i['box2d']['y2']))
            obj_tran = trans(obj, 0)
            objs.append(obj_tran)
            obj = []
    # print("objs",objs)
    '''

def bddjson_yolo5_bm(labelpath = ''):
    parent = str(Path(labelpath).parent) + os.sep
    # labelpaths = "/home/linwang/pyproject/yolo5s_bdd100k/bdd100k/labels/valids/fe1f55fa-19ba3600.txt"
    labelpaths = glob.glob(labelpath+'*.txt')
    newlabel = '/home/linwang/pyproject/yolo5s_bdd100k/bdd100k/labels/newlabel.json'
    # imgpath = "/home/linwang/pyproject/yolo5s_bdd100k/bdd100k/images/valids/fe1f55fa-19ba3600.jpg"
    json_list = []
    for i,labelpath in tqdm.tqdm(enumerate(labelpaths)):
        imgpath = labelpath.replace('labels','images')
        imgpath = imgpath.replace('txt', 'jpg')
        if os.path.isfile(imgpath):
            if i < 2:
                img = cv2.imread(imgpath)
                w,h,_ = img.shape
            with open(labelpath, 'r') as t:
                target = []
                t = t.read().splitlines()
                for sample in t:
                    sample = sample.split(' ')
                    sample = np.array(list(map(float, sample)))
                    # import pdb
                    # pdb.set_trace()
                    sample[1:5] = sample[1:5]*[h,w,h,w]
                    sample = xywh2xyxy(sample[1:5],sample[0])
                    target.append(sample)

            annot_dict = [('input', imgpath), ('target', target)]
            dict1 = dict(annot_dict)
            json_list.append(dict1)
                    # cv2.rectangle(img,(sample[0],sample[1]),(sample[2],sample[3]),(0,0,255),2)
                # cv2.imwrite('/home/linwang/pyproject/yolov5s_bdd100k_backup/yolov5s_bdd100k-master/inference/debug.jpg',img)
        else:
            continue
    with open(newlabel, 'w', encoding='utf-8') as f:
            for sample in json_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def xyxy2xywh(frame,class_target):
    trframe = []
    trframe.append(class_target)
    trframe.append(int((frame[0]+frame[2])/2))
    trframe.append(int((frame[1]+frame[3])/2))
    trframe.append(int(frame[2]-frame[0]))
    trframe.append(int(frame[3]-frame[1]))
    #print(frame,trframe)
    return trframe

def xywh2xyxy(frame,class_target):
    trframe = []
    trframe.append(int(frame[0]-(frame[2]/2)))
    trframe.append(int(frame[1]-(frame[3]/2)))
    trframe.append(int(frame[0]+(frame[2]/2)))
    trframe.append(int(frame[1]+(frame[3]/2)))
    trframe.append(class_target)
    #print(frame,trframe)
    return trframe

def load_data(datapath):
        dataset = []
        with open(datapath, "r") as fid:
            for line in fid:
                sample = json.loads(line)
                dataset.append(sample)
                #print(type(sample))
        return dataset

def ua_data(inpath,imgdir,outpath,jsonpathc,jsonpathn,jsonpathr,jsonpaths):
    '''

    :param inpath: xml
    :imgpath :img
    :param outpath:  xml or json savepath
    :return:
    '''
    annots = [os.path.join(inpath, s) for s in os.listdir(inpath)]  # 训练样本的xml路径
    outimgnum = 1
    json_list_sunny = []
    json_list_rainy = []
    json_list_night = []
    json_list_cloudy = []
    for annot in annots:
        """依次解析XML文件"""
        et = ET.parse(annot)
        element = et.getroot()
        filename = annot.split('/')[-1]
        imgdirname = filename.replace('.xml','')
        #element_objs = element.findall('target')
        #tag = element.tag
        #attrib = element.attrib['name']  # zheshi tupianwenjianjia he xml mingzi
        #value = element.text
        idx = 1
        for child in element:
            bbox_list = []
            #print('1', child.attrib)
            for sec_child in child:
                #print('2', sec_child.attrib)
                for tri_child in sec_child:
                    #print('3' ,tri_child.tag, tri_child.attrib)
                    for four_child in tri_child:
                        if four_child.tag == 'box':
                            bbox = [0]
                            bbox.append(
                                int(float(four_child.attrib['left'])) + int(float(four_child.attrib['width']) / 2))
                            bbox.append(
                                int(float(four_child.attrib['top'])) + int(float(four_child.attrib['height']) / 2))
                            bbox.append(int(float(four_child.attrib['width'])))
                            bbox.append(int(float(four_child.attrib['height'])))
                            bbox_list.append(bbox)
                            #print('4', four_child.tag, bbox)
            if idx < 2:
                weather = child.attrib.get('sence_weather')
                idx += 1
                out_img_path = outpath + '/' + weather
                os.makedirs(out_img_path,exist_ok=True)
            print(bbox_list)
            imgpath = imgdir + '/' + imgdirname + '/' +'img' + str((child.attrib.get('num'))).zfill(5) + '.jpg'
            save_path = outpath + '/'  + weather + '/' + 'img'  + str(outimgnum).zfill(8) + '.jpg'
            outimgnum += 1
            print(imgpath , save_path)
            if bbox_list and outimgnum % 10 == 0:
                shutil.copyfile(imgpath,save_path)
                annot_dict = [('input', save_path), ('target', bbox_list)]
                dict1 = dict(annot_dict)
                if weather == 'cloudy':
                    json_list_cloudy.append(dict1)
                elif weather == 'night':
                    json_list_night.append(dict1)
                elif weather == 'rainy':
                    json_list_rainy.append(dict1)
                else:
                    json_list_sunny.append(dict1)

    with open(jsonpathc, 'w', encoding='utf-8') as f:
            for sample in json_list_cloudy:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonpathn, 'w', encoding='utf-8') as f:
        for sample in json_list_night:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonpathr, 'w', encoding='utf-8') as f:
        for sample in json_list_rainy:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(jsonpaths, 'w', encoding='utf-8') as f:
        for sample in json_list_sunny:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def ua_data_train(inpath,imgdir,outpath,jsonpath):
    '''

    :param inpath: xml
    :imgpath :img
    :param outpath:  xml or json savepath
    :return:
    '''
    annots = [os.path.join(inpath, s) for s in os.listdir(inpath)]  # 训练样本的xml路径
    outimgnum = 1
    json_list = []
    json_list_rainy = []
    json_list_night = []
    json_list_cloudy = []
    for annot in annots:
        """依次解析XML文件"""
        et = ET.parse(annot)
        element = et.getroot()
        filename = annot.split('/')[-1]
        imgdirname = filename.replace('.xml','')
        #element_objs = element.findall('target')
        #tag = element.tag
        #attrib = element.attrib['name']  # zheshi tupianwenjianjia he xml mingzi
        #value = element.text
        idx = 1
        for child in element:
            bbox_list = []
            #print('1', child.attrib)
            for sec_child in child:
                #print('2', sec_child.attrib)
                for tri_child in sec_child:
                    #print('3' ,tri_child.tag, tri_child.attrib)
                    for four_child in tri_child:
                        if four_child.tag == 'box':
                            bbox = [0]
                            bbox.append(
                                int(float(four_child.attrib['left'])) + int(float(four_child.attrib['width']) / 2))
                            bbox.append(
                                int(float(four_child.attrib['top'])) + int(float(four_child.attrib['height']) / 2))
                            bbox.append(int(float(four_child.attrib['width'])))
                            bbox.append(int(float(four_child.attrib['height'])))
                            bbox_list.append(bbox)
                            #print('4', four_child.tag, bbox)
            if idx < 2:
                weather = child.attrib.get('sence_weather')
                idx += 1
                out_img_path = outpath + '/' + weather
                os.makedirs(out_img_path,exist_ok=True)
            print(bbox_list)
            imgpath = imgdir + '/' + imgdirname + '/' +'img' + str((child.attrib.get('num'))).zfill(5) + '.jpg'
            save_path = outpath + '/'  + weather + '/' + 'img'  + str(outimgnum).zfill(8) + '.jpg'
            outimgnum += 1
            print(imgpath , save_path)
            if bbox_list and outimgnum % 10 == 0:
                shutil.copyfile(imgpath,save_path)
                annot_dict = [('input', save_path), ('target', bbox_list)]
                dict1 = dict(annot_dict)
                json_list.append(dict1)

    with open(jsonpath, 'w', encoding='utf-8') as f:
        for sample in json_list:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def iou_xy(Reframe,GTframe):
    #print('正在解析 annotation files')
    # 得到第一个矩形的左上坐标及宽和高
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    # 得到第二个矩形的左上坐标及宽和高
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    # 计算重叠部分的宽和高
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    # 如果重叠部分为负, 即不重叠
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)

    return ratio

def iou_xywh(Reframe,GTframe):
    # print('正在解析 annotation files')
    # 得到第一个矩形的左上坐标及宽和高
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]
    height1 = Reframe[3]

    # 得到第二个矩形的左上坐标及宽和高
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]
    height2 = GTframe[3]

    # 计算重叠部分的宽和高
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    # 如果重叠部分为负, 即不重叠
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio

def clear_box_pred(pred,targets):
    pre_move = []
    tar_move = []
    c1,c2= [],[]
    for i, sample in enumerate(pred):
        if (sample[4]*sample[5]) < 2500 or sample[4]<50 or sample[5]<40:
            pre_move.append(i)
    for j, sample1 in enumerate(targets):
        if (sample1[3] * sample1[4]) < 2500 or sample1[3]<50 or sample1[4]<40:
            tar_move.append(j)
    c1 = c1+pred
    for i in pre_move:
        c1.remove(pred[i])
    c2 = c2 + targets
    for j in tar_move:
        c2.remove(targets[j])
    #print('pred', c1,'\n','tar',c2)
    return c1,c2

def tp_list(preds,targets):
    tplist = []
    tp,fp = 0,0
    iou_th = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for _iou in iou_th:
        for predbox in preds:
            rats = []
            for tarbox in targets:
                    rat = iou_xywh(predbox[2:6], tarbox[1:5])
                    rats.append(rat)
                # print(max(rats))
            if max(rats) >= _iou:
                    tp += 1
            else:
                    fp += 1
        tplist.append(tp)
    return tplist

def ccpd_json_fromtxt(txtpath,imgpath,label_path,log=''):
    file = open(txtpath)
    json_list = []
    while 1:
        lines = file.readlines(240000)
        if not lines:
            break
        for line in lines:
            imgname = imgpath+'/'+line
            imgname = imgname.replace('\n','')
            target_list = []
            #print('start',imgname,'end')
            if os.path.isfile(imgname):
                targets = [0]
                errorup = imgname
                name = line.split('/')[-1]
                target = name.split('-')[3]
                point1, point2, point3, point4 = target.split('_')[0].split('&'), target.split('_')[1].split('&'), target.split('_')[2].split('&'), target.split('_')[3].split('&')
                #print(point1,point2,point3,point4)
                pointmax = [max(int(point1[0]),int(point4[0])),max(int(point1[1]),int(point2[1]))]
                pointmin = [min(int(point3[0]),int(point2[0])),min(int(point3[1]),int(point4[1]))]
                pointcenter = [int((pointmin[0]+pointmax[0])/2),int((pointmin[1]+pointmax[1])/2)]
                w,h = int(pointmax[0]-pointmin[0]),int(pointmax[1]-pointmin[1])
                targets.append(pointcenter[0])
                targets.append(pointcenter[1])
                targets.append(w)
                targets.append(h)
                target_list.append(targets)
                #print(pointcenter,w,h)
                annot_dict = [('input', imgname), ('target', target_list)]
                dict1 = dict(annot_dict)
                print(annot_dict)
                json_list.append(dict1)
        with open(label_path, 'w', encoding='utf-8') as f:
            for sample in json_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    file.close()
    return 'ok'

def ccpd_square(jsonpath,newpath,labelpath):
    json_list = []
    f = open(jsonpath)
    lines = f.readlines()
    print(len(lines))
    for i,line in enumerate(lines):
        info = json.loads(line)
        if os.path.isfile(info['input']):
            #print(info['input'])
            image = cv2.imread(info['input'])
            height, width, _ = image.shape
            if height > width:

                lenght = height - width
                print('height', lenght)
                a = cv2.copyMakeBorder(image, 0, 0, 0, lenght, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            elif width > height:
                lenght = height - width
                a = cv2.copyMakeBorder(image, 0, lenght, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            #image_show('a',a)
            filename = info['input'].split('/')[-1]
            savepath = os.path.join(newpath, filename)
            #cv2.imwrite(savepath,a)
            annot_dict = [('input', savepath), ('target', info['target'])]
            dict1 = dict(annot_dict)
            print(i)
            json_list.append(dict1)

    with open(labelpath, 'w', encoding='utf-8') as f:
        print(len(json_list))
        for sample in json_list:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def cutjson_ccpdtest(jsonpath,out):
    filelist = [[],[],[],[],[],[],[],[],[]]
    bboxlist = []
    errorlist = []
    words = ['ccpd_base','ccpd_blur','ccpd_challenge','ccpd_db','ccpd_fn','ccpd_np','ccpd_rotate','ccpd_tilt','ccpd_weather']
    for word in words:
        os.mknod(out + '/'+f"test_{word}.json")
    f = open(jsonpath)
    lines = f.readlines()
    for line in lines:
        boxnum = 0
        info = json.loads(line)
        if os.path.isfile(info['input']):
            for i,word in enumerate(words):
                if word in info['input']:
                    filelist[i].append(info)
    for x,ls in enumerate(filelist):
        with open(out+'/'+f'tese_{words[x]}.json', 'w', encoding='utf-8') as newf:
            for sample in ls:
                newf.write(json.dumps(sample, ensure_ascii=False) + "\n")

def ccpd_checkjson(jsonpath):
    f = open(jsonpath)
    lines = f.readlines()
    os.makedirs("ccpdcheck/", exist_ok=True)
    for i,line in enumerate(lines):
        boxnum = 0
        info = json.loads(line)
        image = cv2.imread(info['input'])
        file = info['input'].split('/')[-1]
        for target in info['target']:
            target = xywh2xxyy(target[1:],0)
            cv2.rectangle(image,(target[1],target[2]),(target[3],target[4]),(0,0,255),1)
        cv2.imwrite('ccpdcheck/'+file,image)

def ccpd_fromxml(imgpath,xmlpath,labelpath):
    '''

    :param imgpath: "/home/linwang/pyproject/TrainSet/detection/v32/error_ccpd1/"
    :param xmlpath: "/home/linwang/pyproject/TrainSet/detection/v32/error_ccpd1/annot/"
    :param jsonpath: "/home/linwang/pyproject/TrainSet/detection/v32/label_fix.json"
    :return:
    '''
    words = ['ccpd_base', 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_np', 'ccpd_rotate', 'ccpd_tilt',
             'ccpd_weather']
    json_list = []
    for word in words:
        _imgpath = imgpath + word
        imgs = [os.path.join(_imgpath, s) for s in os.listdir(_imgpath)]  # 训练样本的img路径
        for img in imgs:
            filename = img.split('/')[-1]
            x = filename.split('.')[-1]
            filename_img = filename.replace(f'.{x}', '')
            file = img.split('/')[-2] + '/' + img.split('/')[-1]
            # filename_img = filename.split('.')[-2]
            annot = xmlpath + filename_img + '.xml'
            et = ET.parse(annot)
            element = et.getroot()
            element_objs = element.findall('object')
            target_list = []
            for element_obj in element_objs:
                reframe = []
                # class_name = element_obj.find('name').text
                # if class_name == '小车':
                #    first = 0
                # else:
                #    first = 1
                reframe.append(int(element_obj.find('bndbox').find('xmin').text))
                reframe.append(int(element_obj.find('bndbox').find('ymin').text))
                reframe.append(int(element_obj.find('bndbox').find('xmax').text))
                reframe.append(int(element_obj.find('bndbox').find('ymax').text))
                trframe = xyxy2xywh(reframe, 0)
                target_list.append(trframe)
            annot_dict = [('input', file), ('target', target_list)]
            dict1 = dict(annot_dict)
            json_list.append(dict1)
    with open(labelpath,'w',encoding='utf-8') as f:
        for sample in json_list:
            f.write(json.dumps(sample,ensure_ascii=False) + "\n")

def ccpd_fixjson(fixjson,jsonpath):
    '''

    :param fixjson: "/home/linwang/pyproject/TrainSet/detection/v32/label_fix.json"
    : jsonpath /home/linwang/pyproject/TrainSet/detection/v32/
    :return:
    '''
    os.makedirs(jsonpath + 'test',exist_ok=True)
    words = ['train','val','test']
    #for word in words:
    #s    os.mknod(jsonpath + '/' + f"test_{word}.json")
    f = open(fixjson,'r')
    lines = f.readlines()
    for word in words:
        json_list = []
        _jsonpath = jsonpath +'/' + f'label_{word}.json'
        f_old = open(_jsonpath)
        lines_old = f_old.readlines()
        for line_old in tqdm.tqdm(lines_old):
            num = 0
            info_old = json.loads(line_old)
            file_old = info_old['input']
            filename_old = file_old.split('/')[-2] + '/' +file_old.split('/')[-1]
            for line in lines:
                info = json.loads(line)
                file = info['input']
                filename = file.split('/')[-2] + '/'+file.split('/')[-1]
                if filename == filename_old:
                    annot_dict = [('input', file_old), ('target', info['target'])]
                    dict1 = dict(annot_dict)
                    json_list.append(dict1)
                    num = 1
                    img = cv2.imread(file_old)
                    for box in info['target']:
                        box1 = xywh2xxyy(box[1:5],0)
                        cv2.rectangle(img,(box1[1],box1[2]),(box1[3],box1[4]),(255,0,0),2)
                    cv2.imwrite(jsonpath + 'test' +'/'+file_old.split('/')[-1],img)
            if num == 0:
                json_list.append(info_old)
        labelpath = jsonpath + '/' + f"test_{word}.json"
        with open(labelpath, 'w', encoding='utf-8') as f:
            for sample in json_list:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def ccpd_fromxml_addyellow(imgpath,xmlpath,labelpath):
    '''

    :param imgpath: "/home/linwang/pyproject/TrainSet/detection/v32/error_ccpd1/"
    :param xmlpath: "/home/linwang/pyproject/TrainSet/detection/v32/error_ccpd1/annot/"
    :param jsonpath: "/home/linwang/pyproject/TrainSet/detection/v32/label_fix.json"
    :return:
    '''
    annots = []
    json_list = []
    for s in os.listdir(xmlpath):
        if s[-3:] == 'xml':
            annots.append(os.path.join(xmlpath, s))# 训练样本的xml路径
    for annot in annots:
        """依次解析XML文件"""
        filename = annot.split('/')[-1]
        x = filename.split('.')[-1]
        filename_img = filename.replace(f'.{x}', '')
        # filename_img = filename.split('.')[-2]
        file_path = imgpath + '/' + filename_img + '.jpg'
        print(file_path)
        if os.path.isfile(file_path):
            et = ET.parse(annot)
            element = et.getroot()
            element_objs = element.findall('object')
            target_list = []
            for element_obj in element_objs:
                reframe = []
                reframe.append(int(element_obj.find('bndbox').find('xmin').text))
                reframe.append(int(element_obj.find('bndbox').find('ymin').text))
                reframe.append(int(element_obj.find('bndbox').find('xmax').text))
                reframe.append(int(element_obj.find('bndbox').find('ymax').text))
                trframe = xyxy2xywh(reframe, 0)
                target_list.append(trframe)
            annot_dict = [('input', file_path), ('target', target_list)]
            dict1 = dict(annot_dict)
            json_list.append(dict1)
    with open(labelpath, 'w', encoding='utf-8') as f:
        for sample in json_list:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def ccpd_addyel_jsoncut(jsonpath,out_blue,out_yel,out_gre,out_wh,out_other):
    f = open(jsonpath, 'r+', encoding='UTF-8')
    lines = f.readlines()
    jsonlist_blue = []
    jsonlist_yel = []
    jsonlist_gre = []
    jsonlist_wh = []
    jsonlist_other = []
    for line in lines:
        info = json.loads(line)
        x = info['input']
        #print(x,x[-3:])
        if x[-3:] == 'jpg':
            x = x.replace('.jpg', '')
            x = x.split('_')[-1]
            if x == '0':
                jsonlist_blue.append(info)
            elif x == '1':
                jsonlist_yel.append(info)
            elif x == '2':
                jsonlist_gre.append(info)
            elif x == '4':
                jsonlist_wh.append(info)
            else:
                jsonlist_other.append(info)
    with open(out_blue, 'w', encoding='utf-8') as newf:
        for sample in jsonlist_blue:
            newf.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(out_yel, 'w', encoding='utf-8') as newf:
        for sample in jsonlist_yel:
            newf.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(out_gre, 'w', encoding='utf-8') as newf:
        for sample in jsonlist_gre:
            newf.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(out_wh, 'w', encoding='utf-8') as newf:
        for sample in jsonlist_wh:
            newf.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(out_other, 'w', encoding='utf-8') as newf:
        for sample in jsonlist_other:
            newf.write(json.dumps(sample, ensure_ascii=False) + "\n")

def padding(image):
    height, width, _ = image.shape
    if height > width:
        lenght = height - width
        image = cv2.copyMakeBorder(image, 0, 0, 0, lenght, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif width > height:
        lenght = width - height
        image = cv2.copyMakeBorder(image, 0, lenght, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image

def test_img(jsondir):
    words = ['blue','yel','gre','wh']
    num = 0
    for word in words:
        os.makedirs(jsondir + f'/{word}', exist_ok=True)
        jsonpath = jsondir + f'/test_{word}.json'
        f = open(jsonpath, 'r+', encoding='UTF-8')
        lines = f.readlines()
        for line in lines:
            info = json.loads(line)
            if os.path.isfile(info['input']):
                num += 1
                img = cv2.imread(info['input'])
                img = padding(img)
                imgpath = jsondir + f'/{word}/' + f'{num}.jpg'
                cv2.imwrite(imgpath,img)
    return 0

def cut_testimg(dir):
    files = walkfile(dir)
    for file in files:
        video2img(file)

def walkfile(file):
    filelist = []
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            filelist.append(os.path.join(root, f))

        # 遍历所有的文件夹
        # for d in dirs:
        #     print(os.path.join(root, d))
    return filelist
def transcoding():
    # mkdir zip
    # for i in $(ls./ *.mp4); do (ffmpeg -i $i -vf scale=1280:-1 - c: v libx264 - preset veryslow - crf 24 zip /$i); done
    #
    # for i in $(ls./ *.mp4); do echo $i
    pass
def debug(image):
    import pdb
    pdb.set_trace()

def json2xml():
    with open(file) as f:
        pass

#     writexml = writexml()
#     lines = f.readlines()
#     for line in lines :
#         line = line.strip('\n')
#         info = json.loads(line)
#         gz = []
#         name = info['input'].split(os.sep)[-1].replace('.jpg','')
#         h,w = 1920,1080
#         gz.append(name)
#         gz.append(h)
#         gz.append(w)
#         gz.append('1')
#         for sample in info['target']:
#             new = []
#             new.append(sample[5])
#             new.append(sample[0])
#             new.append(sample[1])
#             new.append(sample[2])
#             new.append(sample[3])
#             gz.append(new)
#         if '1200W' in info['input']:
#             os.makedirs(outxml + '1200w/', exist_ok=True)
#             writexml.writetoxml(gz,outxml + '1200w/')
#         else:
#             os.makedirs(outxml + '200w/', exist_ok=True)
#             writexml.writetoxml(gz, outxml + '200w/')
if __name__ == '__main__':
    # dir = "/home/linwang/pyproject/yolov5-5.0/ourdatas/testgt/img/"
    # file = "/home/linwang/pyproject/yolov5-5.0/ourdatas/testgt/label/img.json"
    # outxml = "/home/linwang/pyproject/yolov5-5.0/ourdatas/testgt/annots/"
    #
    jsonp = "/home/linwang/pyproject/yolov5-5.0/merged_label.jso"




