import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(os.path.join(image_dir, vid+'/','output_'+str(i).zfill(4)+'.png'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        # else:
        # # 如果图像已经大于等于目标尺寸，则不需要缩放
        # sc = min(226 / w, 226 / h)
        # img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
    #img = (img/255.)*2 - 1
    img = cv2.resize(img, dsize=(224,224))
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)



def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    # print(os.path.join(image_dir, vid+'/', str(i)+'_x.jpg'))
    imgx = cv2.imread(os.path.join(image_dir, vid+'/', str(i)+'_x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid+'/', str(i)+'_x.jpg'), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    # imgx = (imgx/255.)*2 - 1
    # imgy = (imgy/255.)*2 - 1
    imgx = cv2.resize(imgx, dsize=(224, 224))
    imgy = cv2.resize(imgy, dsize=(224,224))
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=10):
    dataset = []
    # 打开并读取split_file文件，该文件是一个包含视频信息的JSON文件
    """
    {"7UPGT": 
         {"subset":"training",
         "duration": 23.21, 
         "actions": [[149, 16.0, 22.2], [152, 16.0, 22.2]]}
    """
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0 # 计数器i，用来记录处理的视频数量
    for vid in data.keys(): # 遍历所有视频的ID（vid）
        # 检查当前视频是否属于所需的划分（train/val/test等），如果不匹配，跳过该视频

        if data[vid]['subset'] != split:
            continue
        # 检查当前视频目录是否存在，如果不存在，跳过该视频
        print(os.path.join(root, vid))
        if not os.path.exists(os.path.join(root, vid)):
            continue
        # 获取当前视频的帧数，num_frames 是视频中的帧数
        num_frames = len(os.listdir(os.path.join(root, vid)))
        # 如果模式是 'flow'，则视频帧数会除以2（因为光流视频通常有两个方向的帧：水平和垂直）
        if mode == 'flow':
            num_frames = num_frames//2

        # 初始化一个标签矩阵，大小为 (num_classes, num_frames)
        # 每一个动作类别对应一个标签，且每个帧都会有一个相应的标签
        label = np.zeros((num_classes,num_frames), np.float32)

        # 计算每帧的帧率（fps），即每秒有多少帧
        # fps = num_frames/data[vid]['duration']
        fps = 1
        # 遍历每个视频的动作信息
        action_info = data[vid]['annotations']
        time_info = action_info[0]['segment']

        txt = np.loadtxt('Class_Index.txt', dtype=str)
        originidx_to_idx = {}  # 用于将原始索引映射到新的索引
        idx_to_class = {}  # 用于将新的索引映射到类别名称
        # 遍历加载的文本内容，enumerate 生成索引 idx 以及行内容 l
        for idx, l in enumerate(txt):
            idx_to_class[l[1]] = idx + 1
            # 将原始索引（第一列）映射到新的索引（idx + 1）
            originidx_to_idx[int(l[0])] = idx + 1

        # for ann in action_info[0]['label']:
        # 遍历视频中的每一帧
        for fr in range(0,num_frames,1):
            #  检查该帧的时间戳是否在动作的起止时间范围内

            if fr/fps > time_info[0] and fr/fps < time_info[1]:
                # 如果在动作时间段内，为对应的动作类别和帧设置标签为1（即binary classification，表示该帧属于该动作类别）
                label[idx_to_class[action_info[0]['label']], fr] = 1 # binary classification
        # 将视频ID、标签和帧数添加到数据集中
        dataset.append((vid, label, num_frames))


        i += 1
    
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf = self.data[index]
        if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
            return 0, 0, vid

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, 1, nf)
            # print("==========================")
            # print("加载rgb帧完毕！")
            # print("数据类型：",type(imgs))
            # print("数据元素总数：",imgs.size)
            # print("数据形状：",imgs.shape)
            # print("数据维度：",imgs.ndim)
            # print("==========================")
        if self.mode == 'flow':
            imgs = load_flow_frames(self.root, vid, 0, nf)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)
