import os
# os.environ["CUDA_DEVICE_ORDER"]="0"
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='flow', type=str, help='rgb or flow')
parser.add_argument('-load_model', default="models/flow_imagenet.pt", type=str)
parser.add_argument('-root', default='D:/code/flownet2-pytorch-master/work/inference/',
                    type=str)  # rgb C:/Users/CWJ/Desktop/dataset/videos/
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', default='output/flow/', type=str)

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d
from charades_dataset_full import Charades as Dataset



# rgb C:/Users/CWJ/Desktop/dataset/videos/
def run(max_steps=64e3, mode='flow', root='D:/code/flownet2-pytorch-master/work/inference/', split='train_anno.json', batch_size=16,
        load_model='models/flow_imagenet.pt', save_dir='output/flow/'):
    # setup dataset

    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # 遍历每个视频文件夹

    # 创建 Dataset 实例
    dataset = Dataset(split, 'train',root, mode,test_transforms, num=-1, save_dir=save_dir)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                                             pin_memory=False)

    # val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
    #                                              pin_memory=True)

    dataloaders = {'train': dataloader}
    datasets = {'train': dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(400)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    for phase in ['train']:
        i3d.train(False)  # Set model to evaluate mode

        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0

        # Iterate over data.
        for data in dataloaders[phase]:
            inputs, labels, name = data

            # if os.path.exists(os.path.join(save_dir, name[0] + '.npy')):
            #     continue

            b, c, t, h, w = inputs.shape  # input [batch, channel, t 图片张数, h ,w]

            if t > 80:
                print(t)
                with torch.no_grad():
                    features = []
                    for start in range(1, t - 56, 1600):
                        end = min(t - 1, start + 1600 + 56)
                        start = max(1, start - 48)
                        ip = Variable(torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda())
                        features.append(i3d.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
                    np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                with torch.no_grad():
                    inputs = Variable(inputs.cuda())
                    features = i3d.extract_features(inputs)
                    data = features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()
                    np.save(os.path.join(save_dir, name[0]),
                            features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
