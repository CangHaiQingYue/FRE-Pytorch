import argparse
import torch
import cv2
import os
import numpy as np

from torch.utils.data import DataLoader
from data_loader import Data_batch
from fre_model import FRE

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='/home/myWork/myPytorch/FRE/save/0.pth',
                    type=str)
parser.add_argument('--ls_path', default='testing', type=str,
                    help='chose which .lst file')
parser.add_argument('--result_path', default='/home/myWork/myPytorch/FRE/result',
                    type=str, help='where edge maps saved in')
args = parser.parse_args()

########### get filename for cv2.imwrite()
base_path = '/home/myWork/DataSets/HED-BSDS'
if args.ls_path == 'test2':
    path = 'test2.lst'
else:
    path = 'test.lst'
path = os.path.join(base_path, path)
with open(path) as f:
    filename = f.readlines()
filename = [k.split(' ') for k in filename]

#############configure dataset
dataset = Data_batch(args.ls_path)
data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)
iterator = iter(data_loader)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
else:
    raise RuntimeError

################get network
net = FRE()
net.to(device)
net.eval()



net.load_state_dict(torch.load(args.model_path))

k = 0
for image in data_loader:
    image = image.to(device)
    # print(image.size())

    outputs = net(image)

    for i, map in enumerate(outputs):
        map = map.permute(0,2,3,1)
        map = torch.sigmoid(map)
        if i == 5:
            path = filename[k][0][5:-5] + '.png'
            path = os.path.join(args.result_path, path)
            edge = np.uint8(255*map[0].cpu().detach().numpy())
            # edge = 255*map[0].cpu().detach().numpy()
            # print(np.max(edge))
            h, w, c = np.shape(edge)
            if h > w:
                edge = cv2.resize(edge, (321, 481))
            else:
                edge = cv2.resize(edge, (481,321))

            print(path)
            # cv2.imshow('win', edge)
            # cv2.waitKey()
            cv2.imwrite(path, edge)
    k = k + 1
