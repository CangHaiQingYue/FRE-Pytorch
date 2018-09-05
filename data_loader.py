import torch
import cv2
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

base_path = {'training' : '/home/liupengli/myWork/DataSets/rcf',
             'test2' : '/home/liupengli/myWork/DataSets/HED-BSDS',
             'test1' : '/home/liupengli/myWork/DataSets/HED-BSDS',
             'testing' : '/home/liupengli/myWork/DataSets/HED-BSDS'
            }
data_path = {'training' : 'bsds_pascal_train_pair.lst',
             'test2' : '/home/liupengli/myWork/DataSets/HED-BSDS/test2.lst',
             'test1' : '/home/liupengli/myWork/DataSets/HED-BSDS/test1.lst',
             'testing' : '/home/liupengli/myWork/DataSets/HED-BSDS/test.lst'
            }
transform = transforms.Compose(
    [
    transforms.ToTensor()])
mean_pixel_value = [104.00699, 116.66877, 122.67892]
class Data_batch(Dataset):
    def __init__(self, model_state='training'):
        super().__init__()
        self.model_state = model_state
        if model_state == 'training':
            path = os.path.join(base_path[model_state], data_path[model_state])
            com_path = self.read_path(path)
            self.filename = [v.split(' ') for v in com_path]
        elif model_state == 'test2':

            path = data_path[model_state]
            self.filename = self.read_path(path)
        elif model_state == 'testing':
            path = data_path[model_state]
            self.filename = self.read_path(path)
        elif model_state == 'test1':
            path = data_path[model_state]
            self.filename = self.read_path(path)
        else:
            raise ValueError('Master -> Plase check your model_state')

    def __getitem__(self, index):
        if self.model_state == 'training':
            image_path = os.path.join(base_path[self.model_state],self.filename[index][0])
            label_path = os.path.join(base_path[self.model_state],self.filename[index][1])
            image = cv2.imread(image_path.strip()) / 1.0
            label = cv2.imread(label_path.strip())
            image = cv2.resize(image, (320,320))
            image -= mean_pixel_value

            label = cv2.resize(label, (320,320))
            label = label / 255.0
            label[label > 0.5] = 1.0

            image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float)
            # print(image.dtype)
            label = torch.from_numpy(label).permute(2, 0, 1).to(torch.float)
            # print(label_path)
            if  isinstance(image, torch.Tensor):
                #transform(label) convert [0,255] to [0,1]
                return image, label
            else:
                raise ValueError('Master -> Plase check your Image Path')
        else:
            image_path = os.path.join(base_path[self.model_state], self.filename[index])
            image = cv2.imread(image_path.strip()) / 1.0
            # image
            h, w, c = image.shape
            image = cv2.resize(image, (w-1, h-1))

            image -= mean_pixel_value
            image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float)
            if  isinstance(image, torch.Tensor):
                return image
            else:
                raise ValueError('Master -> Plase check your Image Path')

    def __len__(self):
        # print(len(self.filename))
        return len(self.filename)

    def read_path(self, path):
        with open(path, 'r') as f:
            return f.readlines()
#####    TEST DATA LOADER   ########
# dataset = data_batch(model_state='training')
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
# for img, label in train_loader:
#     print(img.size())
#     img = img.permute(0,2,3,1)
#     cv2.imshow('we', np.uint8(255*img.numpy()[0]))
#     cv2.waitKey()