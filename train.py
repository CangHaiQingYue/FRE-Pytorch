import utils
import torch
import argparse

from fre_model import FRE
from data_loader import Data_batch
from fre_loss import sigmoid_loss
from torch.utils.data import DataLoader 
from utils import DeviceError
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='The inputs arguments for FRE-model')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--fine_tune', default=False, type=str2bool,
                    help='if fine tune the model will load form trianed network')
parser.add_argument('--pth_path', default='/home/liupengli/myWork/myPytorch/FRE/save', 
                    type=str, help='path where trained .pth file saved in')
parser.add_argument('--vgg_pth_path', default='/home/liupengli/myWork/DataSets/vgg16.pth', 
                    type=str, help='path where vgg16.pth saved in')
##########################     optimizer     ##########################################
parser.add_argument('--lr', default=0.001, type=float, 
                    help='base learning rate')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='momentum for SGD')
parser.add_argument('--weight_decay', default=0.0002, type=float, 
                    help='weight_decay for weights')
#####################     path     ##########################################3
parser.add_argument('--save_path', default='/home/liupengli/myWork/myPytorch/FRE/save/', 
                    type=str, help='path where state_dict saved in after training')

parser.add_argument('--max_steps', default=12001, type=int,
                    help='max steps')
args = parser.parse_args()


# find GPU
if torch.cuda.is_available():

    
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
    # print(device)
else:
    raise DeviceError('Master: there must be at least one GPU!')


# get dataset
dataset = Data_batch('training')
train_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=args.batch_size)
train_iterator = iter(train_loader)
#initialzie network and parameters
net = FRE()
net.to(device)
net.train()
if args.fine_tune:
    net.load_state_dict(torch.load(args.pth_path))
else:
    # first, use pretrained vgg16.pth initialze backbone(vgg part)
    pre_trained = torch.load(args.vgg_pth_path)
    net = utils.init_vgg_parameters(net, pre_trained)
    # then, initialize side part
    net = utils.init_side_parameters(net)
    #well, fuse layer's initialization may not be very import!
    # just skip

#get loss
Loss = sigmoid_loss()
Loss.to(device)
# optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
#                             momentum=args.momentum,
#                             weight_decay=args.weight_decay)

LR = args.lr
# optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
optimizer = torch.optim.Adam([
          {'params': utils.get_params(net, scope='conv')},
          {'params': utils.get_params(net, scope='conv', bias=True), 'lr': LR*2},
          {'params': utils.get_params(net, scope='conv5'), 'lr': LR*100},
          {'params': utils.get_params(net, scope='conv5', bias=True), 'lr': LR*200},
          {'params': utils.get_params(net, scope='side'), 'lr': LR*0.1},
          {'params': utils.get_params(net, scope='side', bias=True), 'lr': LR*0.2},
          {'params': utils.get_params(net, scope='fuse'), 'lr': LR*0.001},
          {'params': utils.get_params(net, scope='fuse', bias=True), 'lr': LR*0.002},
                              ], lr=LR)
                              

loss = 0.0
loss_ = 0.0
def adjust_lr(optimizer, step_size):
    if step_size % 5000 == 0:
        LR = args.lr * (0.32 ** (step_size // 5000))
        if len(optimizer.param_groups) < 4:
            raise ValueError('Something Wrong with Learning Rate')
        optimizer.param_groups[0]['lr'] = LR
        optimizer.param_groups[1]['lr'] = LR * 2
        optimizer.param_groups[2]['lr'] = LR * 100
        optimizer.param_groups[3]['lr'] = LR * 200
        optimizer.param_groups[4]['lr'] = LR * 0.1
        optimizer.param_groups[5]['lr'] = LR * 0.2
        optimizer.param_groups[6]['lr'] = LR * 0.001
        optimizer.param_groups[7]['lr'] = LR * 0.002
        
    
for i in range(args.max_steps):
    image, label = next(train_iterator)
    # print(label.max())
    image, label = image.to(device), label.to(device)
    output = net(image)
    optimizer.zero_grad()
    if i % 5000 == 0:
        adjust_lr(optimizer, i)
    side0 = Loss(output[0], label)
    side1 = Loss(output[1], label)
    side2 = Loss(output[2], label)
    side3 = Loss(output[3], label)
    side4 = Loss(output[4], label)
    side5 = Loss(output[5], label)
    loss = side0 + side1 + side2 + side3 + side4 + side5
    
    if i % 10 == 0:
        loss_ = loss.cpu()
        print('step {} loss is {}'.format(i, loss_.data.numpy()))
    if i % 2000 == 0:
        torch.save(net.state_dict(), args.save_path + str(i) + '.pth')
    


    loss.backward()

    optimizer.step()



torch.save(net.state_dict(), args.save_path + str(i) + '.pth')
   
