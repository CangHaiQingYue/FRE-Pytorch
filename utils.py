import torch
from deconv import bilinear_upsample_weights

def init_vgg_parameters(net, pre_trained):
    # pre_trained = torch.load(path)
    pre = {k : v for k, v in pre_trained.items() if 'features' in k}
    vgg = {k : v for k, v in net.state_dict().items() if 'conv' in k}

    for c, f in zip(vgg.items(), pre.items()):
        assert c[1].size() == f[1].size()
        vgg[c[0]].data = f[1]
    net.load_state_dict(vgg, strict=False)
    print(' VGG Part Initialized Done')

    return net


def init_side_parameters(net):
    name = {'xavier' : ['0.0', '1.0',  '1.3' , '1.6' ],
       'truncated_normal' : ['2.0'],
        'batch_normal':['0.1', '1.1', '1.4', '1.7'],
        'deconv': ['2.1']
        }
    factor = iter((2, 4, 8, 8))
    side = {k : v for k, v in net.state_dict().items() if 'side' in k}
    for key, value in side.items():
        if 'side1.0.weight' in key:
            torch.nn.init.normal_(net.state_dict()[key].data, mean=0, std=0.01)
            continue
        # all bias initialized by zero. Actually in pytorch, default is zero
        if 'bias' in key:
            torch.nn.init.constant_(net.state_dict()[key].data, 0.0)
            continue
        # residual block use xavier_initialzer
        if key[6:9] in name['xavier']:
            torch.nn.init.xavier_normal_(net.state_dict()[key].data)
        # the layer before upsamle use normal_initialzer 
        elif key[6:9] in name['truncated_normal']:
            torch.nn.init.normal_(net.state_dict()[key].data, mean=0, std=0.01)
        # BN layer's gamm(weights) is 1.0, beta(bias) is 0.0.
        # run_meam and run_var use default init way
        elif key[6:9] in name['batch_normal'] and 'weight' in key:
            torch.nn.init.constant_(net.state_dict()[key].data, 1.0)
        # At last, deconv layers should be initialzed by bilinear
        elif key[6:9] in name['deconv']:
            bilinear_weights = bilinear_upsample_weights(next(factor), 1)
            net.state_dict()[key].data = torch.tensor(bilinear_weights)            

    print(' Side Part Initialized Done')
    return net


class DeviceError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__()
        self.info = ErrorInfo
    def __str__(self):
        return self.info

#################### def learning rate for different layers##############
'''

Those function are deprecated

'''
def get_1x_lr(net):
    a = (net.conv1, net.conv2, net.conv3, net.conv4)
    for i in a:
        yield i.parameters()

def get_100x_lr(net):
    yield net.conv5.parameters()

def get_01x_lr(net):
    a = (net.side1, net.side2, net.side3, net.side4, net.side5)
    for i in a:
        yield i.parameters()

def get_0001_lr(net):
    yield net.fuse.parameters()
######################## set different lr to weights and biase##########
def get_params(net, scope, bias=False):
    name_scope = {'conv' : ['conv1', 'conv2', 'conv3', 'conv4'],
                   'conv5': ['conv5'],
                   'side': ['side1', 'side2', 'side3', 'side4', 'side5'],
                   'fuse': ['fuse']}
    if scope not in name_scope.keys():
        raise ValueError('Master: The Name of Scope Out of Index')
    for name, parameters in net.named_parameters():
        for i in name_scope[scope]:
            if not bias:
                if 'weight' in name and i in name:
                    # return name
                    yield parameters
            else:
                if 'bias' in name and i in name:
                    # print(name)
                    yield parameters
