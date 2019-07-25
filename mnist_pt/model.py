import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicConv2D(nn.Module) :
    def __init__(self, *args, **kwargs) :
        super(DynamicConv2D, self).__init__()

        tmp = nn.Conv2d(*args, **kwargs)

        self.weight = tmp.weight.data.clone().detach().requires_grad_(True)
        self.bias = tmp.bias.data.clone().detach().requires_grad_(True)
        self.padding = tmp.padding

    def forward(self, x) :
        return F.conv2d(x, self.weight, self.bias, padding=self.padding)

    def named_parameters(self, prefix='', recurse=True) :
        return [("weight", self.weight), ("bias", self.bias)]

    def set_tensors(self, param_dict, prefix="") :
        self.weight = param_dict[prefix + "weight"]
        self.bias = param_dict[prefix + "bias"]

class DynamicLinear(nn.Module) :
    def __init__(self, *args, **kwargs) :
        super(DynamicLinear, self).__init__()

        tmp = nn.Linear(*args, **kwargs)

        self.weight = tmp.weight.data.clone().detach().requires_grad_(True)
        self.bias = tmp.bias.data.clone().detach().requires_grad_(True)

    def forward(self, x) :
        return F.linear(x, self.weight, self.bias)

    def named_parameters(self, prefix='', recurse=True) :
        return [("weight", self.weight), ("bias", self.bias)]

    def set_tensors(self, param_dict, prefix="") :
        self.weight = param_dict[prefix + "weight"]
        self.bias = param_dict[prefix + "bias"]

class LeNet(nn.Module) :
    def __init__(self, tensor_dict = {}) :
        super(LeNet, self).__init__()

        self.conv1 = DynamicConv2D(1, 16, 5, padding=2)
        self.conv2 = DynamicConv2D(16, 32, 5, padding=2)
        self.conv3 = DynamicConv2D(32, 64, 5, padding=2)

        self.fc4 = DynamicLinear(1024, 100)
        self.fc5 = DynamicLinear(100, 10)

    def forward(self, x) :
        tval = self.conv1.forward(x)
        tval = F.max_pool2d(tval, kernel_size=3, stride=2, padding=1)
        tval = F.relu(tval)

        tval = self.conv2.forward(tval)
        tval = F.max_pool2d(tval, kernel_size=3, stride=2, padding=1)
        tval = F.relu(tval)

        tval = self.conv3.forward(tval)
        tval = F.max_pool2d(tval, kernel_size=3, stride=2, padding=1)
        tval = F.relu(tval)

        tval = tval.view(-1, 1024)

        tval = F.relu(self.fc4.forward(tval))
        logits = F.relu(self.fc5.forward(tval))

        return logits

    def loss(self, x, y, ex_wts = None) :
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y, reduction='none')
        if ex_wts is not None : loss = loss * ex_wts
        return logits, loss.mean()

    def named_parameters(self, prefix='', recurse=True) :
        res = []
        for name, ch in self.named_children() :
            res.extend([(name + "." + ch_name, p) for ch_name, p in ch.named_parameters(prefix, recurse)])

        return res

    def set_tensors(self, param_dict, prefix="") :
        for name, ch in self.named_children() :
            ch.set_tensors(param_dict, prefix + name + ".")


def reweight_autodiff(model, x, y, x_val, y_val) :
    ex_wts = torch.ones([x.shape[0]], requires_grad=True)
    logits, loss = model.loss(x, y, ex_wts)

    model_vars = list(model.named_parameters())
    model_var_grads = torch.autograd.grad([loss], [p for name, p in model_vars], create_graph=True)
    model_vars_new = dict([(var[0], var[1] - g) for var, g in zip(model_vars, model_var_grads)])

    model2 = LeNet()
    model2.set_tensors(model_vars_new)

    ex_wts_val = torch.ones([x_val.shape[0]], requires_grad=False) / len(x_val)
    logits_val, loss_val = model2.loss(x_val, y_val, ex_wts_val)
    ex_wts_grad = torch.autograd.grad([loss_val], [ex_wts], retain_graph=True)

    exit(1)
