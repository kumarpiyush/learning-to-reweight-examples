import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module) :
    def __init__(self, initial_values = None) :
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)

        self.fc4 = nn.Linear(1024, 100)
        self.fc5 = nn.Linear(100, 10)

        if initial_values is not None :
            for old_p, new_p in zip(self.named_parameters(), initial_values) :
                old_p[1].data = new_p[1].data

    def debug(self) :
        print(list(self.named_parameters())[1])

    def forward(self, x) :
        tval = self.conv1(x)
        tval = F.max_pool2d(tval, kernel_size=3, stride=2, padding=1)
        tval = F.relu(tval)

        tval = self.conv2(tval)
        tval = F.max_pool2d(tval, kernel_size=3, stride=2, padding=1)
        tval = F.relu(tval)

        tval = self.conv3(tval)
        tval = F.max_pool2d(tval, kernel_size=3, stride=2, padding=1)
        tval = F.relu(tval)

        tval = tval.view(-1, 1024)

        tval = F.relu(self.fc4(tval))
        logits = F.relu(self.fc5(tval))

        return logits

    def loss(self, x, y, ex_wts = None) :
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y, reduction='none')
        if ex_wts is not None : loss = loss * ex_wts
        return logits, loss.mean()


def reweight_autodiff(model, x, y, x_val, y_val) :
    ex_wts = torch.ones([x.shape[0]])
    logits, loss = model.loss(x, y, ex_wts)

    model.debug()

    model_vars = list(model.named_parameters())
    model_var_grads = torch.autograd.grad([loss], [p for name, p in model_vars], create_graph=True)

    print(model_var_grads[1])

    model_vars_new = [(var[0], var[1] - g) for var, g in zip(model_vars, model_var_grads)]
    model.debug()

    model2 = LeNet(model_vars_new)
    model2.debug()
    exit(1)
