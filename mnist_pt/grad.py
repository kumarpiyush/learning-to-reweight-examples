import torch

x = torch.tensor([4.0], requires_grad = True)
eps = torch.tensor([0.0], requires_grad = True)

y = eps * x * x
print(y)

grad = torch.autograd.grad([y], [x], create_graph=True)[0]
print(grad)

x2 = x - grad
y2 = x2 * x2

grad2 = torch.autograd.grad([y2], [eps], allow_unused=True)
print(grad2)
