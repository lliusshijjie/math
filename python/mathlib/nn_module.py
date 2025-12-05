"""Simple neural network module built on top of mathlib C++ backend"""

from . import Tensor, Variable, matmul, sum as ml_sum
from . import nn, optim

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
    
    def forward(self, x):
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params
    
    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()
    
    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._parameters['weight'] = Variable(
            nn.init.xavier_uniform([in_features, out_features]), requires_grad=True)
        self._parameters['bias'] = Variable(
            Tensor([out_features], 0.0), requires_grad=True)
    
    def forward(self, x):
        return matmul(x, self._parameters['weight']) + self._parameters['bias']


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer
    
    def forward(self, x):
        for layer in self._modules.values():
            x = layer(x)
        return x


class ReLU(Module):
    def forward(self, x):
        return nn.relu_grad(x)


class Sigmoid(Module):
    def forward(self, x):
        return nn.sigmoid_grad(x)


class Tanh(Module):
    def forward(self, x):
        return nn.tanh_grad(x)

