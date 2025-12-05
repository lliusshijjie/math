"""Simple neural network module built on top of mathlib C++ backend"""

import numpy as np
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

    def state_dict(self):
        """获取模型参数字典"""
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.numpy()
        for name, module in self._modules.items():
            for k, v in module.state_dict().items():
                state[f"{name}.{k}"] = v
        return state

    def load_state_dict(self, state_dict):
        """加载模型参数"""
        for name, param in self._parameters.items():
            if name in state_dict:
                arr = state_dict[name]
                t = Tensor(list(arr.shape), arr.flatten().tolist())
                param.set_data(t)
        for name, module in self._modules.items():
            prefix = f"{name}."
            sub_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            module.load_state_dict(sub_state)

    def save(self, path):
        """保存模型到文件"""
        np.savez(path, **self.state_dict())

    def load(self, path):
        """从文件加载模型"""
        if not path.endswith('.npz'):
            path = path + '.npz'
        state_dict = dict(np.load(path))
        self.load_state_dict(state_dict)


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

