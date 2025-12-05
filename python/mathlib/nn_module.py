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


# ============== Data Loading ==============

class Dataset:
    """数据集基类"""
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TensorDataset(Dataset):
    """从Tensor创建数据集"""
    def __init__(self, *tensors):
        self.tensors = tensors
        self.size = tensors[0].shape[0] if tensors else 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return tuple(self._slice(t, idx) for t in self.tensors)

    def _slice(self, tensor, idx):
        """提取单个样本"""
        shape = list(tensor.shape)
        if len(shape) == 1:
            return Tensor([1], [tensor.numpy()[idx]])
        new_shape = shape[1:]
        data = tensor.numpy()[idx].flatten().tolist()
        return Tensor(new_shape, data)


class DataLoader:
    """数据加载器"""
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.dataset[i] for i in batch_indices]

            # 合并batch
            num_tensors = len(batch[0])
            result = []
            for i in range(num_tensors):
                arrays = [item[i].numpy() for item in batch]
                stacked = np.stack(arrays, axis=0)
                t = Tensor(list(stacked.shape), stacked.flatten().tolist())
                result.append(Variable(t, requires_grad=False))

            yield tuple(result)

