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


class Dropout(Module):
    """Dropout layer"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        x_np = x.numpy()
        mask = (np.random.random(x_np.shape) > self.p).astype(np.float64)
        out_np = x_np * mask / (1 - self.p)

        out_t = Tensor(list(x_np.shape), out_np.flatten().tolist())
        return Variable(out_t, requires_grad=x.requires_grad)


class Embedding(Module):
    """Embedding layer: lookup table for integer indices"""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize weights with normal distribution
        weight_np = np.random.randn(num_embeddings, embedding_dim) * 0.1
        self._parameters['weight'] = Variable(
            Tensor([num_embeddings, embedding_dim], weight_np.flatten().tolist()),
            requires_grad=True
        )

    def forward(self, x):
        """x: integer indices tensor of shape (batch_size,) or (batch_size, seq_len)"""
        indices = x.numpy().astype(int).flatten()
        weight_np = self._parameters['weight'].numpy()

        embedded = weight_np[indices]

        # Reshape output
        x_shape = list(x.numpy().shape)
        out_shape = x_shape + [self.embedding_dim]
        out_t = Tensor(out_shape, embedded.flatten().tolist())
        return Variable(out_t, requires_grad=True)


class BatchNorm1d(Module):
    """Batch Normalization for 1D inputs (N, C) or (N, C, L)"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self._parameters['gamma'] = Variable(
            Tensor([num_features], [1.0] * num_features), requires_grad=True)
        self._parameters['beta'] = Variable(
            Tensor([num_features], [0.0] * num_features), requires_grad=True)

        # Running statistics (not parameters, not saved in state_dict by default)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        gamma = self._parameters['gamma']
        beta = self._parameters['beta']
        x_np = x.numpy()

        if self.training:
            # Compute batch statistics
            mean = x_np.mean(axis=0)
            var = x_np.var(axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x_np - mean) / np.sqrt(var + self.eps)

        # Convert back to Variable
        x_norm_t = Tensor(list(x_np.shape), x_norm.flatten().tolist())
        x_norm_v = Variable(x_norm_t, requires_grad=x.requires_grad)

        # Scale and shift: y = gamma * x_norm + beta
        # Broadcast gamma and beta
        gamma_np = gamma.numpy()
        beta_np = beta.numpy()

        result_np = x_norm * gamma_np + beta_np
        result_t = Tensor(list(x_np.shape), result_np.flatten().tolist())
        return Variable(result_t, requires_grad=x.requires_grad)

    def state_dict(self):
        state = super().state_dict()
        state['running_mean'] = self.running_mean
        state['running_var'] = self.running_var
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if 'running_mean' in state_dict:
            self.running_mean = state_dict['running_mean']
        if 'running_var' in state_dict:
            self.running_var = state_dict['running_var']


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

