import pytest
import numpy as np
import mathlib as ml


class TestTensor:
    def test_construction(self):
        t = ml.Tensor([2, 3])
        assert t.shape == [2, 3]
        assert t.size == 6
        
    def test_from_numpy(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        t = ml.Tensor(arr)
        assert t.shape == [2, 3]
        
    def test_to_numpy(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        t = ml.Tensor(arr)
        result = t.numpy()
        np.testing.assert_array_equal(result, arr)
        
    def test_arithmetic(self):
        a = ml.Tensor.ones([2, 2])
        b = ml.Tensor.full([2, 2], 2.0)
        c = a + b
        np.testing.assert_array_equal(c.numpy(), np.full((2, 2), 3.0))
        
    def test_matmul(self):
        a = ml.Tensor(np.array([[1, 2], [3, 4]], dtype=np.float64))
        b = ml.Tensor(np.array([[1, 0], [0, 1]], dtype=np.float64))
        c = a.matmul(b)
        np.testing.assert_array_equal(c.numpy(), np.array([[1, 2], [3, 4]]))


class TestVariable:
    def test_construction(self):
        v = ml.Variable(ml.Tensor([3], 1.0), requires_grad=True)
        assert v.requires_grad == True
        assert v.shape == [3]
        
    def test_backward(self):
        x = ml.Variable(ml.Tensor([3], [1.0, 2.0, 3.0]), requires_grad=True)
        y = x * x
        loss = ml.sum(y)
        loss.backward()
        # gradient of x^2 is 2x
        np.testing.assert_array_almost_equal(
            x.grad_numpy(), np.array([2.0, 4.0, 6.0])
        )
        
    def test_chain(self):
        x = ml.Variable(ml.Tensor([2], [2.0, 3.0]), requires_grad=True)
        y = x * 2.0
        z = y + 1.0
        loss = ml.sum(z)
        loss.backward()
        np.testing.assert_array_almost_equal(x.grad_numpy(), np.array([2.0, 2.0]))


class TestNN:
    def test_sigmoid(self):
        t = ml.Tensor([3], [0.0, 1.0, -1.0])
        result = ml.nn.sigmoid(t)
        expected = 1 / (1 + np.exp(-np.array([0.0, 1.0, -1.0])))
        np.testing.assert_array_almost_equal(result.numpy(), expected)
        
    def test_relu(self):
        t = ml.Tensor([4], [-1.0, 0.0, 1.0, 2.0])
        result = ml.nn.relu(t)
        np.testing.assert_array_equal(result.numpy(), np.array([0, 0, 1, 2]))
        
    def test_mse(self):
        pred = ml.Tensor([3], [1.0, 2.0, 3.0])
        target = ml.Tensor([3], [1.0, 2.0, 3.0])
        loss = ml.nn.mse(pred, target)
        assert abs(loss) < 1e-10
        
    def test_init(self):
        t = ml.nn.init.xavier_uniform([3, 4])
        assert t.shape == [3, 4]


class TestOptim:
    def test_sgd(self):
        x = ml.Variable(ml.Tensor([1], [5.0]), requires_grad=True)
        opt = ml.optim.SGD([x], lr=0.1)
        
        for _ in range(50):
            opt.zero_grad()
            loss = ml.sum(x * x)
            loss.backward()
            opt.step()
            
        assert abs(x.numpy()[0]) < 0.1
        
    def test_adam(self):
        x = ml.Variable(ml.Tensor([1], [5.0]), requires_grad=True)
        opt = ml.optim.Adam([x], lr=0.1)
        
        for _ in range(200):
            opt.zero_grad()
            loss = ml.sum(x * x)
            loss.backward()
            opt.step()
            
        assert abs(x.numpy()[0]) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

