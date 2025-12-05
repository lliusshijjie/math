"""
XOR Problem - Verify mathlib can handle deep learning tasks
"""
import sys
sys.path.insert(0, '..')

import mathlib as ml

# XOR dataset
X = ml.Variable(ml.Tensor([4, 2], [0,0, 0,1, 1,0, 1,1]), requires_grad=False)
Y = ml.Variable(ml.Tensor([4, 1], [0, 1, 1, 0]), requires_grad=False)

# Model: 2 -> 4 -> 1
model = ml.Sequential(
    ml.Linear(2, 4),
    ml.Tanh(),
    ml.Linear(4, 1),
    ml.Sigmoid()
)

optimizer = ml.optim.Adam(model.parameters(), lr=0.1)

print("Training XOR...")
for epoch in range(1000):
    optimizer.zero_grad()
    
    pred = model(X)
    loss = ml.nn.mse_loss(pred, Y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.numpy()[0]:.6f}")

# Test
print("\nResults:")
pred = model(X)
for i in range(4):
    x1, x2 = int(X.numpy()[i, 0]), int(X.numpy()[i, 1])
    p = pred.numpy()[i, 0]
    print(f"  {x1} XOR {x2} = {p:.4f} (expected: {int(Y.numpy()[i, 0])})")

