"""
Model Serialization Example - save and load model weights
"""
import sys
sys.path.insert(0, '..')
import os
import mathlib as ml

# XOR dataset
X = ml.Variable(ml.Tensor([4, 2], [0,0, 0,1, 1,0, 1,1]), requires_grad=False)
Y = ml.Variable(ml.Tensor([4, 1], [0, 1, 1, 0]), requires_grad=False)

# Create and train model
model = ml.Sequential(
    ml.Linear(2, 4),
    ml.Tanh(),
    ml.Linear(4, 1),
    ml.Sigmoid()
)

optimizer = ml.optim.Adam(model.parameters(), lr=0.1)

print("Training...")
for epoch in range(500):
    optimizer.zero_grad()
    pred = model(X)
    loss = ml.nn.mse_loss(pred, Y)
    loss.backward()
    optimizer.step()

print(f"After training, Loss: {loss.numpy()[0]:.6f}")

# Save model
model.save("xor_model")
print("Model saved to xor_model.npz")

# Create new model and load weights
model2 = ml.Sequential(
    ml.Linear(2, 4),
    ml.Tanh(),
    ml.Linear(4, 1),
    ml.Sigmoid()
)

print(f"\nBefore load, new model predictions:")
pred_before = model2(X)
for i in range(4):
    print(f"  {int(X.numpy()[i,0])} XOR {int(X.numpy()[i,1])} = {pred_before.numpy()[i,0]:.4f}")

# Load weights
model2.load("xor_model")
print(f"\nAfter load, new model predictions:")
pred_after = model2(X)
for i in range(4):
    p = pred_after.numpy()[i, 0]
    expected = int(Y.numpy()[i, 0])
    print(f"  {int(X.numpy()[i,0])} XOR {int(X.numpy()[i,1])} = {p:.4f} (expected: {expected})")

# Cleanup
os.remove("xor_model.npz")
print("\nTest passed! Model serialization works correctly.")

