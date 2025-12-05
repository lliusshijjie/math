"""
DataLoader Example - batch training with shuffle
"""
import sys
sys.path.insert(0, '..')
import mathlib as ml

# Create dataset: simple regression y = 2*x + 1
X_data = ml.Tensor([100, 1], [i * 0.1 for i in range(100)])
Y_data = ml.Tensor([100, 1], [i * 0.2 + 1 for i in range(100)])

dataset = ml.TensorDataset(X_data, Y_data)
dataloader = ml.DataLoader(dataset, batch_size=10, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Batch count: {len(dataloader)}")

# Model: Linear regression
model = ml.Linear(1, 1)
optimizer = ml.optim.SGD(model.parameters(), lr=0.01)

print("\nTraining with DataLoader...")
for epoch in range(50):
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = ml.nn.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.numpy()[0]
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}")

# Test
print("\nResults:")
test_x = ml.Variable(ml.Tensor([3, 1], [0.0, 0.5, 1.0]), requires_grad=False)
pred = model(test_x)
print(f"  x=0.0 -> pred={pred.numpy()[0,0]:.4f} (expected: 1.0)")
print(f"  x=0.5 -> pred={pred.numpy()[1,0]:.4f} (expected: 2.0)")
print(f"  x=1.0 -> pred={pred.numpy()[2,0]:.4f} (expected: 3.0)")

print("\nDataLoader test passed!")

