import os
from tinygrad import Tensor, nn

# Set the backend to CPU
os.environ["TINYGRAD_DEVICE"] = "CPU"

class LinearNet:
  def __init__(self):
    self.l1 = Tensor.kaiming_uniform(784, 128, device="CPU")
    self.l2 = Tensor.kaiming_uniform(128, 10, device="CPU")
  def __call__(self, x:Tensor) -> Tensor:
    return x.flatten(1).dot(self.l1).relu().dot(self.l2)

model = LinearNet()
optim = nn.optim.Adam([model.l1, model.l2], lr=0.001)

x, y = Tensor.rand(4, 1, 28, 28, device="CPU"), Tensor([2,4,3,7], device="CPU")  # replace with real mnist dataloader

with Tensor.train():
  for i in range(10):
    optim.zero_grad()
    loss = model(x).sparse_categorical_crossentropy(y).backward()
    optim.step()
    print(i, loss.item())