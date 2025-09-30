# Copyright 2023 The PyGlove Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train MNIST.

This is a basic working ML program which trains MNIST.
The code is modified from the pytorch mnist example:
https://github.com/pytorch/examples/blob/main/mnist/main.py
"""
from absl import app
import pyglove as pg
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from torchvision import transforms


class Net(nn.Module):
  """Nerual architecture for MNIST."""

  def __init__(self):
    super(Net, self).__init__()
    filters1 = pg.oneof([32, 64], name='filters1')
    self.conv1 = nn.Conv2d(1, filters1, 3, 1)
    self.conv2 = nn.Conv2d(filters1, filters1 * 2, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    fc1_input_dims = filters1 * 2 * 144
    fc1_output_dims = pg.oneof([64, 128, 256], name='fc1_dims')
    self.fc1 = nn.Linear(fc1_input_dims, fc1_output_dims)
    self.fc2 = nn.Linear(fc1_output_dims, 10)

  def forward(self, x):
    activation = pg.oneof([F.relu, F.sigmoid], name='activation')
    x = self.conv1(x)
    x = activation(x)
    x = self.conv2(x)
    x = activation(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = activation(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output


def train(model, device, train_loader, optimizer, epoch,
          dry_run=False, log_interval=100):
  """Train model."""
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),    # pytype: disable=wrong-arg-types
          100. * batch_idx / len(train_loader), loss.item()))
    if dry_run:
      break


def test(model, device, test_loader):
  """Test model."""
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()
      pred = output.argmax(dim=1, keepdim=True)
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)               # pytype: disable=wrong-arg-types
  accuracy = correct / len(test_loader.dataset)       # pytype: disable=wrong-arg-types
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),   # pytype: disable=wrong-arg-types
      100. * accuracy))
  return accuracy


def train_and_eval(lr=1.0, batch_size=64, epochs=1, dry_run=False) -> float:
  """Train model and test model."""
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])
  dataset1 = datasets.MNIST('.', train=True, download=True, transform=transform)
  dataset2 = datasets.MNIST('.', train=False, transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
  test_loader = torch.utils.data.DataLoader(dataset2, batch_size=128)

  device = torch.device('cpu')
  model = Net().to(device)
  optimizer = optim.Adadelta(model.parameters(), lr=lr)

  scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
  accuracy = 0.0
  for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch, dry_run=dry_run)
    if dry_run:
      break
    accuracy = test(model, device, test_loader)
    scheduler.step()
  return accuracy


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  def inspect_search_space():
    _ = train_and_eval(dry_run=True)

  search_space = pg.hyper.trace(inspect_search_space, require_hyper_name=True)
  search_algorithm = pg.evolution.regularized_evolution()

  for automl_context, feedback in pg.sample(
      search_space, search_algorithm, num_examples=5
  ):
    with automl_context():
      print(
          'Starting trial #%d with parameters %s.'
          % (
              feedback.id,
              {
                  k: search_space.evaluate(v)
                  for k, v in (search_space.hyper_dict or {}).items()
              },
          )
      )
      accuracy = train_and_eval(epochs=1)
      feedback(accuracy)


if __name__ == '__main__':
  app.run(main)
