# The MNIST-1D dataset | 2020
# Sam Greydanus

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def make_mlp_layer(
  neurons_in:int,
  neurons_out:int,
  activation_function,
  bias:bool,
  batchnorm:bool
) -> List:
  modules = []
  if batchnorm:
    modules.append(nn.BatchNorm1d(neurons_in))
  return modules + [nn.Linear(neurons_in, neurons_out, bias=bias), activation_function()]

def make_conv_layer(
  channels_in:int,
  channels_out:int,
  kernel_size:int,
  activation_function,
  bias:bool,
  batchnorm:bool
) -> List:
  modules = []
  if batchnorm:
    modules.append(nn.BatchNorm1d(channels_in))
  return modules + [nn.Conv1d(channels_in, channels_out, kernel_size, bias=bias), activation_function()]


class LinearBase(nn.Module):
  def __init__(self, input_size, output_size):
    super(LinearBase, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
    print("Initialized LinearBase model with {} parameters".format(self.count_params()))

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    return self.linear(x)

class MLPBase(nn.Module):
  def __init__(self, input_size, output_size, hidden_size=100):
    super(MLPBase, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, output_size)
    print("Initialized MLPBase model with {} parameters".format(self.count_params()))

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x):
    h = self.linear1(x).relu()
    h = h + self.linear2(h).relu()
    return self.linear3(h)

class MLP(nn.Module):
  '''
  A simple multi-layer perceptron with custom activations and optional batchnorm.

  Parameters
  ----------
  input_size : int
    Number of input features
  output_size : int
    Number of output features
  hidden_size : int
    Number of neurons in each hidden layer
  depth : int
    Number of hidden layers
  activation_function
    The activation function to use
  bias : bool
    Whether to use bias in the linear layers
  batchnorm : bool
    Whether to use batchnorm in the hidden layers
  '''
  def __init__(
    self,
    input_size:int,
    output_size:int=10,
    hidden_size:int=100,
    depth:int=3,
    activation_function=nn.ReLU,
    bias:bool=True,
    batchnorm:bool=True
  ) -> None:
    super(MLP, self).__init__()

    layers = []
    for i in range(depth):
      neurons_out = hidden_size if i < depth-1 else output_size
      if i == 0:
        neurons_in = input_size
      else:
        neurons_in = hidden_size
      layers += make_mlp_layer(neurons_in, neurons_out, activation_function, bias, batchnorm=(batchnorm and i>0))

    self.network = nn.Sequential(*layers)
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.network(x)

class CNN(nn.Module):
  '''
  A simple convolutional neural network with custom activations and optional batchnorm.

  Parameters
  ----------
  input_size : int
    Number of input features
  output_size : int
    Number of output features
  num_layers : int
    Number of convolutional layers
  kernel_size : int
    Size of the convolutional kernel
  base_width : int
    Number of channels in the first layer
  activation_function
    The activation function to use
  bias : bool
    Whether to use bias in the convolutional layers
  batchnorm : bool
    Whether to use batchnorm in the convolutional layers
  '''
  def __init__(
    self,
    input_size:int,
    output_size:int=10,
    num_layers:int=3,
    kernel_size:int=3,
    base_width:int=16,
    activation_function=nn.ReLU,
    bias:bool=True,
    batchnorm:bool=True
  ) -> None:
    super(CNN, self).__init__()

    layers = []
    in_channels = [1] + [base_width*i for i in range(1, num_layers)]
    out_channels = [base_width*i for i in range(1, num_layers+1)]
    for i in range(num_layers):
      layers += make_conv_layer(in_channels[i], out_channels[i], kernel_size, activation_function, bias, batchnorm=(batchnorm and i>0))

    layers.append(nn.AdaptiveAvgPool1d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(out_channels[-1], output_size, bias=bias))

    
    self.network = nn.Sequential(*layers)
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.network(x)
    

class ConvBase(nn.Module):
    def __init__(self, output_size, channels=25, linear_in=125):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.linear = nn.Linear(linear_in, output_size) # flattened channels -> 10 (assumes input has dim 50)
        print("Initialized ConvBase model with {} parameters".format(self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False): # the print statements are for debugging
        x = x.view(-1,1,x.shape[-1])
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1) # flatten the conv features
        return self.linear(h3) # a linear classifier goes on top

class GRUBase(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_size=6, time_steps=40, bidirectional=True):
    super(GRUBase, self).__init__()
    self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=bidirectional)
    flat_size = 2*hidden_size*time_steps if bidirectional else hidden_size*time_steps
    self.linear = torch.nn.Linear(flat_size, output_size)
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional
    print("Initialized GRUBase model with {} parameters".format(self.count_params()))

  def count_params(self):
    return sum([p.view(-1).shape[0] for p in self.parameters()])

  def forward(self, x, h0=None): # assumes seq has [batch, time]
    x = x.view(*x.shape[:2], 1) # add a spatial dimension
    k = 2 if self.bidirectional else 1
    h0 = 0*torch.randn(k, x.shape[0], self.hidden_size) if h0 is None else h0
    h0 = h0.to(x.device) # GPU support

    output, hn = self.gru(x, h0)
    output = output.reshape(output.shape[0],-1) # [batch, time*hidden_size]
    return self.linear(output)