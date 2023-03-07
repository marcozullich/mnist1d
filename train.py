# The MNIST-1D dataset | 2020
# Sam Greydanus

import time, copy
import numpy as np

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy, AverageMeter

def train_model(
  num_epochs:int,
  dataloader:torch.utils.data.Dataset,
  model:nn.Module,
  optimizer:optim.Optimizer,
  criterion:nn.Module,
  scheduler:optim.lr_scheduler._LRScheduler=None,
  device:Union[str, torch.device]=None,
  gradient_clip:float=None
):
  # criterion = nn.CrossEntropyLoss()
  # optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

  # x_train, x_test = torch.Tensor(dataset['x']), torch.Tensor(dataset['x_test'])
  # y_train, y_test = torch.LongTensor(dataset['y']), torch.LongTensor(dataset['y_test'])

  if device is None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model.train()
  model = model.to(device)
  # x_train, x_test, y_train, y_test = [v.to(args.device) for v in [x_train, x_test, y_train, y_test]]

  results = {"train_loss":[], "train_acc":[]}

  for epoch in range(num_epochs):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, (data, target) in enumerate(dataloader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      prediction = model(data)
      loss = criterion(prediction, target)
      loss.backward()

      if gradient_clip is not None and gradient_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

      optimizer.step()

      loss_value = loss.item()
      acc_value = accuracy(prediction, target)
      

      results["train_loss"].append(loss_value)
      results["train_acc"].append(acc_value)

      loss_meter.update(loss_value, data.size(0))
      acc_meter.update(accuracy(prediction, target), data.size(0))
    
    if scheduler is not None:
        # NB: step on train loss, not validation loss
        scheduler.step(loss_value)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss_meter.avg:.4f} | Acc: {acc_meter.avg:.4f}")
  return results

def eval_model(
  dataloader:torch.utils.data.Dataset,
  model:nn.Module,
  criterion:nn.Module,
  device:Union[str, torch.device]=None
):
  if device is None:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model.eval()
  model = model.to(device)

  loss_meter = AverageMeter()
  acc_meter = AverageMeter()

  for batch_idx, (data, target) in enumerate(dataloader):
    data, target = data.to(device), target.to(device)
    prediction = model(data)
    loss = criterion(prediction, target)

    loss_value = loss.item()
    acc_value = accuracy(prediction, target)

    loss_meter.update(loss_value, data.size(0))
    acc_meter.update(accuracy(prediction, target), data.size(0))
  
  print(f"Loss: {loss_meter.avg:.4f} | Acc: {acc_meter.avg:.4f}")
  return loss_meter.avg, acc_meter.avg
