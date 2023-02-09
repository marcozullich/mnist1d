import torch
import argparse
import pickle as pkl
import pandas as pd
import os
from collections import OrderedDict

from data import data_to_dataloaders
from models import MLP, CNN
from train import train_model, eval_model
import activations


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', choices=['MLP', 'CNN'], help="Model architecture (choices: MLP, CNN; default: MLP)")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of layers (default: 3)")
    parser.add_argument('--hidden_size', type=int, default=16, help="Hidden size for layers. For MLP → number of neurons, for CNN → base width of number of channels (output dim = base_width * layer #)")
    parser.add_argument('--kernel_size', type=int, default=5, help="Kernel size for CNN (default: 5)")
    parser.add_argument('--activation_function', type=str, default='ReLU', help="Activation function. Choose any subclass of torch.nn.Module or activations.py (default: ReLU)")
    parser.add_argument('--batchnorm', action='store_true', help="Use batchnorm (default: False)")
    parser.add_argument('--bias', action='store_true', help="Use bias (default: False)")
    parser.add_argument('--data', type=str, default='mnist1d_data.pkl', help="Path to data (default: data.pkl)")
    parser.add_argument("--optimizer", type=str, default='SGD', choices=['Adam', 'RAdam', 'SGD'], help="Optimizer (choices: Adam, SGD; default: Adam)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (default: 0.9)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (default: 0.0005)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--save_folder", type=str, default='results', help="Folder to save results (default: results)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: None -> use GPU if available, else CPU)")
    parser.add_argument("--debug", action='store_true', help="Debug mode (default: False)")

    args = parser.parse_args()
    return args

def debug_mode(args):
    args.model = 'CNN'
    args.num_layers = 3
    args.hidden_size = 16
    args.kernel_size = 5
    args.activation_function = 'ReLU'
    args.batchnorm = False
    args.bias = False
    args.data = 'mnist1d_data.pkl'
    args.optimizer = 'SGD'
    args.lr = 0.1
    args.momentum = 0.9
    args.weight_decay = 0.0005
    args.batch_size = 64
    args.epochs = 10
    args.save_folder = 'results'
    return args

def get_data(location:str) -> dict:
    with open(location, 'rb') as f:
        data = pkl.load(f)
    return data

def get_activation(name):
    '''
    Get activation function from activations.py (prioritized) or torch.nn (fallback).
    '''
    try:
        return getattr(activations, name)
    except AttributeError:
        try:
            return getattr(torch.nn, name)
        except AttributeError:
            raise ValueError(f"Activation function {name} not supported or not recognized.")

def save_model(model, path, args):
    os.makedirs(path, exist_ok=True)
    files_list = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    while True:
        random_number = torch.randint(0, 100000, (1,)).item()
        if f"model_{random_number}.pt" not in files_list:
            break
    torch.save(model.state_dict(), os.path.join(path, f"model_{random_number}.pt"))

    args_save = ["model", "activation_function", "num_layers", "hidden_size", "kernel_size", "batchnorm", "bias", "optimizer", "lr", "momentum", "weight_decay", "batch_size", "epochs", "train_accuracy", "test_accuracy"]
    args_dict = OrderedDict([("model_ID", random_number)]) | OrderedDict([(k, v) for k, v in args.__dict__.items() if k in args_save])
    if os.path.isfile(os.path.join(path, "results.csv")):
        df = pd.read_csv(os.path.join(path, "results.csv"))
        df = df.append(args_dict, ignore_index=True)
    else:
        df = pd.DataFrame(args_dict, index=[0])
    df.to_csv(os.path.join(path, "results.csv"), index=False)





def init():
    args = get_args()
    if args.debug:
        args = debug_mode(args)

    data_dict = get_data(args.data)
    input_dim = data_dict['x'].shape[1]
    trainloader, testloader = data_to_dataloaders(data_dict, args.batch_size, num_workers=8)
    print("Created trainloader and testloader")

    activation_function = get_activation(args.activation_function)

    if args.model == 'MLP':
        model = MLP(input_size=input_dim, output_size=10, hidden_size=args.hidden_size, depth=args.num_layers, activation_function=activation_function, bias=args.bias, batchnorm=args.batchnorm)
    elif args.model == 'CNN':
        model = CNN(input_size=input_dim, output_size=10, num_layers=args.num_layers, kernel_size=args.kernel_size, base_width=args.hidden_size, activation_function=activation_function, bias=args.bias, batchnorm=args.batchnorm)
    else:
        raise ValueError(f"Model {args.model} not supported or not recognized.")
    print("Model architecture", "\n", model)

    try:
        optimizer_class = getattr(torch.optim, args.optimizer)
    except AttributeError:
        raise ValueError(f"Optimizer {args.optimizer} not supported or not recognized.")
    optimizer = optimizer_class(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_results = train_model(args.epochs, trainloader, model, optimizer, torch.nn.CrossEntropyLoss(), args.device)
    test_results = eval_model(testloader, model, torch.nn.CrossEntropyLoss(), args.device)

    args.train_accuracy = train_results["train_acc"][-1]
    args.test_accuracy = test_results[1]

    save_model(model, args.save_folder, args)


if __name__ == "__main__":
    init()