#!/usr/bin/env python3
"""Evaluation script for SafeFusion."""

import argparse
import torch
import yaml
from pathlib import Path

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate SafeFusion')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating on {device}')
    print(f'Model weights: {args.weights}')

if __name__ == '__main__':
    main()
