#!/usr/bin/env python3
"""
Simple MNIST Training Script
Works on both AWS and Nebius (no platform-specific code)

Author: Reem Sabawi
Purpose: Demonstrate portable PyTorch training across cloud providers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import argparse


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST classification
    Architecture: Conv -> Pool -> Conv -> Pool -> FC -> FC
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # First conv block
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        
        # Second conv block
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        
        # Flatten and fully connected
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_one_epoch(model, train_loader, optimizer, criterion, device, verbose=True):
    """
    Train model for one epoch
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to GPU
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Print progress
        if verbose and batch_idx % 100 == 0:
            current_acc = 100 * correct / total
            print(f'  Batch {batch_idx:3d}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {current_acc:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """
    Validate model on test set
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    return test_loss, test_acc


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train MNIST classifier')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')
    parser.add_argument('--save-model', action='store_true', help='save trained model')
    args = parser.parse_args()
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'🚀 MNIST Training on {device.type.upper()}')
    print(f'{"="*60}\n')
    
    if torch.cuda.is_available():
        print(f'GPU Device: {torch.cuda.get_device_name(0)}')
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'GPU Memory: {gpu_memory:.2f} GB')
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'PyTorch Version: {torch.__version__}')
    else:
        print('⚠️  No GPU detected - training will be slow!')
    
    print(f'\nTraining Configuration:')
    print(f'  Epochs: {args.epochs}')
    print(f'  Batch Size: {args.batch_size}')
    print(f'  Learning Rate: {args.lr}')
    print(f'  Data Directory: {args.data_dir}\n')
    print(f'{"="*60}\n')
    
    # Data loading
    print('📥 Loading MNIST dataset...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        args.data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        args.data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    print(f'✅ Dataset loaded:')
    print(f'   Training samples: {len(train_dataset):,}')
    print(f'   Test samples: {len(test_dataset):,}')
    print(f'   Batches per epoch: {len(train_loader)}\n')
    
    # Model setup
    print('🏗️  Initializing model...')
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'✅ Model created: {total_params:,} parameters\n')
    
    # Training loop
    print(f'{"="*60}')
    print(f'🔥 Starting Training')
    print(f'{"="*60}\n')
    
    overall_start = time.time()
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}:')
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        test_loss, test_acc = validate(
            model, test_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f'\n  📊 Epoch {epoch+1} Results:')
        print(f'     Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'     Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%')
        print(f'     Time: {epoch_time:.2f}s')
        print(f'  {"-"*56}\n')
    
    total_time = time.time() - overall_start
    
    # Final results
    print(f'{"="*60}')
    print(f'✅ Training Complete!')
    print(f'{"="*60}')
    print(f'\nFinal Results:')
    print(f'  Test Accuracy: {test_acc:.2f}%')
    print(f'  Total Time: {total_time:.2f}s')
    print(f'  Avg Time per Epoch: {total_time/args.epochs:.2f}s')
    
    # Save model
    if args.save_model:
        model_path = 'mnist_cnn.pth'
        torch.save(model.state_dict(), model_path)
        print(f'\n💾 Model saved: {model_path}')
    
    print(f'\n{"="*60}\n')


if __name__ == '__main__':
    main()
