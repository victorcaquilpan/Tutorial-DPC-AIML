#!/usr/bin/env python3
"""
Fashion-MNIST Image Classification Training Script
Usage: python train.py --path_data /path/to/data --path_results /path/to/results
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FashionMNISTDataset(Dataset):
    """Custom Dataset class for Fashion-MNIST CSV data"""
    
    def __init__(self, data_path, transform=None, is_train=True):
        if is_train:
            self.data = pd.read_csv(os.path.join(data_path))
        else:
            self.data = pd.read_csv(os.path.join(data_path))
        
        self.labels = self.data['label'].values
        self.images = self.data.drop('label', axis=1).values
        self.transform = transform
        
        # Fashion-MNIST class names
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Reshape flat image to 28x28
        image = self.images[idx].reshape(28, 28).astype(np.uint8)
        label = self.labels[idx]
        
        # Convert to 3-channel image for ResNet (RGB)
        image = np.stack([image, image, image], axis=2)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ResNetClassifier(nn.Module):
    """ResNet-18 based classifier for Fashion-MNIST"""
    
    def __init__(self, num_classes=10):
        super(ResNetClassifier, self).__init__()
        # Load pre-trained ResNet-18
        self.backbone = resnet18(pretrained=True)
        
        # Modify first conv layer to accept smaller input size
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # Remove maxpool for small images
        
        # Replace final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def create_data_loaders(data_path, batch_size=32, val_split=0.2):
    """Create train, validation, and test data loaders"""
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),  # Resize to 32x32 for better ResNet performance
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets
    train_dataset = FashionMNISTDataset(data_path, transform=train_transform, is_train=True)
    test_dataset = FashionMNISTDataset(data_path, transform=val_transform, is_train=False)
    
    # Split training data into train and validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, train_dataset.class_names

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(pbar.n+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(pbar.n+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_model(model, test_loader, device, class_names):
    """Test the model and generate classification report"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    return accuracy, report

def main():
    parser = argparse.ArgumentParser(description='Fashion-MNIST ResNet Training')
    parser.add_argument('--path_data', type=str, required=True,
                        help='Path to the data directory containing CSV files')
    parser.add_argument('--path_results', type=str, required=True,
                        help='Path to save results and checkpoints')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.path_results, exist_ok=True)
    
    # Default hyperparameters
    hyperparams = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'weight_decay': 1e-4,
        'step_size': 7,
        'gamma': 0.1,
        'val_split': 0.2
    }
    
    print("Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print()
    
    # Save hyperparameters
    with open(os.path.join(args.path_results, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        args.path_data, 
        batch_size=hyperparams['batch_size'], 
        val_split=hyperparams['val_split']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {class_names}")
    print()
    
    # Initialize model
    model = ResNetClassifier(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=hyperparams['learning_rate'], 
                          weight_decay=hyperparams['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=hyperparams['step_size'], 
                                         gamma=hyperparams['gamma'])
    
    # Training loop
    print("Starting training...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(hyperparams['num_epochs']):
        print(f"\nEpoch {epoch+1}/{hyperparams['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.path_results, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch+1, train_loss, val_loss, 
                           train_acc, val_acc, best_model_path)
            print(f"New best validation accuracy: {val_acc:.2f}%")
    
    # Plot training curves
    plot_path = os.path.join(args.path_results, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)
    print(f"\nTraining curves saved to: {plot_path}")
    
    # Test the best model
    print("\nTesting best model...")
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_acc, test_report = test_model(model, test_loader, device, class_names)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save test results
    results = {
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc,
        'classification_report': test_report,
        'hyperparameters': hyperparams,
        'training_summary': {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1]
        }
    }
    
    with open(os.path.join(args.path_results, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed! Results saved to: {args.path_results}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()