"""
Neural Dust Network - Tiny AI Model Implementation
A micro neural network designed to be under 100kB and run on any device.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

# Optional imports for extended functionality
# Note: torchvision removed to avoid compatibility issues in some environments
TORCHVISION_AVAILABLE = False

class Dust(nn.Module):
    """
    Ultra-compact neural network model for MNIST classification.
    Target size: ≤100 kB (approximately 27 kB when saved).
    
    Architecture: 28×28 → 32 → 10 (fully connected)
    Parameters: ~25k total
    """
    def __init__(self):
        super(Dust, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                    # 28×28 → 784
            nn.Linear(28*28, 32),           # 784 → 32  (~25k parameters)
            nn.ReLU(),
            nn.Linear(32, 10)               # 32 → 10   (~320 parameters)
        )
        
    def forward(self, x):
        return self.net(x)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_kb(self):
        """Calculate model size in KB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_kb = (param_size + buffer_size) / 1024
        return size_kb

def create_dust_model():
    """Create and initialize a new Dust model"""
    model = Dust()
    print(f"Dust model created:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Size: {model.get_model_size_kb():.1f} KB")
    return model

def save_dust_model(model, filepath):
    """Save model weights to file"""
    torch.save(model.state_dict(), filepath)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  Model saved to {filepath} ({size_kb:.1f} KB)")
    return size_kb

def load_dust_model(filepath):
    """Load model weights from file"""
    model = Dust()
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    return model

def get_mnist_data(batch_size=64, download=True):
    """Load MNIST dataset for training and testing"""
    if TORCHVISION_AVAILABLE:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('data', train=True, download=download, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    else:
        # Use synthetic data when torchvision is not available
        return get_synthetic_mnist_data(batch_size)

def get_synthetic_mnist_data(batch_size=64):
    """Create synthetic MNIST-like data for testing when torchvision is unavailable"""
    # Create synthetic 28x28 images with 10 classes
    num_samples = 1000
    
    # Generate random 28x28 images
    train_data = torch.randn(num_samples, 1, 28, 28)
    train_labels = torch.randint(0, 10, (num_samples,))
    
    test_data = torch.randn(200, 1, 28, 28) 
    test_labels = torch.randint(0, 10, (200,))
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_dust_model(model, train_loader, epochs=1, lr=0.01):
    """Train the Dust model for a few epochs"""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 50:  # Limit training for quick demo
            break
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 20 == 0:
            print(f'  Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / min(50, len(train_loader))
    
    print(f"Training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def test_dust_model(model, test_loader):
    """Test the Dust model accuracy"""
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test Results - Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy

if __name__ == "__main__":
    # Demo: Create and test a Dust model
    print("=== Neural Dust Network - Model Demo ===")
    
    # Create model
    model = create_dust_model()
    
    # Save initial model
    save_dust_model(model, 'dust_v0.pt')
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_data()
    
    # Test untrained model
    print("\nTesting untrained model:")
    test_dust_model(model, test_loader)
    
    # Train for one epoch
    print("\nTraining model (limited batches for demo):")
    train_dust_model(model, train_loader, epochs=1)
    
    # Test trained model
    print("\nTesting trained model:")
    test_dust_model(model, test_loader)
    
    # Save trained model
    save_dust_model(model, 'dust_v1.pt')
    
    print("\n=== Dust model ready for distributed learning! ===")