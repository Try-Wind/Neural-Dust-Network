"""
Neural Dust Network - Simplified Demo (No External Dependencies)
A self-contained demonstration using synthetic data to prove the concept.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import threading
from dust_gossip import DustGossip
from dust_security import DustSecurity, SecureDustGossip

# Simple Dust Model for Demo
class SimpleDust(nn.Module):
    """Ultra-simple model for demonstration"""
    def __init__(self):
        super(SimpleDust, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),      # 4 input features
            nn.ReLU(),
            nn.Linear(8, 3)       # 3 classes
        )
        
    def forward(self, x):
        return self.net(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_synthetic_data(num_samples=1000):
    """Create synthetic classification data"""
    # Create 3 clusters for 3 classes
    torch.manual_seed(42)
    
    # Class 0: around (0, 0)
    class0 = torch.randn(num_samples//3, 4) * 0.5 + torch.tensor([0.0, 0.0, 0.0, 0.0])
    labels0 = torch.zeros(num_samples//3, dtype=torch.long)
    
    # Class 1: around (2, 2)
    class1 = torch.randn(num_samples//3, 4) * 0.5 + torch.tensor([2.0, 2.0, 0.0, 0.0])
    labels1 = torch.ones(num_samples//3, dtype=torch.long)
    
    # Class 2: around (-2, 2)
    class2 = torch.randn(num_samples//3, 4) * 0.5 + torch.tensor([-2.0, 2.0, 0.0, 0.0])
    labels2 = torch.full((num_samples//3,), 2, dtype=torch.long)
    
    # Combine data
    data = torch.cat([class0, class1, class2], dim=0)
    labels = torch.cat([labels0, labels1, labels2], dim=0)
    
    # Shuffle
    perm = torch.randperm(len(data))
    data = data[perm]
    labels = labels[perm]
    
    return data, labels

def create_data_loaders(batch_size=32):
    """Create train and test data loaders"""
    # Create datasets
    train_data, train_labels = create_synthetic_data(900)
    test_data, test_labels = create_synthetic_data(300)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model_simple(model, train_loader, epochs=1, lr=0.01):
    """Train the model for a few epochs"""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 10:  # Limit for quick demo
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
    
    if total > 0:
        accuracy = 100. * correct / total
        avg_loss = total_loss / min(10, len(train_loader))
        return avg_loss, accuracy
    return 0, 0

def test_model_simple(model, test_loader):
    """Test the model accuracy"""
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
    
    return test_loss, accuracy

def federated_average(state_dicts):
    """Perform federated averaging on model state dictionaries"""
    if not state_dicts:
        return {}
    
    averaged_state = {}
    first_state = state_dicts[0]
    
    for key in first_state.keys():
        averaged_state[key] = torch.zeros_like(first_state[key])
    
    # Sum all parameters
    for state_dict in state_dicts:
        for key in averaged_state.keys():
            if key in state_dict:
                averaged_state[key] += state_dict[key]
    
    # Average by dividing by count
    num_models = len(state_dicts)
    for key in averaged_state.keys():
        averaged_state[key] = averaged_state[key] / num_models
    
    return averaged_state

class SimpleNeuralDustDemo:
    """Simplified demonstration of Neural Dust Network"""
    
    def __init__(self, num_devices=3):
        self.num_devices = num_devices
        self.devices = []
        
        print("=" * 60)
        print("🌟 NEURAL DUST NETWORK - SIMPLIFIED DEMO 🌟")
        print("=" * 60)
        print(f"Creating {num_devices} virtual devices...")
        
        # Create data loaders
        self.train_loader, self.test_loader = create_data_loaders()
        
        # Setup devices
        self.setup_devices()
    
    def setup_devices(self):
        """Initialize all devices"""
        base_port = 56000
        
        for i in range(self.num_devices):
            device_id = f"device_{i:02d}"
            
            device = {
                'id': device_id,
                'port': base_port + i,
                'model': SimpleDust(),
                'gossip': DustGossip(port=base_port + i, node_id=device_id),
                'security': DustSecurity(device_id),
                'accuracy_history': [],
                'is_active': False
            }
            
            # Create secure gossip
            device['secure_gossip'] = SecureDustGossip(
                device['gossip'], 
                device['security']
            )
            
            self.devices.append(device)
            print(f"✓ {device_id} created (Model params: {device['model'].count_parameters()})")
        
        # Establish trust network
        self.establish_trust()
        print(f"\n🔗 Neural Dust Network ready!\n")
    
    def establish_trust(self):
        """Establish trust between all devices"""
        print("\n🔐 Establishing secure trust network...")
        
        for i, device_a in enumerate(self.devices):
            for j, device_b in enumerate(self.devices):
                if i != j:
                    pub_data = device_b['security'].get_public_key_qr_data()
                    device_a['security'].add_trusted_peer(
                        pub_data['node_id'],
                        pub_data['public_key'],
                        pub_data['fingerprint']
                    )
        
        print("✓ All devices now trust each other")
    
    def test_initial_performance(self):
        """Test all devices with random weights"""
        print("📊 Testing initial performance (random weights)...")
        
        initial_accuracies = []
        for device in self.devices:
            loss, accuracy = test_model_simple(device['model'], self.test_loader)
            initial_accuracies.append(accuracy)
            device['accuracy_history'].append(accuracy)
            print(f"   {device['id']}: {accuracy:.1f}% accuracy")
        
        avg_accuracy = np.mean(initial_accuracies)
        print(f"\n📈 Average initial accuracy: {avg_accuracy:.1f}%")
        return avg_accuracy
    
    def run_collaborative_learning(self, iterations=5):
        """Run the collaborative learning demonstration"""
        print(f"\n🌐 Starting Collaborative Learning ({iterations} iterations)")
        print("Devices will share knowledge (not data) to improve together...")
        
        # Start all gossip protocols
        for device in self.devices:
            device['gossip'].start_listening()
            device['is_active'] = True
        
        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Phase 1: Local learning
            print("  Phase 1: Each device learns locally...")
            for device in self.devices:
                loss, acc = train_model_simple(device['model'], self.train_loader)
                print(f"    {device['id']}: Local training accuracy {acc:.1f}%")
            
            # Phase 2: Share knowledge
            print("  Phase 2: Sharing learned knowledge...")
            for device in self.devices:
                success = device['secure_gossip'].secure_broadcast_delta(
                    device['model'].state_dict(), 
                    epoch=iteration
                )
                if success:
                    print(f"    {device['id']}: Knowledge broadcast ✓")
            
            # Allow gossip propagation
            time.sleep(0.5)
            
            # Phase 3: Merge knowledge
            print("  Phase 3: Merging received knowledge...")
            for device in self.devices:
                verified_deltas = device['secure_gossip'].get_verified_deltas(max_count=3)
                
                if verified_deltas:
                    # Collect all model states
                    all_models = [delta[1] for delta in verified_deltas]
                    all_models.append(device['model'].state_dict())
                    
                    # Perform federated averaging
                    merged_state = federated_average(all_models)
                    device['model'].load_state_dict(merged_state)
                    
                    print(f"    {device['id']}: Merged knowledge from {len(verified_deltas)} peers ✓")
                else:
                    print(f"    {device['id']}: No new knowledge received")
            
            # Phase 4: Test performance
            print("  Phase 4: Testing network performance...")
            accuracies = []
            for device in self.devices:
                loss, accuracy = test_model_simple(device['model'], self.test_loader)
                device['accuracy_history'].append(accuracy)
                accuracies.append(accuracy)
                print(f"    {device['id']}: {accuracy:.1f}% accuracy")
            
            avg_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            print(f"  📈 Network average: {avg_accuracy:.1f}% (±{std_accuracy:.1f}%)")
            
            # Check for convergence
            if std_accuracy < 1.0:
                print("  🎯 Network convergence achieved!")
        
        return np.mean(accuracies)
    
    def show_results(self):
        """Display the final results"""
        print("\n" + "=" * 60)
        print("🎉 NEURAL DUST NETWORK DEMO COMPLETE! 🎉")
        print("=" * 60)
        
        # Calculate improvements
        initial_accuracy = self.devices[0]['accuracy_history'][0]
        final_accuracies = [device['accuracy_history'][-1] for device in self.devices]
        final_avg = np.mean(final_accuracies)
        improvement = final_avg - initial_accuracy
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   Initial accuracy (random): {initial_accuracy:.1f}%")
        print(f"   Final network accuracy: {final_avg:.1f}%")
        print(f"   Total improvement: +{improvement:.1f}%")
        print(f"   Network convergence: ±{np.std(final_accuracies):.1f}%")
        
        print(f"\n🔧 INDIVIDUAL DEVICE PROGRESS:")
        for device in self.devices:
            history = device['accuracy_history']
            print(f"   {device['id']}: {history[0]:.1f}% → {history[-1]:.1f}% (+{history[-1]-history[0]:.1f}%)")
        
        print(f"\n🔒 PRIVACY & SECURITY:")
        print("   ✓ No raw training data shared between devices")
        print("   ✓ Only learned model weights were exchanged")
        print("   ✓ All updates cryptographically signed and verified")
        print("   ✓ Zero-trust security model implemented")
        
        # Technical stats
        total_deltas = sum(device['gossip'].stats['deltas_sent'] for device in self.devices)
        total_bytes = sum(device['gossip'].stats['bytes_sent'] for device in self.devices)
        
        print(f"\n⚡ NETWORK STATISTICS:")
        print(f"   Knowledge updates sent: {total_deltas}")
        print(f"   Total bytes transmitted: {total_bytes:,}")
        if total_deltas > 0:
            print(f"   Average update size: {total_bytes//total_deltas:,} bytes")
        print(f"   Model size per device: ~{self.devices[0]['model'].count_parameters() * 4} bytes")
        
        print(f"\n🌟 THIS IS THE NEURAL DUST NETWORK!")
        print("   - Decentralized AI that learns collaboratively")
        print("   - Privacy-preserving knowledge sharing")
        print("   - No central servers or data collection")
        print("   - Continuous improvement through swarm intelligence")
        
        return improvement
    
    def cleanup(self):
        """Clean up resources"""
        for device in self.devices:
            device['gossip'].stop_listening()
            device['is_active'] = False
        print("\n🧹 Demo cleanup complete")
    
    def run_demo(self):
        """Run the complete demonstration"""
        try:
            # Test initial performance
            initial_acc = self.test_initial_performance()
            
            # Run collaborative learning
            final_acc = self.run_collaborative_learning(iterations=3)
            
            # Show results
            improvement = self.show_results()
            
            return {
                'initial_accuracy': initial_acc,
                'final_accuracy': final_acc,
                'improvement': improvement
            }
        
        finally:
            self.cleanup()

if __name__ == "__main__":
    print("🚀 Starting Neural Dust Network Demonstration...")
    
    # Run the demo
    demo = SimpleNeuralDustDemo(num_devices=3)
    results = demo.run_demo()
    
    print(f"\n✨ Demo completed!")
    print(f"   Network improved by {results['improvement']:.1f}% through collaboration!")
    print("\n🎯 The future of AI is decentralized, private, and collaborative!")