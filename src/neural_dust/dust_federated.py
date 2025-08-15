"""
Neural Dust Network - Federated Averaging Implementation
Merges model weight deltas from multiple devices without a central server.
"""

import torch
import torch.nn as nn
import copy
import time
import threading
from collections import defaultdict
import statistics
import numpy as np
from .dust_model import Dust, create_dust_model
from .dust_gossip import DustGossip

class FederatedDustNode:
    """
    A node in the Neural Dust Network that can learn locally and merge knowledge
    from other nodes through federated averaging.
    """
    
    def __init__(self, node_id=None, port=54545, merge_interval=60):
        self.node_id = node_id
        self.port = port
        self.merge_interval = merge_interval
        
        # Initialize components
        self.model = create_dust_model()
        self.gossip = DustGossip(port=port, node_id=node_id)
        
        # Federated learning state
        self.current_epoch = 0
        self.last_merge_time = time.time()
        self.merge_history = []
        self.is_running = False
        self.merge_thread = None
        
        # Statistics
        self.stats = {
            'local_training_rounds': 0,
            'successful_merges': 0,
            'total_peers_seen': set(),
            'model_accuracy_history': [],
            'merge_timestamps': []
        }
        
        print(f"FederatedDustNode initialized - ID: {self.gossip.node_id}")
    
    def merge_deltas(self, max_deltas=3, min_deltas=1):
        """
        Merge received model deltas using federated averaging.
        
        Args:
            max_deltas: Maximum number of deltas to merge at once
            min_deltas: Minimum number of deltas required for merging
        
        Returns:
            bool: True if merge was successful
        """
        # Get received deltas
        deltas = self.gossip.get_received_deltas(max_count=max_deltas, cleanup=True)
        
        if len(deltas) < min_deltas:
            return False
        
        print(f"Merging {len(deltas)} model deltas...")
        
        try:
            # Extract state dictionaries
            state_dicts = [delta[1] for delta in deltas]
            
            # Add our current model to the averaging
            state_dicts.append(self.model.state_dict())
            
            # Perform federated averaging
            merged_state = self._federated_average(state_dicts)
            
            # Load merged weights into our model
            self.model.load_state_dict(merged_state)
            
            # Update statistics
            self.stats['successful_merges'] += 1
            self.stats['merge_timestamps'].append(time.time())
            
            # Track unique peers
            for filepath, _ in deltas:
                # Extract node_id from filename
                parts = filepath.split('_')
                if len(parts) >= 4:
                    peer_id = parts[3]
                    self.stats['total_peers_seen'].add(peer_id)
            
            self.current_epoch += 1
            print(f"Merge completed successfully (epoch {self.current_epoch})")
            
            return True
            
        except Exception as e:
            print(f"Error during merge: {e}")
            return False
    
    def _federated_average(self, state_dicts):
        """
        Perform federated averaging on multiple model state dictionaries.
        
        Args:
            state_dicts: List of PyTorch state dictionaries
        
        Returns:
            dict: Averaged state dictionary
        """
        if not state_dicts:
            return {}
        
        # Initialize averaged state with zeros
        averaged_state = {}
        first_state = state_dicts[0]
        
        for key in first_state.keys():
            # Initialize with zeros of the same shape and dtype
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
    
    def local_training_step(self, train_loader, num_batches=5):
        """
        Perform local training on the model.
        
        Args:
            train_loader: DataLoader for training data
            num_batches: Number of batches to train on
        """
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
                
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        if total > 0:
            accuracy = 100. * correct / total
            avg_loss = total_loss / min(num_batches, len(train_loader))
            
            self.stats['local_training_rounds'] += 1
            self.stats['model_accuracy_history'].append(accuracy)
            
            print(f"Local training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            return avg_loss, accuracy
        
        return 0, 0
    
    def broadcast_weights(self):
        """Broadcast current model weights to the network"""
        success = self.gossip.broadcast_delta(self.model.state_dict(), self.current_epoch)
        if success:
            print(f"Broadcasted model weights (epoch {self.current_epoch})")
        return success
    
    def continuous_learning_loop(self, train_loader):
        """
        Main loop for continuous federated learning.
        
        Args:
            train_loader: DataLoader for training data
        """
        print(f"Starting continuous learning loop (merge interval: {self.merge_interval}s)")
        
        while self.is_running:
            try:
                # Local training
                self.local_training_step(train_loader, num_batches=3)
                
                # Broadcast our updates
                self.broadcast_weights()
                
                # Check if it's time to merge
                if time.time() - self.last_merge_time >= self.merge_interval:
                    self.merge_deltas(max_deltas=3, min_deltas=1)
                    self.last_merge_time = time.time()
                
                # Short sleep to prevent overwhelming the network
                time.sleep(2)
                
            except Exception as e:
                print(f"Error in learning loop: {e}")
                time.sleep(5)  # Longer sleep on error
    
    def start_federated_learning(self, train_loader):
        """Start the federated learning process"""
        if not self.is_running:
            self.is_running = True
            
            # Start gossip protocol
            self.gossip.start_listening()
            
            # Start continuous learning in background thread
            self.merge_thread = threading.Thread(
                target=self.continuous_learning_loop, 
                args=(train_loader,), 
                daemon=True
            )
            self.merge_thread.start()
            
            print("Federated learning started!")
    
    def stop_federated_learning(self):
        """Stop the federated learning process"""
        self.is_running = False
        
        if self.merge_thread:
            self.merge_thread.join(timeout=5)
        
        self.gossip.stop_listening()
        print("Federated learning stopped!")
    
    def get_model_state(self):
        """Get current model and its statistics"""
        return {
            'model': self.model,
            'epoch': self.current_epoch,
            'node_id': self.gossip.node_id,
            'stats': self.stats.copy()
        }
    
    def evaluate_model(self, test_loader):
        """Evaluate current model performance"""
        self.model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        return test_loss, accuracy
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_federated_learning()
        self.gossip.cleanup()

class DustSwarm:
    """
    Manages multiple DustNodes to simulate a swarm of learning devices.
    """
    
    def __init__(self, num_nodes=3, base_port=54545):
        self.nodes = []
        self.base_port = base_port
        
        # Create nodes with different ports
        for i in range(num_nodes):
            node = FederatedDustNode(
                node_id=f"node_{i:02d}",
                port=base_port + i,
                merge_interval=30  # Faster merging for demo
            )
            self.nodes.append(node)
        
        print(f"DustSwarm created with {num_nodes} nodes")
    
    def start_swarm_learning(self, train_loader):
        """Start federated learning on all nodes"""
        for node in self.nodes:
            node.start_federated_learning(train_loader)
        print("Swarm learning started!")
    
    def stop_swarm_learning(self):
        """Stop federated learning on all nodes"""
        for node in self.nodes:
            node.stop_federated_learning()
        print("Swarm learning stopped!")
    
    def get_swarm_stats(self):
        """Get statistics from all nodes"""
        stats = {}
        for i, node in enumerate(self.nodes):
            stats[f"node_{i}"] = node.get_model_state()
        return stats
    
    def cleanup(self):
        """Clean up all nodes"""
        for node in self.nodes:
            node.cleanup()

if __name__ == "__main__":
    print("=== Neural Dust Network - Federated Learning Demo ===")
    
    # This is a placeholder for a real demo
    # The actual demo will be in the main demo script
    print("Use dust_demo.py for a complete demonstration")