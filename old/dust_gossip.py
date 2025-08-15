"""
Neural Dust Network - Gossip Protocol Implementation
Handles broadcasting and receiving of model weight deltas between devices.
"""

import socket
import threading
import time
import pickle
import lz4.frame
import os
import glob
import json
from datetime import datetime
import hashlib
import struct

class DustGossip:
    """
    Handles the gossip protocol for Neural Dust Network.
    Broadcasts and receives compressed model weight deltas.
    """
    
    def __init__(self, port=54545, node_id=None, max_delta_size=4096):
        self.port = port
        self.node_id = node_id or self._generate_node_id()
        self.max_delta_size = max_delta_size
        self.delta_dir = f"deltas_{self.node_id}"
        self.is_listening = False
        self.listen_thread = None
        self.stats = {
            'deltas_sent': 0,
            'deltas_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'last_activity': None
        }
        
        # Ensure delta directory exists
        os.makedirs(self.delta_dir, exist_ok=True)
        
        print(f"DustGossip initialized - Node ID: {self.node_id}, Port: {self.port}")
    
    def _generate_node_id(self):
        """Generate a unique node identifier"""
        return hashlib.md5(f"{time.time()}_{os.getpid()}".encode()).hexdigest()[:8]
    
    def _create_delta_packet(self, delta_data, epoch=None):
        """Create a gossip packet with metadata"""
        packet = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'epoch': epoch or int(time.time()),
            'delta': delta_data,
            'size': len(delta_data)
        }
        return packet
    
    def broadcast_delta(self, model_state_dict, epoch=None):
        """
        Broadcast model weight delta to all devices on the network.
        
        Args:
            model_state_dict: PyTorch model state dictionary
            epoch: Optional epoch number for ordering
        """
        try:
            # Serialize the model weights
            delta_data = pickle.dumps(model_state_dict)
            
            # Check size limit
            if len(delta_data) > self.max_delta_size:
                print(f"WARNING: Delta size ({len(delta_data)} bytes) exceeds limit ({self.max_delta_size} bytes)")
                return False
            
            # Compress the data
            compressed_delta = lz4.frame.compress(delta_data)
            
            # Create packet with metadata
            packet = self._create_delta_packet(compressed_delta, epoch)
            packet_data = pickle.dumps(packet)
            
            # Broadcast via UDP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                sock.sendto(packet_data, ('<broadcast>', self.port))
                
                # Update statistics
                self.stats['deltas_sent'] += 1
                self.stats['bytes_sent'] += len(packet_data)
                self.stats['last_activity'] = datetime.now().isoformat()
                
                compression_ratio = len(delta_data) / len(compressed_delta)
                print(f"Delta broadcast: {len(compressed_delta)} bytes "
                      f"(compression: {compression_ratio:.1f}x, epoch: {epoch})")
                
                return True
                
            except Exception as e:
                print(f"Broadcast failed: {e}")
                return False
            finally:
                sock.close()
                
        except Exception as e:
            print(f"Error creating delta: {e}")
            return False
    
    def listen_deltas(self):
        """
        Listen for incoming model deltas from other devices.
        Runs in a separate thread.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind(('', self.port))
            print(f"Listening for deltas on port {self.port}...")
            self.is_listening = True
            
            while self.is_listening:
                try:
                    # Set timeout to allow clean shutdown
                    sock.settimeout(1.0)
                    data, addr = sock.recvfrom(8192)  # Larger buffer for metadata
                    
                    # Ignore our own broadcasts
                    if addr[0] == socket.gethostbyname(socket.gethostname()):
                        continue
                    
                    # Unpack the packet
                    packet = pickle.loads(data)
                    
                    # Ignore packets from ourselves
                    if packet.get('node_id') == self.node_id:
                        continue
                    
                    # Decompress the delta
                    compressed_delta = packet['delta']
                    delta_data = lz4.frame.decompress(compressed_delta)
                    
                    # Validate size
                    if len(delta_data) > self.max_delta_size:
                        print(f"WARNING: Received oversized delta ({len(delta_data)} bytes), ignoring")
                        continue
                    
                    # Save delta to file
                    timestamp = int(time.time() * 1000)
                    node_id = packet.get('node_id', 'unknown')
                    epoch = packet.get('epoch', timestamp)
                    
                    filename = f"{self.delta_dir}/delta_{epoch}_{node_id}_{timestamp}.pkl"
                    
                    with open(filename, 'wb') as f:
                        f.write(delta_data)
                    
                    # Update statistics
                    self.stats['deltas_received'] += 1
                    self.stats['bytes_received'] += len(data)
                    self.stats['last_activity'] = datetime.now().isoformat()
                    
                    print(f"Received delta from {node_id} ({addr[0]}): {len(delta_data)} bytes, epoch {epoch}")
                    
                except socket.timeout:
                    # Normal timeout, continue listening
                    continue
                except Exception as e:
                    if self.is_listening:  # Only log if we're supposed to be listening
                        print(f"Error receiving delta: {e}")
                    
        except Exception as e:
            print(f"Listen socket error: {e}")
        finally:
            sock.close()
            print("Stopped listening for deltas")
    
    def start_listening(self):
        """Start listening for deltas in a background thread"""
        if not self.is_listening:
            self.listen_thread = threading.Thread(target=self.listen_deltas, daemon=True)
            self.listen_thread.start()
            time.sleep(0.1)  # Give thread time to start
    
    def stop_listening(self):
        """Stop listening for deltas"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
    
    def get_received_deltas(self, max_count=None, cleanup=True):
        """
        Get list of received delta files.
        
        Args:
            max_count: Maximum number of deltas to return (None for all)
            cleanup: Whether to remove old delta files after reading
        
        Returns:
            List of (filename, model_state_dict) tuples
        """
        pattern = f"{self.delta_dir}/delta_*.pkl"
        files = sorted(glob.glob(pattern))
        
        if max_count:
            files = files[-max_count:]  # Get most recent
        
        deltas = []
        for filepath in files:
            try:
                with open(filepath, 'rb') as f:
                    state_dict = pickle.load(f)
                deltas.append((filepath, state_dict))
            except Exception as e:
                print(f"Error loading delta {filepath}: {e}")
        
        # Clean up processed files
        if cleanup:
            for filepath, _ in deltas:
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error removing {filepath}: {e}")
        
        return deltas
    
    def get_stats(self):
        """Get gossip protocol statistics"""
        return self.stats.copy()
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()
        # Optionally clean up delta directory
        try:
            import shutil
            if os.path.exists(self.delta_dir):
                shutil.rmtree(self.delta_dir)
        except Exception as e:
            print(f"Cleanup error: {e}")

# Utility functions for easy use
def start_gossip(port=54545, node_id=None):
    """Create and start a gossip instance"""
    gossip = DustGossip(port=port, node_id=node_id)
    gossip.start_listening()
    return gossip

def broadcast_model_weights(model, gossip_instance, epoch=None):
    """Convenience function to broadcast model weights"""
    return gossip_instance.broadcast_delta(model.state_dict(), epoch)

if __name__ == "__main__":
    # Demo: Start gossip protocol
    print("=== Neural Dust Network - Gossip Protocol Demo ===")
    
    # Create gossip instance
    gossip = start_gossip()
    
    try:
        # Simulate sending some test data
        test_data = {'test_param': [1, 2, 3, 4, 5]}
        
        print(f"\nBroadcasting test delta...")
        success = gossip.broadcast_delta(test_data, epoch=1)
        
        if success:
            print("Test broadcast successful!")
        
        # Listen for a few seconds
        print("Listening for deltas for 5 seconds...")
        time.sleep(5)
        
        # Check received deltas
        deltas = gossip.get_received_deltas()
        print(f"Received {len(deltas)} deltas")
        
        # Show stats
        stats = gossip.get_stats()
        print(f"\nGossip Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    finally:
        gossip.cleanup()
        print("\nGossip protocol demo completed!")