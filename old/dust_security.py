"""
Neural Dust Network - Security Layer Implementation
Provides Ed25519 digital signatures for delta verification and trust management.
"""

import time
import hashlib
import json
import os
from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError
import nacl.encoding
import pickle
from datetime import datetime, timedelta

class DustSecurity:
    """
    Handles cryptographic security for Neural Dust Network.
    Uses Ed25519 signatures to verify delta authenticity and prevent tampering.
    """
    
    def __init__(self, node_id=None, key_file=None):
        self.node_id = node_id or "default_node"
        self.key_file = key_file or f"dust_keys_{self.node_id}.json"
        
        # Load or generate keypair
        self.signing_key, self.verify_key = self._load_or_generate_keys()
        
        # Trust management
        self.trusted_peers = {}  # peer_id -> verify_key
        self.trust_file = f"dust_trust_{self.node_id}.json"
        self._load_trusted_peers()
        
        # Anti-replay protection
        self.seen_signatures = {}  # signature -> timestamp
        self.signature_ttl = 3600  # 1 hour TTL for signatures
        
        print(f"DustSecurity initialized for node {self.node_id}")
        print(f"Public key: {self.get_public_key_fingerprint()}")
    
    def _load_or_generate_keys(self):
        """Load existing keypair or generate new one"""
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, 'r') as f:
                    key_data = json.load(f)
                
                private_key_bytes = bytes.fromhex(key_data['private_key'])
                signing_key = SigningKey(private_key_bytes)
                verify_key = signing_key.verify_key
                
                print(f"Loaded existing keypair from {self.key_file}")
                return signing_key, verify_key
                
            except Exception as e:
                print(f"Error loading keys: {e}, generating new keypair")
        
        # Generate new keypair
        signing_key = SigningKey.generate()
        verify_key = signing_key.verify_key
        
        # Save keypair
        key_data = {
            'node_id': self.node_id,
            'private_key': signing_key.encode().hex(),
            'public_key': verify_key.encode().hex(),
            'created': datetime.now().isoformat()
        }
        
        with open(self.key_file, 'w') as f:
            json.dump(key_data, f, indent=2)
        
        print(f"Generated new keypair and saved to {self.key_file}")
        return signing_key, verify_key
    
    def _load_trusted_peers(self):
        """Load trusted peer public keys"""
        if os.path.exists(self.trust_file):
            try:
                with open(self.trust_file, 'r') as f:
                    trust_data = json.load(f)
                    
                for peer_id, peer_info in trust_data.items():
                    public_key_bytes = bytes.fromhex(peer_info['public_key'])
                    verify_key = VerifyKey(public_key_bytes)
                    self.trusted_peers[peer_id] = {
                        'verify_key': verify_key,
                        'added': peer_info.get('added'),
                        'fingerprint': peer_info.get('fingerprint')
                    }
                
                print(f"Loaded {len(self.trusted_peers)} trusted peers")
                
            except Exception as e:
                print(f"Error loading trusted peers: {e}")
    
    def _save_trusted_peers(self):
        """Save trusted peer public keys"""
        trust_data = {}
        for peer_id, peer_info in self.trusted_peers.items():
            trust_data[peer_id] = {
                'public_key': peer_info['verify_key'].encode().hex(),
                'added': peer_info.get('added', datetime.now().isoformat()),
                'fingerprint': peer_info.get('fingerprint')
            }
        
        with open(self.trust_file, 'w') as f:
            json.dump(trust_data, f, indent=2)
    
    def get_public_key_fingerprint(self):
        """Get a human-readable fingerprint of our public key"""
        public_key_bytes = self.verify_key.encode()
        fingerprint = hashlib.sha256(public_key_bytes).hexdigest()[:16]
        return fingerprint
    
    def get_public_key_qr_data(self):
        """Get public key data suitable for QR code sharing"""
        return {
            'node_id': self.node_id,
            'public_key': self.verify_key.encode().hex(),
            'fingerprint': self.get_public_key_fingerprint()
        }
    
    def add_trusted_peer(self, peer_id, public_key_hex, fingerprint=None):
        """Add a trusted peer's public key"""
        try:
            public_key_bytes = bytes.fromhex(public_key_hex)
            verify_key = VerifyKey(public_key_bytes)
            
            # Verify fingerprint if provided
            if fingerprint:
                calculated_fingerprint = hashlib.sha256(public_key_bytes).hexdigest()[:16]
                if calculated_fingerprint != fingerprint:
                    raise ValueError("Fingerprint mismatch")
            
            self.trusted_peers[peer_id] = {
                'verify_key': verify_key,
                'added': datetime.now().isoformat(),
                'fingerprint': hashlib.sha256(public_key_bytes).hexdigest()[:16]
            }
            
            self._save_trusted_peers()
            print(f"Added trusted peer: {peer_id}")
            return True
            
        except Exception as e:
            print(f"Error adding trusted peer: {e}")
            return False
    
    def remove_trusted_peer(self, peer_id):
        """Remove a trusted peer"""
        if peer_id in self.trusted_peers:
            del self.trusted_peers[peer_id]
            self._save_trusted_peers()
            print(f"Removed trusted peer: {peer_id}")
            return True
        return False
    
    def sign_delta(self, delta_data, epoch=None, additional_data=None):
        """
        Sign a model delta with our private key.
        
        Args:
            delta_data: Serialized model weights
            epoch: Optional epoch number
            additional_data: Optional additional metadata
        
        Returns:
            dict: Signed packet with metadata
        """
        timestamp = time.time()
        
        # Create message to sign
        message_data = {
            'node_id': self.node_id,
            'timestamp': timestamp,
            'epoch': epoch or int(timestamp),
            'delta_hash': hashlib.sha256(delta_data).hexdigest(),
            'additional_data': additional_data
        }
        
        message_bytes = json.dumps(message_data, sort_keys=True).encode()
        
        # Sign the message
        signed_message = self.signing_key.sign(message_bytes)
        signature = signed_message.signature
        
        # Create signed packet
        signed_packet = {
            'message': message_data,
            'signature': signature.hex(),
            'public_key': self.verify_key.encode().hex(),
            'delta': delta_data
        }
        
        return signed_packet
    
    def verify_delta(self, signed_packet, require_trusted=True):
        """
        Verify a signed delta packet.
        
        Args:
            signed_packet: Signed packet to verify
            require_trusted: Whether to require the sender to be in our trust list
        
        Returns:
            tuple: (is_valid, node_id, delta_data)
        """
        try:
            message = signed_packet['message']
            signature_hex = signed_packet['signature']
            public_key_hex = signed_packet['public_key']
            delta_data = signed_packet['delta']
            
            node_id = message['node_id']
            timestamp = message['timestamp']
            delta_hash = message['delta_hash']
            
            # Check if signature is too old or replayed
            signature_key = f"{node_id}_{signature_hex}"
            if signature_key in self.seen_signatures:
                print(f"Replay attack detected from {node_id}")
                return False, None, None
            
            # Check timestamp (reject packets older than 1 hour)
            if time.time() - timestamp > 3600:
                print(f"Packet from {node_id} is too old")
                return False, None, None
            
            # Verify delta hash
            calculated_hash = hashlib.sha256(delta_data).hexdigest()
            if calculated_hash != delta_hash:
                print(f"Delta hash mismatch from {node_id}")
                return False, None, None
            
            # Get verify key
            if require_trusted and node_id not in self.trusted_peers:
                print(f"Untrusted peer {node_id}, rejecting delta")
                return False, None, None
            
            if node_id in self.trusted_peers:
                verify_key = self.trusted_peers[node_id]['verify_key']
            else:
                # Use provided public key (if not requiring trust)
                public_key_bytes = bytes.fromhex(public_key_hex)
                verify_key = VerifyKey(public_key_bytes)
            
            # Verify signature
            message_bytes = json.dumps(message, sort_keys=True).encode()
            signature_bytes = bytes.fromhex(signature_hex)
            
            verify_key.verify(message_bytes, signature_bytes)
            
            # Mark signature as seen (anti-replay)
            self.seen_signatures[signature_key] = timestamp
            
            # Clean old signatures periodically
            self._cleanup_old_signatures()
            
            print(f"Verified delta from trusted peer {node_id}")
            return True, node_id, delta_data
            
        except BadSignatureError:
            print(f"Invalid signature from {node_id}")
            return False, None, None
        except Exception as e:
            print(f"Error verifying delta: {e}")
            return False, None, None
    
    def _cleanup_old_signatures(self):
        """Remove old signatures to prevent memory bloat"""
        current_time = time.time()
        cutoff_time = current_time - self.signature_ttl
        
        # Remove signatures older than TTL
        old_sigs = [sig for sig, timestamp in self.seen_signatures.items() 
                   if timestamp < cutoff_time]
        
        for sig in old_sigs:
            del self.seen_signatures[sig]
    
    def get_trust_stats(self):
        """Get trust and security statistics"""
        return {
            'node_id': self.node_id,
            'public_key_fingerprint': self.get_public_key_fingerprint(),
            'trusted_peers_count': len(self.trusted_peers),
            'trusted_peers': list(self.trusted_peers.keys()),
            'signatures_seen': len(self.seen_signatures)
        }
    
    def export_trust_bundle(self):
        """Export trust data for sharing"""
        return {
            'node_id': self.node_id,
            'public_key': self.verify_key.encode().hex(),
            'fingerprint': self.get_public_key_fingerprint(),
            'trusted_peers': [
                {
                    'peer_id': peer_id,
                    'fingerprint': peer_info['fingerprint']
                }
                for peer_id, peer_info in self.trusted_peers.items()
            ]
        }

class SecureDustGossip:
    """
    Secure version of DustGossip with cryptographic verification.
    """
    
    def __init__(self, gossip_instance, security_instance):
        self.gossip = gossip_instance
        self.security = security_instance
    
    def secure_broadcast_delta(self, model_state_dict, epoch=None):
        """Broadcast a cryptographically signed model delta"""
        # Serialize the model weights
        delta_data = pickle.dumps(model_state_dict)
        
        # Sign the delta
        signed_packet = self.security.sign_delta(delta_data, epoch)
        
        # Broadcast the signed packet
        return self.gossip.broadcast_delta(signed_packet, epoch)
    
    def get_verified_deltas(self, max_count=None, require_trusted=True):
        """Get and verify received deltas"""
        # Get raw deltas
        raw_deltas = self.gossip.get_received_deltas(max_count, cleanup=False)
        
        verified_deltas = []
        for filepath, signed_packet in raw_deltas:
            # Verify the signature
            is_valid, node_id, delta_data = self.security.verify_delta(
                signed_packet, require_trusted
            )
            
            if is_valid:
                # Deserialize the model weights
                model_state_dict = pickle.loads(delta_data)
                verified_deltas.append((node_id, model_state_dict))
                
                # Remove processed file
                try:
                    os.remove(filepath)
                except:
                    pass
            else:
                print(f"Rejected invalid/untrusted delta from {filepath}")
        
        return verified_deltas

if __name__ == "__main__":
    print("=== Neural Dust Network - Security Layer Demo ===")
    
    # Create security instances for two nodes
    node1_security = DustSecurity("node_001")
    node2_security = DustSecurity("node_002")
    
    # Simulate trust establishment (normally done via QR codes)
    node1_public_data = node1_security.get_public_key_qr_data()
    node2_public_data = node2_security.get_public_key_qr_data()
    
    # Add each other as trusted peers
    node1_security.add_trusted_peer(
        node2_public_data['node_id'],
        node2_public_data['public_key'],
        node2_public_data['fingerprint']
    )
    
    node2_security.add_trusted_peer(
        node1_public_data['node_id'],
        node1_public_data['public_key'],
        node1_public_data['fingerprint']
    )
    
    # Test signing and verification
    test_data = {'test_weights': [1.0, 2.0, 3.0]}
    serialized_data = pickle.dumps(test_data)
    
    # Node 1 signs data
    signed_packet = node1_security.sign_delta(serialized_data, epoch=1)
    print("✓ Delta signed by node 1")
    
    # Node 2 verifies data
    is_valid, sender_id, verified_data = node2_security.verify_delta(signed_packet)
    
    if is_valid:
        print(f"✓ Delta verified from {sender_id}")
        reconstructed_data = pickle.loads(verified_data)
        print(f"✓ Data integrity confirmed: {reconstructed_data}")
    else:
        print("✗ Delta verification failed")
    
    # Show trust stats
    print(f"\nNode 1 trust stats: {node1_security.get_trust_stats()}")
    print(f"Node 2 trust stats: {node2_security.get_trust_stats()}")
    
    print("\n=== Security layer demo completed! ===")