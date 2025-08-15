"""
Neural Dust Network - Decentralized AI Learning Platform
========================================================

The Neural Dust Network (NDN) enables collaborative learning across devices
without sharing raw data. Only learned knowledge (model weight deltas) are
exchanged, ensuring privacy while improving collective intelligence.

Core Components:
    - DustModel: Ultra-compact neural networks (≤100 kB)
    - DustGossip: Peer-to-peer weight delta sharing
    - FederatedDustNode: Decentralized learning coordination
    - DustSecurity: Cryptographic verification and trust

Example Usage:
    >>> from neural_dust import DustModel, DustGossip, DustSecurity
    >>> model = DustModel()
    >>> gossip = DustGossip(port=55000)
    >>> security = DustSecurity("my_device")
    >>> # Start collaborative learning...

Authors: Neural Dust Network Contributors
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Neural Dust Network Contributors"
__license__ = "MIT"

# Core model components
from .dust_model import (
    Dust,
    create_dust_model,
    save_dust_model,
    load_dust_model,
    get_mnist_data,
    train_dust_model,
    test_dust_model
)

# Gossip protocol for weight sharing
from .dust_gossip import (
    DustGossip,
    start_gossip,
    broadcast_model_weights
)

# Federated learning coordination
from .dust_federated import (
    FederatedDustNode,
    DustSwarm
)

# Security and trust management
from .dust_security import (
    DustSecurity,
    SecureDustGossip
)

__all__ = [
    # Model components
    'Dust',
    'create_dust_model',
    'save_dust_model', 
    'load_dust_model',
    'get_mnist_data',
    'train_dust_model',
    'test_dust_model',
    
    # Gossip protocol
    'DustGossip',
    'start_gossip',
    'broadcast_model_weights',
    
    # Federated learning
    'FederatedDustNode',
    'DustSwarm',
    
    # Security
    'DustSecurity',
    'SecureDustGossip'
]