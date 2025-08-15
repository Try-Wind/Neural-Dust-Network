# Neural Dust Network - Examples 🚀

This directory contains example implementations and demonstrations of the Neural Dust Network.

## 📁 Examples Overview

### `basic_demo.py` - Quick Demonstration
**Perfect for: First-time users, investors, demos**

```bash
python examples/basic_demo.py
```

**What it demonstrates:**
- 3 simulated devices learning collaboratively
- Privacy-preserving weight sharing via gossip protocol
- Ed25519 cryptographic security
- 64.8% accuracy improvement through collaboration
- Zero raw data sharing

**Runtime:** ~30 seconds  
**Output:** Real-time collaborative learning with statistics

---

### `advanced_demo.py` - Full MNIST Demo
**Perfect for: Researchers, developers, technical evaluation**

```bash
python examples/advanced_demo.py
```

**What it demonstrates:**
- Full MNIST digit classification
- Complete federated learning cycle
- Advanced security features
- Performance benchmarking
- Network convergence analysis

**Runtime:** ~2-3 minutes  
**Output:** Detailed performance metrics and analysis

---

## 🎯 Running the Examples

### Prerequisites
```bash
# Install the Neural Dust Network package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Quick Start
```bash
# Basic demo (recommended first run)
python examples/basic_demo.py

# Advanced demo with MNIST
python examples/advanced_demo.py

# Command-line interface
neural-dust-demo
```

## 📊 Expected Results

### Basic Demo Output
```
🌟 NEURAL DUST NETWORK - SIMPLIFIED DEMO 🌟
Creating 3 virtual devices...
✓ device_00 created (Model params: 67)
✓ device_01 created (Model params: 67)  
✓ device_02 created (Model params: 67)

📊 RESULTS SUMMARY:
   Initial accuracy (random): 14.3%
   Final network accuracy: 79.1%
   Total improvement: +64.8%
   Network convergence: ±8.3%

⚡ NETWORK STATISTICS:
   Knowledge updates sent: 9
   Total bytes transmitted: 12,730
   Average update size: 1,414 bytes
```

## 🔧 Customizing the Examples

### Changing the Number of Devices
```python
# In basic_demo.py, modify:
demo = SimpleNeuralDustDemo(num_devices=5)  # Default: 3
```

### Adjusting Security Settings
```python
# Enable/disable security features:
demo = SimpleNeuralDustDemo(secure_mode=False)  # Default: True
```

### Network Configuration
```python
# Change base port for testing multiple networks:
device['port'] = 56000 + i  # Default range: 56000-56002
```

## 🌐 Real Device Testing

To test across actual devices on the same network:

1. **Modify IP Configuration:**
```python
# In dust_gossip.py, change broadcast address:
sock.sendto(packet_data, ('192.168.1.255', self.port))  # Your subnet
```

2. **Run on Multiple Machines:**
```bash
# Device 1:
python -c "from examples.basic_demo import SimpleNeuralDustDemo; SimpleNeuralDustDemo(num_devices=1).run_demo()"

# Device 2:  
python -c "from examples.basic_demo import SimpleNeuralDustDemo; SimpleNeuralDustDemo(num_devices=1).run_demo()"
```

3. **Monitor Network Traffic:**
```bash
# Optional: Monitor UDP traffic
sudo tcpdump -i any port 56000
```

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Check for conflicting processes:
lsof -i :56000

# Kill conflicting processes:
sudo kill -9 <PID>
```

**Firewall Blocking UDP:**
```bash
# Allow UDP traffic (Linux):
sudo ufw allow 56000:56002/udp

# Allow UDP traffic (macOS):
sudo pfctl -d  # Disable firewall temporarily
```

**Import Errors:**
```bash
# Ensure package is installed:
pip install -e .

# Or add to Python path:
export PYTHONPATH="${PYTHONPATH}:/path/to/neural-dust-network/src"
```

## 📈 Performance Optimization

### For Faster Convergence:
- Reduce `merge_interval` in FederatedDustNode
- Increase number of local training batches
- Use larger models (more parameters)

### For Lower Resource Usage:
- Decrease model size in `Dust` class
- Reduce gossip frequency
- Limit number of peers for averaging

## 🎪 Demo Scenarios

### Scenario 1: "Coffee Shop Demo"
**Goal:** Show collaborative learning in a public setting
```python
demo = SimpleNeuralDustDemo(num_devices=2)
demo.run_demo()  # Quick 30-second demo
```

### Scenario 2: "Technical Deep Dive"
**Goal:** Demonstrate full capabilities to technical audience
```python
# Run advanced demo with detailed logging
demo = AdvancedNeuralDustDemo(num_devices=5, verbose=True)
demo.run_full_analysis()
```

### Scenario 3: "Investor Pitch"
**Goal:** Show the "magic moment" of devices getting smarter together
```python
# Focus on the dramatic accuracy improvement
demo.test_initial_performance()     # Show random performance
demo.run_collaborative_learning()   # Show improvement
demo.show_results()                 # Highlight 64.8% gain
```

---

**💡 Tip:** Start with `basic_demo.py` to understand the core concepts, then explore `advanced_demo.py` for deeper technical insights!