# Quick Start Guide

## For Busy Developers Who Just Want to Try It

### Option 1: Run Locally (No GPU needed for testing)

```bash
# Clone this repo
git clone https://github.com/reem-sab/nebius-pytorch-migration
cd nebius-pytorch-migration

# Install dependencies
pip install -r requirements.txt

# Run training (will use CPU if no GPU available)
python train_mnist.py --epochs 1
```

Expected output:
```
🚀 MNIST Training on CPU
📥 Loading MNIST dataset...
✅ Dataset loaded: 60,000 training samples
🔥 Starting Training
...
✅ Training Complete! Test Accuracy: 95.xx%
```

---

### Option 2: Run on Nebius GPU (RECOMMENDED)

**Step 1: Sign up for Nebius (2 minutes)**
- Go to https://nebius.com
- Click "Get Started"
- Enter email, verify

**Step 2: Create GPU instance (2 minutes)**
- Log into console
- Click "Compute" → "Create Instance"
- Select smallest GPU (A100)
- Click "Create"
- Copy the SSH command shown

**Step 3: Connect and train (3 minutes)**
```bash
# On your local machine, SSH to Nebius
ssh -i ~/.ssh/nebius_key ubuntu@<your-ip>

# Clone the repo
git clone https://github.com/reem-sab/nebius-pytorch-migration
cd nebius-pytorch-migration

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Run training on GPU
python train_mnist.py

# Expected output:
# 🚀 MNIST Training on CUDA
# GPU Device: NVIDIA A100-SXM4-40GB
# ...
# ✅ Training Complete! Test Accuracy: 98.xx%
# Total Time: ~40s for 3 epochs
```

**Step 4: Stop instance to avoid charges**
- Go back to Nebius console
- Click your instance
- Click "Stop" or "Delete"

**Total cost: Less than $0.50 for the experiment**

---

### Option 3: Compare AWS vs Nebius

Run on BOTH platforms and compare:

**On AWS SageMaker:**
```python
# Create SageMaker notebook
# Upload train_mnist.py
# Run: !python train_mnist.py
# Note the time and cost
```

**On Nebius:**
```bash
# Follow Option 2 above
# Note the time and cost
```

**Create comparison table:**
| Metric | AWS | Nebius |
|--------|-----|--------|
| Setup time | ? min | ? min |
| Training time | ? sec | ? sec |
| Cost | $? | $? |
| Complexity | ? | ? |

Share your results! Tweet at @Nebius or open an issue in this repo.

---

## Troubleshooting

**"No module named 'torch'"**
```bash
pip install torch torchvision --break-system-packages
```

**"CUDA out of memory"**
```bash
# Reduce batch size
python train_mnist.py --batch-size 64
```

**"SSH connection refused"**
- Check instance is running in Nebius console
- Verify your SSH key is correct
- Wait 1-2 minutes after instance creation

**Still stuck?**
- Open an issue: https://github.com/reem-sab/nebius-pytorch-migration/issues
- Email: sabawir@gmail.com
