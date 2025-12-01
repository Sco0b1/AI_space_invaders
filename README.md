This is a Rainbow DQN (Deep Q-Network) implementation for playing Atari Space Invaders, created as what appears to be a university project.
What it does:
Trains an AI agent to play Space Invaders using deep reinforcement learning. The agent learns by playing the game repeatedly, getting better over time through trial and error.
Key Components:
Rainbow DQN - An advanced version of DQN that combines 6 improvements:

Double DQN - Better value estimation
Dueling Networks - Separates state value from action advantages
Prioritized Replay - Learns more from important experiences
Multi-step Learning - Looks ahead multiple steps
Distributional RL (C51) - Models full reward distribution
Noisy Networks - Exploration without epsilon-greedy

Architecture:

agent.py - The RL agent with training logic
networks.py - Neural network architectures (CNN-based)
space_invaders.py - Main training/evaluation script
preprocessing.py - Frame processing (grayscale, resize, stack)
efficient_replay_buffer.py - Memory-efficient experience storage
config.py - All hyperparameters

How to use:
```
bash setup.sh
```
# Train for 10k episodes
```
python space_invaders.py train --episodes 10000
```
# Evaluate trained model
```
python space_invaders.py evaluate models/rainbow_best.pth
```
# Train basic DQN instead
```
python space_invaders.py train --basic --episodes 1000
```
The implementation is well-structured with memory-efficient replay buffer, comprehensive logging, checkpointing, and visualization of training results. It's licensed under GPL-3.0.Z

Installation Steps:
1. Prerequisites
Make sure you have:

Python 3.8+ installed
pip (Python package manager)

2. Download/Clone the project
If you have the files, navigate to the project directory in your terminal.
3. Run the setup script
On Linux/Mac:
bashbash setup.sh
On Windows:
bashbash setup.sh
(or if you don't have bash, see manual installation below)
This script will:

Check your Python version
Create a virtual environment
Install all dependencies from requirements.txt
Verify the installation

4. Activate the virtual environment
On Linux/Mac:
```
bashsource venv/bin/activate
```
On Windows:
```
bashvenv\Scripts\activate
```
6. Start training!
bash# Quick test (10 episodes)
```
python space_invaders.py train --episodes 10
```

# Full training (10,000 episodes - will take hours/days)
```
python space_invaders.py train --episodes 10000
```

Manual Installation (if setup.sh doesn't work):
bash# 1. Create virtual environment
```
python -m venv venv
```
# 2. Activate it
# Linux/Mac:
```
source venv/bin/activate
```
# Windows:
```
venv\Scripts\activate
```

# 3. Upgrade pip
```
pip install --upgrade pip
```

# 4. Install dependencies
```
pip install -r requirements.txt
```

Notes:

PyTorch is ~2.5GB - installation may take a while
CUDA GPU is optional but highly recommended for faster training (CPU will be very slow)
Training 10,000 episodes can take days on CPU, hours on GPU
The script will save models to models/ and logs to logs/
