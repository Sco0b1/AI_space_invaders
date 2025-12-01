#Training parameters
EPISODES = 10000
BATCH_SIZE = 32
LEARNING_RATE = 0.0000625
ADAM_EPSILON = 1.5e-4
GAMMA = 0.99

#Exploration parameters
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999

#Network update parameters
TARGET_UPDATE_FREQUENCY = 8000
TAU = 1.0
REPLAY_BUFFER_SIZE = 600000
MIN_REPLAY_SIZE = 80000

#Prioritized experience replay
USE_PRIORITIZED_REPLAY = True
PRIORITY_ALPHA = 0.5
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_FRAMES = 15000000

#Multi-step learning
USE_MULTI_STEP = True
N_STEP = 3

#Distributional RL (C51)
USE_DISTRIBUTIONAL = True
N_ATOMS = 51
V_MIN = -10
V_MAX = 10

#Noisy networks
USE_NOISY_NETS = True
NOISY_STD = 0.5

#Frame processing
FRAME_STACK = 4
FRAME_WIDTH = 84
FRAME_HEIGHT = 84

#Checkpointing
SAVE_FREQUENCY = 1000
PRINT_FREQUENCY = 1

#Paths
MODEL_SAVE_PATH = "models/"
LOG_PATH = "logs/"

#Architecture
USE_DUELING = True
USE_DOUBLE_DQN = True

def validate_config():
    """Validate configuration parameters"""
    assert N_ATOMS > 1, "N_ATOMS must be > 1"
    assert V_MIN < V_MAX, "V_MIN must be < V_MAX"
    assert 0 < PRIORITY_ALPHA <= 1, "PRIORITY_ALPHA must be in (0, 1]"
    assert 0 < PRIORITY_BETA_START <= 1, "PRIORITY_BETA_START must be in (0, 1]"
    assert N_STEP >= 1, "N_STEP must be >= 1"
    assert GAMMA > 0 and GAMMA <= 1, "GAMMA must be in (0, 1]"

validate_config()