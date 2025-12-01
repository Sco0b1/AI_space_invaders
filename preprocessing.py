import numpy as np
import cv2
from collections import deque
from config import FRAME_STACK, FRAME_WIDTH, FRAME_HEIGHT


class FramePreprocessor:
    """Preprocess and stack frames for DQN input"""
    
    def __init__(self, frame_stack=FRAME_STACK):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
    
    def preprocess_frame(self, frame):
        """Convert to grayscale, resize, and normalize"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT), 
                            interpolation=cv2.INTER_AREA)
        return (resized / 255.0).astype(np.float32)
    
    def reset(self, initial_frame):
        """Initialize frame stack with first frame"""
        processed = self.preprocess_frame(initial_frame)
        
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        
        return np.array(self.frames, dtype=np.float32)
    
    def add_frame(self, frame):
        """Add new frame to stack and return updated stack"""
        processed = self.preprocess_frame(frame)
        self.frames.append(processed)
        return np.array(self.frames, dtype=np.float32)