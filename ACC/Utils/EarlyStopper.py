import logging
from collections import deque
import numpy as np


class EarlyStopper:
    def __init__(self, patience=3, min_delta=5.0, verbose=True):
        """
        patience: How many chunks to wait without improvement before stopping.
        min_delta: Minimum increase in reward to count as an "improvement".
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = -float('inf')
        self.early_stop = False
        self.recent_scores = deque(maxlen=5)  # Keep track for logging

    def __call__(self, current_score):
        self.recent_scores.append(current_score)
        avg_recent = np.mean(self.recent_scores)

        # Check if this is the best score we've seen (plus a margin)
        if current_score > self.best_score + self.min_delta:
            self.best_score = avg_recent
            self.counter = 0  # Reset patience
            if self.verbose:
                logging.info(f"Improvement detected! New Best: {self.best_score:.2f}")
        else:
            self.counter += 1
            if self.verbose:
                logging.info(
                    f"No significant improvement. Patience: {self.counter}/{self.patience}. (Current: {current_score:.2f}, Best: {self.best_score:.2f})")

            if self.counter >= self.patience:
                self.early_stop = True
                logging.info("Early Stopping triggered! Moving to next scenario.")

        return self.early_stop