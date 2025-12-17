import torch
import torch.nn.functional as F
import wandb
import numpy as np

class LossLogger:
    def __init__(self, agent, log_every=50):
        self.agent = agent
        self.log_every = log_every
        self.calls = 0

    def __call__(self, dataset):
        self.calls += 1

        if self.calls % self.log_every != 0 or wandb.run is None:
            return

        if self.agent._replay_memory.size < 255:
            return

        try:
            batch = self.agent._replay_memory.get(255)
            state, action, reward, next_state, absorbing, _ = batch

            q1 = self.agent._critic_approximator[0].predict(state, action)

            actor_out = self.agent._actor_approximator.predict(state)

            wandb.log({
                "Q/mean": float(np.mean(q1)),
                "Q/std": float(np.std(q1)),
                "Q/min": float(np.min(q1)),
                "Q/max": float(np.max(q1)),
                "Actor/mean": float(np.mean(actor_out)),
                "Buffer/size": self.agent._replay_memory.size,
            })
        except Exception as e:
            print(f"LOSS LOGGER ERROR: {e}")
