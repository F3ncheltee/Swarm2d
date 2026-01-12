import numpy as np

class RandomPolicy:
    def __init__(self, action_space):
        """
        A policy that returns random actions.

        Args:
            action_space: The environment's action space.
        """
        self.action_space = action_space

    def act(self, observation):
        """
        Return a random action as a dictionary.

        Args:
            observation: The current observation from the environment.

        Returns:
            A dictionary containing a random action.
        """
        return self.action_space.sample()
