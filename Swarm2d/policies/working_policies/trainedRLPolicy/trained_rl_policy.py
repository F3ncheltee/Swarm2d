import torch
import numpy as np
import os
import sys
from PyQt6.QtWidgets import QFileDialog

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# It's crucial to import the class definition of the network you want to load
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
import os

from policies.actors.SHARED.SharedAgentGNN import SharedActorPolicy


class TrainedRLPolicy:
    def __init__(self, env, agent_id, policy_path=None):
        self.env = env
        self.team_id = team_id
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            # If no path is provided, open a dialog to select one
            # Note: This will pause the main thread until a file is selected.
            print("No model path provided for TrainedRLPolicy. Opening file dialog...")
            # We pass None as the parent, a simple alternative for a non-GUI context
            model_path, _ = QFileDialog.getOpenFileName(None, "Select a Trained Model File", "", "PyTorch Model Files (*.pth *.pt)")
        
        if model_path and os.path.exists(model_path):
            self.load_policy(model_path)
        else:
            print(f"Warning: No model file selected or path '{model_path}' does not exist. This policy will do nothing.")

    def load_policy(self, model_path):
        try:
            print(f"Loading trained model from: {model_path}")
            
            # --- 1. Get parameters from the environment ---
            # A trained policy's architecture must match the environment it was trained in.
            # We get these parameters from the env metadata.
            metadata = self.env.metadata
            num_agents_on_team = self.env.agent_counts_per_team[self.team_id]

            # --- 2. Instantiate the policy network ---
            # Use the parameters from the environment to create a policy shell.
            self.policy = SharedActorPolicy(
                num_agents_on_team=num_agents_on_team,
                self_feature_dim=metadata['self_obs_dim'],
                map_channels=metadata['raw_map_channels'],
                grid_size=metadata['actor_map_grid_size'],
                obs_radius=metadata['obs_radius'],
                # Add other necessary parameters from metadata or defaults
                # These should match the parameters used during training
            ).to(self.device)

            # --- 3. Load the saved weights (state_dict) ---
            saved_state = torch.load(model_path, map_location=self.device)
            
            # The saved file might contain more than just the policy weights 
            # (e.g., optimizer state). We look for the policy state dict.
            if 'policy_state_dict' in saved_state:
                policy_state_dict = saved_state['policy_state_dict']
            else:
                # Assume the file contains only the policy state_dict
                policy_state_dict = saved_state

            self.policy.load_state_dict(policy_state_dict)
            
            # --- 4. Set to evaluation mode ---
            self.policy.eval()
            
            print("Model loaded successfully.")

        except Exception as e:
            print(f"Error loading policy: {e}")
            import traceback
            traceback.print_exc()
            self.policy = None

    def get_actions(self, obs, agent_ids):
        """
        Gets actions for the specified agents using the loaded RL policy.
        """
        if not self.policy:
            # Return do-nothing actions if the policy failed to load
            return [{"movement": np.zeros(2), "pickup": 0} for _ in agent_ids]

        actions = []
        for agent_id in agent_ids:
            # The 'act' method of the policy expects a single observation,
            # not a batch. We extract the observation for the current agent.
            single_obs = obs[agent_id]
            
            # The policy's `act` method handles device placement and un-batching
            action_dict, _, _, _ = self.policy.act(single_obs)
            actions.append(action_dict)
            
        return actions
