import numpy as np

class PolicyTemplate:
    """
    This is a template for creating a new policy. To use it, follow these steps:
    
    1. Copy this file and rename it to reflect your new policy's name 
       (e.g., 'my_awesome_policy.py').
    
    2. Change the class name from 'PolicyTemplate' to a name that matches your file, 
       using PascalCase (e.g., 'MyAwesomePolicy').
       
    3. Implement the `get_actions` method. This is the core of your policy.
       It receives the global observation `obs` and a list of `agent_ids` that this
       policy is responsible for. It should return a list of action dictionaries,
       one for each agent in `agent_ids`.
    """
    def __init__(self, env, team_id=None):
        """
        The constructor for your policy. You can use this to store any state you
        need, such as a reference to the environment (`env`) or the `team_id`
        this policy is controlling.
        """
        self.env = env
        self.team_id = team_id

    def get_actions(self, obs, agent_ids):
        """
        This method is called at each step of the simulation to get the actions
        for the agents controlled by this policy.

        Args:
            obs (dict): The global observation from the environment. This contains
                        information about all entities in the simulation.
            agent_ids (list of int): A list of agent IDs that this policy needs to
                                     provide actions for.

        Returns:
            list of dict: A list of action dictionaries, one for each agent in 
                          `agent_ids`. Each dictionary should have the format:
                          {"movement": np.array([x, y]), "pickup": 0 or 1}
        """
        actions = []
        for agent_id in agent_ids:
            # --- Implement your decision-making logic here ---
            #
            # Example: A simple random action
            movement = np.random.uniform(-1, 1, size=2)
            pickup = np.random.randint(0, 2)
            
            # Add the action for the current agent to the list
            actions.append({"movement": movement, "pickup": pickup})
            
        return actions
