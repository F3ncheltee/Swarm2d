# Actor Network Architectures

This directory contains the various neural network architectures for the "actor" component of the actor-critic reinforcement learning framework. The actor's role is to take an agent's observation as input and produce an action.

The subdirectories contain different experimental architectures that explore various approaches to processing the complex, graph-based observations from the `Swarm2d` environment.

## Architectures

### `/MAAC` (Multi-Agent Actor-Critic)

This directory implements a Multi-Agent Actor-Critic architecture, which is a common approach in multi-agent reinforcement learning.

-   `maac_attentionGNN.py`: Defines the core `MAACPolicy`. This policy uses a Graph Neural Network (GNN) to process the scene graph from the observation. A key feature of this architecture is its use of **role-based learning**. It maintains separate network heads for different "roles" (e.g., "worker," "fighter"), and an attention mechanism selects which role is most appropriate at each timestep.
-   `updateactorMAAC.py`: Contains the specific logic for updating the weights of the `MAACPolicy` actor during training.

### `/NCA` (Neural Cellular Automata)

This architecture is inspired by Neural Cellular Automata. It processes the graph of entities in a way that mimics local communication, allowing complex global behaviors to emerge from simple local rules.

-   `nca_networkGNN.py`: Defines the `NCA_PINSANPolicy`. This policy uses a GNN that is updated iteratively, similar to the steps in a cellular automaton. This allows information to propagate through the graph of nearby agents, facilitating coordinated actions.
-   `updateactorNCA.py`: Contains the update logic for the `NCA_PINSANPolicy`.

### `/SHARED`

This architecture explores the use of a single, shared network for all agents on a team, promoting homogeneous and coordinated behavior.

-   `SharedAgentGNN.py`: Defines the `SharedActorPolicy`. This policy uses a GNN and also incorporates a "trail memory" system. Agents can "write" semantic information to a spatial grid, and other agents can "read" from this grid, allowing for a form of indirect communication and coordination.
-   `updateactorShared.py`: Contains the update logic for the `SharedActorPolicy`.

## Common Elements

-   **Graph Neural Networks (GNNs):** All these architectures heavily rely on GNNs to process the relationships between agents and other entities in the environment.
-   **Actor Updates:** Each architecture has its own `updateactor...py` script, which contains the loss calculations and backpropagation steps necessary to train the actor network.
-   **Experimental Nature:** These policies are designed for research and may contain experimental features. They represent different hypotheses about how to best achieve coordination and complex behavior in a multi-agent setting.
