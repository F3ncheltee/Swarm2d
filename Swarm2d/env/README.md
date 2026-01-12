## Swarm2D Environment

The `Swarm2DEnv` is a high-performance, multi-agent reinforcement learning (MARL) environment designed for research in swarm robotics and complex coordination tasks. It simulates teams of agents operating in a 2D world with continuous physics, requiring them to sense, navigate, and interact with resources and other agents.

The environment is built on `gymnasium.Env` for a standard API, uses PyBullet for efficient physics simulation, and features a sophisticated, multi-modal observation space designed to enable complex, memory-driven agent policies.


### Basic Usage

Here is a minimal example of how to instantiate and interact with the environment:

```python
import gymnasium as gym
from Swarm2d.env import Swarm2DEnv

# Load a scenario file to configure the environment
scenario_path = 'path/to/your/scenario.yaml'
env = Swarm2DEnv(scenario_path=scenario_path)

# Reset the environment and get the initial observation
observation, info = env.reset()

# Run the simulation for a few steps
for _ in range(100):
    # Get a random action for each agent
    actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
    
    # Step the environment
    observation, reward, terminated, truncated, info = env.step(actions)
    
    # Render the environment (optional)
    env.render()

env.close()
```

## Core Gameplay Mechanics

The core objective in `Swarm2DEnv` is for teams of agents to compete in gathering resources and bringing them back to their home base (Hive) while navigating a dynamic environment and interacting with opponents.

### Key Entities

*   **Agents:** The primary actors in the simulation. Agents belong to a team, have physical properties (position, velocity), and internal states (health, energy). They consume energy over time and must manage their health during combat.
*   **Hives:** The home base for each team. Agents must bring resources to their team's Hive to score points. Hives are typically stationary.
*   **Resources:** Collectible items scattered throughout the environment. When an agent picks up a resource, it becomes "grappled" to them. The resource is only secured once the agent returns it to their Hive.

### Core Mechanics

*   **Resource Gathering:** Agents can move to a resource and initiate a "pickup/attachment"-action to pick it up. They must then transport it back to their Hive. This task is complicated by the physics simulation, as carrying a resource can affect an agent's movement. Furthermore resources are scaled "cooperatively" - that means that multiple agents are needed to carry larger resources back to the hive, a highly cooperative task involving multiple stages of cooperation.
*   **Adversarial Resilience:** Agents can engage with non-cooperative agents from opposing teams. These high-interaction scenarios are resolved through the physics engine—by physically pushing or impacting opponents, agents can apply force to them. Defeated agents are temporarily removed from the simulation before respawning. Multiple agents are stronger together and can protect / contest territory together.
--> These mechanics resemble a mixed environment with multiple stages of cooperative or competitive behavioural possibilities, furthermore a basic hive - contest mechanic is implemented, so that agents can engage with hives and overtake them. 
These mechanics are supported via the rewards, but also via the intrinsic consequences given by the environment, engaging a competitor spawns resources, but also drains energy, making it a costly, but worthy encounter. Similarly delivering resources creates a energy-recharge loop, based on the mass of the resource itself - enabling a dynamic intrinsic reward for resource delivery.

## Core Components

The environment is built from several key Python modules:

-   `env.py`: The main entry point, defining the `Swarm2DEnv` class that adheres to the `gymnasium.Env` interface. It manages the simulation state, handles the `step` and `reset` logic, and orchestrates all other components.
-   `physics.py`: Manages all physical interactions using the PyBullet engine. It handles entity movement, collisions, and grapple physics in a vectorized manner for performance.
-   `observations.py`: Responsible for generating the rich, multi-modal observations for each agent. This includes line-of-sight calculations, constructing local perception graphs, and managing memory systems.
-   `rewards.py`: Calculates rewards for agents based on their actions and outcomes, such as resource collection, delivery, and combat results. The reward structure is configurable per-team.
-   `managers.py`: A collection of helper classes that manage specific aspects of the simulation, such as the `SpawnManager` for creating agents/resources and the `RenderManager` for debug visualizations.
-   `batched_graph_memory.py`: Optimized, GPU-accelerated module that manages the persistent graph-based memory for all agents in a unified, batched manner.

## Simulation Flow (`step` function)

A single call to `env.step(actions)` proceeds through a carefully ordered sequence of operations to ensure correctness and performance:

1.  **State Decay & Respawn:** Entity states (e.g., health, energy) are decayed. Delivered resources or defeated agents are removed and may be respawned.
2.  **Vectorized Proximity Search:** A single efficient proximity search is performed for all entities to find potential interaction pairs (e.g., agent-resource, agent-agent).
3.  **Action Application:** Agent actions (movement, pickup) are translated into physical forces and logical interactions (e.g., applying grapple constraints).
4.  **Physics Simulation:** The PyBullet engine is stepped forward once, resolving all forces and constraints to determine the new physical state of all entities.
5.  **Post-Physics Logic:** Game logic that depends on the outcome of the physics step is processed. This includes resource delivery checks at hives, combat outcomes, and grapple breaks.
6.  **Reward Calculation:** Rewards are computed based on the events that occurred during the step.
7.  **Observation Generation:** A new, comprehensive observation is generated for each agent based on the updated world state. This involves visibility checks and querying the memory systems.

## Customization and Extension

The environment is designed to be highly configurable and extensible:

-   **Scenario Files:** The entire environment setup can be defined in a single YAML file, specifying the number of teams, agents, resources, obstacles, and various physical parameters. See the example below.
-   **Modular Managers:** You can modify or replace individual manager classes (e.g., `RewardManager`, `SpawnManager`) to change core game logic without altering the main environment file.
-   **Dynamic Policies:** The simulation can load different agent policies at runtime, allowing for easy testing and comparison of different behaviors.

### Example Scenario File (`scenario.yaml`)

```yaml
world:
  width: 100
  height: 100
  gravity: [0, 0]

teams:
  - id: 1
    num_agents: 8
    hive_pos: [20, 50]
  - id: 2
    num_agents: 8
    hive_pos: [80, 50]

resources:
  - type: 'gold'
    num: 10
    value: 1

agent:
  radius: 1.0
  max_speed: 10.0
  energy_decay: 0.01
```

## Action Space

The action space for each agent is a dictionary (`gym.spaces.Dict`) containing two components that allow for continuous control over movement and a binary action for interaction.

| Key       | Space                  | Description                                                                                                                                  |
|-----------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `move`    | `Box(low=-1, high=1, shape=(2,))` | A 2D vector that applies a force to the agent. This allows for continuous control over acceleration and direction.                   |
| `pickup/grapple` | `Discrete(2)`          | A binary action to either initiate or release a grapple. This is used to pick up resources or interact with other objects in the future.       |

This hybrid action space enables agents to learn fine-grained motor control for navigation while also making discrete, high-level decisions about interacting with the world.

## Reward System

The reward system is designed to be flexible, configurable, and multi-faceted to encourage complex behaviors. All reward components are defined on a per-team basis within the scenario file, allowing you to tune the incentives for different strategies.

The total reward is a sum of the following components, which are calculated across different parts of the simulation (e.g., `rewards.py`, `physics.py`).

### Resource & Objective Rewards
*   **Resource Delivery (`r_delivery`):** A large positive reward given to the team when an agent successfully delivers a resource to their Hive. This is the primary objective-based reward.
*   **Progress Towards Delivery (`r_progress`):** A small, continuous positive reward for reducing the distance between a carried resource and the target Hive.

### Exploration & Discovery Rewards
*   **Intrinsic Exploration (`r_exploration_intrinsic`):** A reward for visiting less-frequented cells of the map, encouraging agents to cover new ground.
*   **Resource Discovery (`r_resource_found`):** A sparse reward given the first time an agent observes a resource. This reward has a cooldown to prevent exploitation.

### Adversarial & Survival Rewards
*   **Combat Win (`r_combat_win`):** A positive reward given to a team for defeating a non-cooperative agent.
*   **Combat Loss (`r_combat_lose`):** A negative reward (penalty) when an agent is defeated by an opponent.
*   **Death (`r_death`):** A significant negative penalty applied directly to an agent when it is defeated.

### Physical Interaction Rewards
*   **Grapple Control (`r_grapple_control`):** A small continuous positive reward for actively physically manipulating/carrying a resource or another agent.
*   **Being Grappled (`r_grapple_controlled`):** A continuous negative reward (penalty) for being physically restrained by a non-cooperative agent.
*   **Grapple Break (`r_grapple_break`):** A positive reward for successfully breaking free from an opponent's constraint.

### Hive Interaction Rewards
*   **Hive Win (`r_hive_win`):** A large team-wide reward for destroying an enemy hive.
*   **Hive Loss (`r_hive_lose`):** A large team-wide penalty when the team's own hive is destroyed.
*   **Continuous Hive Attack:** The system also supports a continuous reward for agents that remain near an enemy hive, encouraging sieges.

### Unimplemented Reward Keys
The following reward keys are defined in `constants.py` but are not yet implemented in the core simulation logic. They serve as placeholders for potential future features:
`r_attachment`, `r_combat_continuous`, `r_hive_total`, `r_obstacle_found`, `r_enemy_found`, `r_hive_found`, `r_teammate_lost_nearby`, `r_torque_win`.

## Visualizing the Environment

To understand the simulation state and agent behaviors, you can render the environment. Calling `env.render()` after a `step()` will produce a visual representation of the current state, which is invaluable for debugging and analysis.

## Observation Space and Memory Architecture

The `Swarm2DEnv` provides a rich, multi-modal observation to the agent at each step. This observation is designed to give the agent both immediate, high-fidelity sensory information and a persistent, long-term memory of the world. The final observation is a dictionary containing four key components:

| Key          | Type                     | Description                                                                                                                                                             |
|--------------|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `self`       | `torch.Tensor` (Vector)  | A flat vector containing the agent's own state (proprioception), such as position, velocity, health, energy, and information about its immediate goals (e.g., vector to hive). |
| `map`        | `torch.Tensor` (3D)      | A live, **egocentric** image-like representation of the agent's immediate surroundings. It's a multi-channel tensor showing the presence of entities within the agent's line-of-sight. This is the agent's raw, immediate perception. |
| `memory_map` | `torch.Tensor` (3D)      | A persistent, **allocentric** (world-oriented), top-down 2D grid representing the agent's accumulated knowledge. It is updated at each step with information from the live `map`. Channels in this map decay over time, representing fading memory and uncertainty. |
| `graph`      | `torch_geometric.data.Data` | The primary and most powerful observation component. This is a **unified relational graph** that combines the agent's live perception with its persistent graph memory. It provides a structured, relational view of the world. |

This multi-modal observation gives the policy a comprehensive understanding of the environment, combining immediate sensory data with a persistent, decaying memory of the world.

### In-Depth Look: The Unified Graph Observation

The `graph` is the most sophisticated component of the agent's perception. It represents the world as a structured network of entities, enabling relational reasoning. The system is designed to provide maximum detail for the agent's immediate surroundings (the **fovea**) while intelligently abstracting distant information (the **periphery**) to maintain computational efficiency and provide high-level strategic cues.

#### The Fovea & Periphery Model

-   **The Fovea (Live Perception):** All entities currently within the agent's observation radius and line-of-sight are represented as individual, high-fidelity nodes. Edges in this region are created based on spatial proximity, forming a detailed relational map of the agent's immediate tactical situation.
-   **The Periphery (Persistent Memory):** Entities that have been observed in the past but are no longer in the fovea are stored in a persistent graph memory. To keep this memory graph manageable and strategically useful, these nodes are clustered.

#### Adaptive Clustering Mechanism

The clustering process in the periphery is adaptive and content-aware, creating a multi-layered abstraction of the world.

-   **High-Detail Buffer Zone:** A small zone immediately surrounding the fovea where memory nodes are **not clustered**. This provides a high-resolution buffer that prevents entities from abruptly changing representation as they move in and out of sight.
-   **Mid-Periphery (Detailed Clustering):** In the region beyond the buffer, nodes are clustered with high granularity. The clustering is **content-aware**:
    -   **Agents and Hives** are clustered separately for each team.
    -   **Resources** are clustered based on their type (cooperative vs. single-agent).
-   **Far-Periphery (Generic Clustering):** In the most distant regions of memory, nodes are aggregated into large, coarse "super-nodes" based only on their fundamental entity type (e.g., all distant agents might be grouped into a single cluster).

This adaptive approach ensures that the agent retains detailed information about strategically important areas while compressing distant, less relevant memories into high-level summaries.

#### Graph Construction: Edges and Structure

The final unified graph is assembled from these components, with several types of edges created to represent different relationships:

-   **Hierarchical Edges:** The agent's own "ego" node is connected to a curated set of important "landmark" nodes in the periphery. These landmarks are chosen based on strategic importance, such as the closest enemy clusters, resource clusters, and all known hives. This provides the agent with a high-level "skeleton" of its long-term memory.
-   **Semantic Edges:** Beyond simple proximity, the graph includes edges that represent abstract relationships, allowing the policy to perform deeper reasoning:
    -   **Kinematic Edges:** Connect agents with significant relative velocity, useful for interpreting chase or evasion behaviors.
    -   **Affiliation Edges:** Connect agents to their own and opposing hives, providing crucial strategic context.
    -   **Interaction State Edges:** Connect agents that are actively physically engaging each other.
    -   **Shared Intent Edges:** Connect allied agents that are moving towards the same target.

### 1) The `map` (Raw Map): Egocentric, Real-Time Perception

The `map` is the agent's immediate, first-person perception of the world. It's a multi-channel, image-like tensor that provides a real-time snapshot of entities within the agent's observation radius, respecting line-of-sight occlusions.

-   **Architecture**: It is a `(C, H, W)` tensor where `C` is the number of channels, and the height `H` and width `W` are determined by the `obs_map_size` parameter.
-   **Egocentric View**: The map is agent-centric; the agent is always at the center of this observation grid.
-   **Dynamic Scaling**: The scale of the map is dynamic. A `world_to_map_scale` factor is calculated based on the agent's `obs_radius` to ensure its entire circular field of view fits perfectly within the square map tensor. This allows perception resolution to adapt if the agent's observation radius changes.

#### Map Generation via Gaussian Splatting

To create a rich and continuous representation suitable for neural networks, the environment renders entities onto the map via **Gaussian Splatting**.

1.  **Entity Query**: The system first identifies all entities within the agent's observation radius.
2.  **Splatting**: Each entity is then "splatted" onto the map as a 2D Gaussian distribution centered at its relative position.
    -   **Size Encoding**: The entity's physical size is encoded in the standard deviation (sigma) of the Gaussian. Larger entities produce a wider, more spread-out splat, providing the agent with precise size information.
    -   **Intensity Falloff**: The brightness of the splat naturally fades from the center outwards, creating the "soft edge" effect. This provides a smooth gradient for learning and implicitly encodes distance.
3.  **Channel Mapping**: The calculated intensity values for each entity are written into the appropriate channel of the map tensor (e.g., an enemy agent's splat goes into the `enemy_presence` channel).
4.  **Overlap Resolution**: If multiple Gaussians overlap on the same pixel, the `max` intensity is taken. This ensures that the most prominent or closest entity is what the agent perceives in that pixel.

#### Raw Map Channels (`RAW_CH`)

The `map` tensor contains the following channels, each representing the presence and intensity of a specific entity type:

| Channel                  | Index | Description                                                              |
| ------------------------ | ----- | ------------------------------------------------------------------------ |
| `ally_presence`          | 0     | Presence of allied agents (excluding self).                              |
| `enemy_presence`         | 1     | Presence of enemy agents.                                                |
| `resource_presence`      | 2     | Presence of standard (single-agent) resources.                           |
| `coop_resource_presence` | 3     | Presence of cooperative (multi-agent) resources.                         |
| `hive_ally_presence`     | 4     | Presence of the agent's own or allied hives.                             |
| `hive_enemy_presence`    | 5     | Presence of enemy hives.                                                 |
| `obstacle_presence`      | 6     | Presence of static obstacles.                                            |
| `self_presence`          | 7     | A marker for the agent's own position, typically centered.                 |

### 2) The `memory_map`: A Detailed Look

The `memory_map` is a persistent, top-down 2D grid that provides each agent with long-term spatial awareness. It functions as an allocentric (world-oriented) map that accumulates knowledge over time, simulating a fading memory of the environment.

#### Architecture and Data Flow

-   **Foundation:** Each agent maintains its own high-resolution grid that represents the entire world space. This grid is persistent across timesteps.
-   **Updating from Raw Perception:** At each step, the agent's live, egocentric `map` observation (its immediate line-of-sight view) is projected onto the correct coordinates of the persistent, allocentric `memory_map`. This process acts like "painting" the agent's current view onto a larger canvas, updating the agent's knowledge of the world.
-   **Dynamic Scaling:** The resolution of the map is determined by the agent's observation radius (`obs_radius`). The `world_to_map_scale` factor ensures that the agent's entire field of view fits perfectly into its egocentric `map` tensor, which is then used to update the persistent memory. This allows the scale of perception to be dynamically adjusted.

#### Time-Based Channels and Memory Fading

A key feature of the `memory_map` is its ability to model forgetting. Instead of storing binary presence/absence, most channels store floating-point timestamps of when an entity was last observed.

-   **Timestamping:** When a dynamic entity (like an agent or a resource) is seen, the current simulation step is recorded in the corresponding channel at that location. Obstacles are an exception; they are considered static and store a permanent presence value of `1.0`.
-   **Age Calculation:** When the `memory_map` is prepared for the policy, these timestamps are converted into a normalized **age** value, which ranges from `0.0` (seen this step) to `1.0` (seen long ago).
-   **Recency Normalization:** This age calculation is normalized by the `recency_normalization_period` (typically 250 steps). An observation that is 250 steps old or older will have a normalized age of `1.0`, effectively representing the limit of the agent's reliable memory. A value of `1.1` is used to signify "never seen."

#### The Coverage Map

Generated from the `memory_map`, the `coverage_map` is a single-channel layer that provides crucial spatial density information through a separate channel.

-   **Purpose:** It represents the **spatial density** of all remembered entities. It answers the question: "How much 'stuff' (obstacles, resources, agents) is in this area?"
-   **Calculation:** It is calculated by checking for the presence of any entity across all relevant `memory_map` channels. The resulting high-resolution density map is then downsampled to the final observation size.
-   **Usage:** The `coverage_map` is fed directly to the policy as a distinct channel, giving the agent a sense of how "cluttered" or "empty" different parts of the world are. It is also a key component in the visualization pipeline, where it modulates the brightness of rendered entities.

#### 4. Final Output Channels

The final `memory_map` tensor passed to the policy includes several channel categories:
-   **Age-Based Channels:** `last_seen_resource`, `last_seen_ally`, `last_seen_enemy`, etc., containing the normalized age (`0.0` to `1.0`).
-   **Binary Presence:** `obstacle_presence` (`1.0` if present).
-   **Exploration:** `explored` channel, which also uses age to show a faded history of the agent's path.
-   **Global Context:** `vec_hive_x`, `vec_hive_y` (a vector field pointing to the home hive), `step_norm` (normalized timestep), and `you_are_here` (a marker for the agent's own position).
-   **Density:** The `coverage` map.

This combination of persistent, time-decaying memory and high-level contextual channels provides the agent with a rich and robust understanding of its environment.

### The Two Memory Systems

An agent's "memory" is managed by two distinct but complementary systems that produce the `memory_map` and the unified `graph`:

1.  **ActorMapState (The 2D Memory Map):**
    *   **Purpose:** Provides a persistent, top-down grid view of the world.
    *   **Mechanism:** Each agent has a full-resolution grid representing the entire environment. At each step, the agent's live egocentric `map` is "painted" onto this persistent grid at the correct world coordinates.
    *   **Features:** Channels in this map have a "certainty" that decays over time. For example, the location of a mobile non-cooperative agent becomes less certain if it hasn't been seen recently. This provides the policy with a sense of spatial awareness and uncertainty.
    *   **Output:** This system produces the `memory_map` observation.

2.  **BatchedPersistentGraphMemory (The Hierarchical Graph Memory):**
    *   **Purpose:** This is the most powerful component of the agent's perception. It provides a structured, relational graph that combines live perception with a highly abstracted long-term memory.
    *   **Fovea (High-Detail View):** In the agent's immediate vicinity (the "fovea"), all entities are represented as fine-grained, individual nodes. These nodes are richly interconnected based on spatial proximity, giving the agent a detailed relational understanding of its current tactical situation.
    *   **Periphery (Abstracted Memory):** To maintain efficiency, memories outside the fovea are processed through a sophisticated abstraction pipeline:
        1.  **Adaptive Clustering:** Memories are grouped into "super-nodes." This clustering is adaptive—distant memories are grouped into large, coarse clusters, while memories just outside the fovea are grouped into smaller, more detailed clusters.
        2.  **Sparse Skeleton:** These clusters are then connected to each other using a localized radius. This avoids a tangled "hairball" of connections, instead creating a clean, sparse "skeleton" representing the high-level spatial structure of the agent's long-term memory.
    *   **Hierarchical Bridge Connections:** The detailed fovea is intelligently linked to the abstracted periphery. The agent's "ego" node forms a small number of high-importance connections to "bridge" nodes in the memory skeleton. These bridges are selected based on both proximity and content-awareness, prioritizing connections to the closest overall clusters as well as the nearest known enemy and resource clusters.
    *   **Dynamic Edge Attributes:** Every edge in the final graph is enriched with dynamic data, including the relative position and distance between nodes, and a "connection strength" based on the certainty of the memories it connects. This transforms edges from simple lines into rich information channels.
    *   **Output:** This system produces the final, unified `graph` observation.

### Data Flow Summary

The process from raw state to final, memory-augmented observation is as follows:

1.  **Live Perception:** The environment first generates a `graph` and a `map` of entities currently within the agent's line-of-sight. The `self` vector is also computed.
2.  **Memory Update:**
    *   The live `map` is used to update the `ActorMapState`, refreshing the agent's 2D `memory_map`.
    *   The live `graph` is passed to the `BatchedPersistentGraphMemory` to update the persistent memory graph (adding new nodes and updating existing ones).
3.  **Unified Graph Generation:** The `BatchedPersistentGraphMemory` then generates the final, unified `graph` by combining the new live graph (for the fovea) with a clustered representation of its stored memory (for the periphery).
4.  **Final Assembly:** The environment assembles the dictionary with the `self` vector, the live `map`, the persistent `memory_map`, and the final unified `graph`, and returns it to the agent.
