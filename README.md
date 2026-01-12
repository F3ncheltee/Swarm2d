

<img src="https://github.com/user-attachments/assets/75448a02-df5f-48b5-92c5-cba426778443" width="400" alt="Swarm2D Configuration" />

# Swarm2d: A Multi-Agent Reinforcement Learning Environment

Swarm2d is a highly configurable multi-agent reinforcement learning (MARL) environment designed for research in swarm intelligence, multi-agent coordination, and complex emergent behaviors. It provides a robust simulation platform for developing and evaluating sophisticated control policies for autonomous agents in dynamic, 2D worlds. Built on the robust PyBullet physics engine, Swarm2d provides a rich testbed for exploring complex emergent behaviors, swarm intelligence, and intricate coordination strategies among teams of autonomous agents. The environment is architected with a modular, manager-based system, allowing for deep customization and extensibility of its core components, from physics and agent capabilities to observation and reward structures.

At its core, Swarm2d is a multi-team, multi-objective environment where teams of agents must navigate a dynamic world to achieve goals that are often in direct conflict. This fosters a natural setting for studying both cooperative and competitive strategies. Agents must balance resource management, territorial control, and direct combat to ensure their team's survival and dominance. The platform includes a powerful training pipeline compatible with standard reinforcement learning frameworks and a feature-rich GUI for real-time simulation, monitoring, and debugging.

## Core Environment and Gameplay Mechanics

The Swarm2d world is a dynamic, physics-based 2D arena designed for multi-agent simulations. It features a rich set of interacting entities, including agents, resources, hives, and obstacles, all governed by the PyBullet physics engine. At the beginning of each simulation episode, the environment is procedurally generated to create diverse and challenging scenarios, ensuring that entities are placed in valid, non-overlapping positions.

#### The Arena: Obstacles and Boundaries
The simulation takes place within a rectangular arena enclosed by impassable boundary walls, ensuring the simulation remains contained. The landscape is populated with static obstacles that serve as barriers and create tactical complexity. A configurable number of obstacles (simple geometric boxes of varying, randomized sizes) are placed throughout the arena, with spawning logic that ensures they do not block hives or overlap with each other.

#### Agents: The Core of the Swarm
Agents are the primary actors in the environment, belonging to distinct teams. They are complex physical entities spawned around their respective team's hive and possess a suite of properties and actions that drive the simulation.

*   **Core Properties**: Each agent possesses fundamental attributes:
    *   **Health and Energy**: Agents have both health and energy pools. Energy is consumed through actions, and if depleted, health begins to drain. An agent is neutralized when its health reaches zero, triggering a respawn cooldown before it can rejoin the simulation at its team's hive.
    *   **Heterogeneous Physical Attributes**: The system supports diverse agent populations. Attributes like radius, mass, and strength are randomized upon spawning based on configurable base values and variance. An agent's mass is dynamically calculated from its radius and strength, creating a diverse population where larger, stronger agents are also heavier.
    *   **Team Affiliation**: Every agent belongs to a team, which dictates its objectives, allies, and enemies.

*   **Physics-Based Movement**: Agent movement is force-based. An agent's policy dictates a desired direction and magnitude, which is translated into a force applied to its physical body. The final applied force is a product of the agent's base speed, its current energy level, and any active status effects. As dynamic bodies in PyBullet, agents are subject to friction, damping, and restitution, leading to realistic acceleration, deceleration, and collisions.

*   **Action Space**: Agents have a concise yet powerful set of actions:
    *   **Movement**: Continuous control over the direction of applied force.
    *   **Interaction (Pickup / Grapple)**: A single, context-sensitive action to either pick up a nearby resource or initiate a grapple with an enemy agent.
    *   **Release**: An action to release a held resource or break free from a grapple.

#### Resource Gathering: Individual and Cooperative Dynamics
The primary economic objective in Swarm2d is to gather resources and deliver them to a team's hive. Resources appear dynamically and have varied properties.

*   **Dynamic Spawning**: The environment maintains a target number of active resources. When a resource is collected, a new one is spawned at a random, unoccupied location.
*   **Physical Properties**: Resources are not static pickups; they are physical objects that must be gripped and carried. Their mass and size impact agent movement. A resource's "value" directly influences its physical radius and mass.
*   **Single & Cooperative Carry**: While smaller resources can be carried by a single agent, a configurable probability exists for spawning larger, heavier "cooperative" resources. These require the coordinated effort of multiple agents to transport effectively, as a single agent attempting to move one will suffer a significant speed penalty.
*   **Grip Strength**: Agents must maintain a "grip" on resources, determined by their strength, health, and energy. A weakened agent may drop its payload, creating an opportunity for rival teams.

#### Hives: The Heart of the Team
Each team is assigned a hive, which serves as its base of operations, resource drop-off point, and respawn location. Hives are spawned in a patterned, circular layout with random jitter to ensure a unique map configuration each time.

*   **Hive Logic**: Hives are static structures with their own health and a "food store." Agents score points by bringing resources to an allied hive, which contributes to its food reserves. Agents near their own hive passively regenerate energy.
*   **Team-Based Mechanics**: The primary goal is to maintain the health of one's own hive while depleting the enemy's.
*   **Hive Combat**: Hives can be attacked by nearby enemy agents, which first drains the food store and then its health. The presence of allied agents near the hive mitigates this damage. As a hive takes damage, it "bleeds" valuable resources into the surrounding area, creating a high-risk, high-reward incentive for attackers.
*   **Destruction, Rebuilding, and Capture**: A destroyed hive drops a single, high-value cooperative resource (its "core") and enters a cooldown state. After the cooldown, the hive can be rebuilt by its original owners. However, if a rival team dominates the location, they can capture it, converting it to their side for a significant strategic advantage.

## Advanced Combat System

Combat in Swarm2d is a deep and tactical system that involves close-quarters mechanics, strategic positioning, and resource management, all grounded in the physics engine.

#### Physics-Based Melee Combat
Engagements are resolved through a sophisticated damage calculation that considers:
*   **Agent Stats**: An agent's strength, energy level, health, and physical size contribute to its damage output and mitigation.
*   **Positional Advantage**: The system rewards tactical maneuvering. Flank and rear attacks deal significantly more damage, while attacking while stationary or moving away from a target is less effective.
*   **Energy Shield**: Agents have an energy reserve that acts as a regenerating shield, absorbing damage before their health is affected.

#### Sophisticated Grappling Mechanics
Agents can choose to grapple their opponents, initiating a complex sub-system of control and counter-play.
*   **Initiation and Control**: A successful grapple establishes a physical constraint between two agents. The grappling agent's goal is to control and damage the target, while the grappled agent's goal is to break free.
*   **Grip and Escape**: The grapple is maintained by the attacker's "grip strength," a value influenced by their stats, fatigue, and the torque they apply. The defender can counter this by struggling and applying counter-torque. A successful escape is a skill-based interaction.
*   **Combat while Grappled**: A grappling agent deals continuous crushing damage to its target, with a chance for critical hits. The grappled agent can inflict "struggle damage" back onto its attacker, creating a high-stakes duel.

## Physics and Simulation Engine

The simulation is built upon the PyBullet physics engine, which governs all interactions. While PyBullet is a 3D engine, the environment is strictly constrained to two dimensions by programmatically resetting each object's Z-axis position and zeroing out any non-planar velocities at every simulation step.

#### Core Physics Implementation
Every entity is represented as a PyBullet MultiBody, allowing for robust handling of physical properties.
*   **Collision Model**: A system of collision groups and masks ensures that objects only interact with appropriate counterparts.
*   **Force Application**: Agent movement is driven by external forces. The magnitude of the force an agent can exert is a dynamic value calculated based on its base speed, current energy level, and any active status effects (e.g., being grappled significantly reduces movement force).
*   **Interaction via Constraints**: Complex interactions like carrying a resource or grappling an enemy are physically modeled using `JOINT_POINT2POINT` constraints. The `maxForce` of these constraints represents "grip strength" and is dynamically updated based on the agent's state.

#### Dynamic Mass Calculation
An agent's mass is not fixed but is procedurally determined at spawn time based on its individual attributes to make each agent's physical presence unique.
*   **Base Mass**: Calculations begin with a standard base mass.
*   **Size Influence (Volume)**: Mass is scaled by the agent's volume (calculated from its radius). Larger agents are inherently more massive.
*   **Strength Influence (Density)**: Mass is further modified by the agent's strength attribute, which acts as a proxy for density.
*   **Mass Clamping**: The final calculated mass is clamped within a predefined range to maintain simulation stability.

This dynamic mass system directly impacts gameplay; a heavier agent is more effective at mitigating damage and has a greater physical impact during grapple struggles.

## Agent Perception and Observation Space

Agents perceive the world through a configurable sensor model, limited by a finite range (`obs_radius`) and line-of-sight, with environmental obstacles and other agents causing occlusions. This creates a realistic "fog of war." The observation provided to each agent is a sophisticated combination of three distinct components.

#### 1. Self-Observation Vector
Each agent receives a flat vector of its own features, providing a concise summary of its internal state. All values are normalized. The features include:
*   `pos_x_norm`, `pos_y_norm`: Normalized X and Y coordinates.
*   `vel_x_norm`, `vel_y_norm`: Normalized X and Y velocity components.
*   `rel_res_x_norm`, `rel_res_y_norm`: Vector to the nearest carried resource.
*   `rel_hive_x_norm`, `rel_hive_y_norm`: Vector to the agent's own hive.
*   `is_carrying`: Binary flag for carrying a resource.
*   `hive_dist_norm`: Normalized distance to its own hive.
*   `speed_norm`: Normalized magnitude of velocity.
*   `radius_norm`, `obs_radius_norm`, `strength_norm`: Physical attributes.
*   `hive_health_norm`, `energy_norm`, `health_norm`: Status levels.
*   `team_energy_norm`: Average energy of the agent's team.
*   `boundary_x1`, `boundary_x2`, `boundary_y1`, `boundary_y2`: Normalized distances to world boundaries.
*   `grip_strength_norm`: Current grip strength for grappling.
*   `agent_id`, `team_id_val`: Unique identifiers.

#### 2. Raw Map: Immediate Awareness

The `raw_map` is the agent's real-time, first-person perception of its immediate surroundings. It functions as a multi-channel, image-like tensor that is regenerated at every simulation step.

-   **Egocentric View**: The map is centered on the agent, providing a consistent "first-person" perspective.
-   **Circular Field of View**: The agent's perception is limited to a circular area defined by its `obs_radius`. To accurately represent this on a square grid, the intensity of observed entities fades to zero at the edge of the radius, creating a "soft edge" effect. This provides the agent's policy with a smooth, continuous signal that implicitly encodes distance/size.
-   **Multi-Channel Representation**: Each channel in the tensor corresponds to a specific type of entity (e.g., allies, enemies, resources, obstacles). The brightness of a pixel in a channel indicates the presence and proximity of that entity type.

##### Visualization

The `visualize_obs.py` script provides an intuitive rendering of this observation, which is invaluable for debugging and analysis.

<img src="https://github.com/user-attachments/assets/121b1836-8e58-45e0-8433-317281f63004" width="500" alt="Raw Map Visualization" />

-   **High-Resolution Radius**: A smooth white circle is overlaid to clearly show the exact boundary of the agent's observation radius.
-   **Descriptive Textbox**: A summary box explains the key features of the observation for clarity.
-   **Color-Coded Legend**: A legend at the bottom explains the color for each entity type, making the tactical situation easy to understand at a glance.

#### 3. Memory Map: Persistent Spatial Awareness
The memory map provides persistent, allocentric (world-fixed) spatial awareness, functioning as a long-term memory of the world's layout. At each step, the agent's live `raw_map` is used to update this persistent world model. A key feature of this system is its simulation of memory decay; instead of simple presence/absence, the map stores timestamps of the last observation, which are converted into a normalized "age." This allows the agent's policy to reason about the uncertainty of older information. The final map provided to the policy is a rich, multi-channel tensor that includes these age-based entity locations, static obstacles, and global context channels. For a detailed breakdown of its architecture and channels, see the `Swarm2d/env/README.md`.

#### 4. Hierarchical Graph Memory with Dynamic Edge Attributes
This is the most powerful component of the agent's perception. It represents the world as a structured, hierarchical network of entities, enabling sophisticated relational reasoning. This system is designed to provide maximum detail for the agent's immediate surroundings (the fovea) while intelligently abstracting distant information (the periphery) to maintain computational efficiency.

*   **The Fovea: A High-Detail View of the Present**:
    *   **Fine-Grained Nodes**: All entities within the agent's current observation radius are represented as individual, un-clustered nodes, providing a high-fidelity view of the immediate environment.
    *   **Rich Local Connections**: Edges within the fovea are created using a `radius_graph`, connecting entities based on spatial proximity. This builds a rich, relational web that captures the tactical layout of the agent's surroundings.

*   **The Periphery: An Abstracted Map of Long-Term Memory**:
    *   **Adaptive Clustering**: To keep the graph manageable, memories outside the fovea are clustered. This process is adaptive: memories just outside the agent's sight are grouped into small clusters, while very distant memories are aggregated into much larger, more abstract "super-nodes." This mimics the way memory becomes less granular with distance.
    *   **Sparse Memory Skeleton**: The resulting clusters are connected to each other with a localized radius proportional to their size. This creates a clean, sparse "skeleton" of the agent's long-term spatial knowledge, avoiding the "hairball" of a fully-connected graph.

*   **Hierarchical Bridge Connections**: The fovea and periphery are linked in an intelligent, content-aware manner. Instead of connecting all fovea nodes to all nearby memory clusters, the agent's own "ego" node creates sparse connections to a few key "bridge" nodes in the periphery. These bridges are selected based on both proximity and importance, prioritizing the closest clusters as well as the nearest known enemy and resource clusters.

*   **Dynamic Edge Attributes**: Every edge in the final graph is enriched with a dynamic attribute vector that provides critical relational context:
    1.  **Relative Position Vector (`dx`, `dy`):** A normalized vector from the source to the destination node.
    2.  **Normalized Distance:** The scalar distance between the nodes.
    3.  **Connection Strength:** A value from 0.0 to 1.0 calculated from the "certainty" of the two connected nodes. An edge between two live observations has a strength of 1.0, while an edge to a fading memory is weaker, providing a powerful signal of information reliability.

##### Visualization of the Unified Graph

The `visualize_obs.py` script also provides a powerful 3D visualization of the unified graph, which is essential for understanding the agent's complex memory state.

<img src="https://github.com/user-attachments/assets/b8392131-a01c-4340-9a3c-b2582877a5d3" width="700" alt="Graph Visualization" />

-   **Spatial Layout (X, Y):** Nodes are plotted in 2D space according to their world coordinates.
-   **Certainty (Z-axis):** The height of a node on the Z-axis represents its "certainty," a metric combining its recency and whether it's part of a cluster (persistent patterns are more certain).
-   **Memory Decay (Transparency):** The transparency of a node is directly tied to the age of the observation. Live nodes in the fovea are fully opaque, while older memories in the periphery appear faded.
-   **Clustering (Node Size):** The size of a node is proportional to the number of individual memories it represents. Large spheres are "super-nodes" that abstract many distant observations.
-   **Team & Type Coloring:** Nodes are color-coded based on their team (for agents and hives) or entity type (for resources and obstacles), providing an at-a-glance overview of the strategic landscape.
-   **Semantic Edges:** Edges are colored based on the relationship they represent (e.g., team affiliation, combat, relative velocity), revealing the rich relational information available to the policy.

## Information Dynamics and Coordination

#### Occlusion and Line-of-Sight
Occlusion is implemented via PyBullet's efficient batched raycasting to provide realistic information constraints. An agent's Line-of-Sight (LOS) to a target is clear only if a ray cast between them hits nothing or hits the target entity first. If the ray hits an obstacle or another agent before the target, the target is considered occluded. Agents have a 360-degree sensor radius, not a limited frontal cone, creating an effective "fog of war" where knowledge is limited to current LOS observations and decaying memory.

#### Implicit Coordination and Communication
The codebase does not contain a mechanism for direct, explicit message passing between agents. Communication is entirely implicit, emerging from agents observing the state and actions of their teammates. The graph-based observation is critical for this.

*   **Direct Relational Reasoning**: The graph structure explicitly connects an agent to its teammates, allowing a Graph Neural Network (GNN) policy to reason about teammate states and their spatial relationships to enemies or resources.
*   **Shared Perception of Team State**: Because agents in the same area have overlapping fields of view, their memory graphs are structurally similar. When one agent acts, the change is observed by nearby teammates, whose policies can then react to the updated state, creating a feedback loop where actions become signals.
*   **Strategic Cues from Adaptive Clustering**: The clustering of peripheral memory provides powerful strategic cues. An agent's graph might contain a "super node" representing a distant enemy cluster, allowing the policy to learn macro-level strategies like regrouping before engaging, which can propagate through the team as other agents observe this behavior.

## Modular Reward System

The environment utilizes a comprehensive and modular reward system to encourage complex behaviors. Each reward is composed of a `default_value` and a `default_multiplier`, allowing for fine-tuning and curriculum learning. Rewards are categorized as follows:

*   **Resource Collection & Delivery**: Rewards for attaching to a resource (`r_attachment`), making progress towards a hive while carrying (`r_progress`), and successfully delivering it (`r_delivery`). A bonus multiplier is applied for delivering cooperative resources (`coop_collection_bonus`).
*   **Grappling**: Continuous rewards for maintaining a grapple (`r_grapple_control`) and penalties for being grappled (`r_grapple_controlled`). One-time rewards are given for breaking free (`r_grapple_break`) and for overpowering an opponent's torque (`r_torque_win`).
*   **Combat & Agent Survival**: Rewards for winning a combat encounter (`r_combat_win`) and penalties for losing (`r_combat_lose`) or dying (`r_death`). A continuous reward is given for dealing damage (`r_combat_continuous`), and a penalty is applied when a nearby teammate is lost (`r_teammate_lost_nearby`).
*   **Hive Control**: Continuous rewards for attacking an enemy hive (`r_hive_attack_continuous`) and for the team's overall hive health (`r_hive_health_continuous`). Large one-time rewards are given for capturing an enemy hive (`r_hive_capture`) or rebuilding a friendly one (`r_hive_rebuild`), with a large team-wide penalty when a hive is destroyed (`r_hive_destroyed_penalty`).
*   **Discovery & Exploration**: To encourage information gathering, small rewards are given for discovering resources (`r_resource_found`), obstacles (`r_obstacle_found`), enemies (`r_enemy_found`), and enemy hives (`r_hive_found`). An intrinsic exploration reward (`r_exploration_intrinsic`) encourages agents to visit new or rarely-visited areas.

## Agent Policies

The behavior of agents is determined by a selection of modular policies, ranging from simple baselines to sophisticated MARL systems, allowing for comparative analysis.

#### SimpleRL Baseline
A fully decentralized and independent learner, where each agent uses an identical Multi-Layer Perceptron (MLP) network. It is trained using the classic REINFORCE algorithm and serves as a crucial benchmark to measure the performance gains of more complex policies.

#### NCA-PINSAN (Physics-Informed Neuro-Symbolic Adaptive Neighborhood Neural Cellular Automata)
A novel MARL architecture inspired by Neural Cellular Automata, where agent states evolve based on local interactions.
*   **Physics-Informed**: A GNN uses physics-based edge features (relative distance, velocity) to ground communication in reality.
*   **Neuro-Symbolic**: Features a "symbolic head" that learns to output latent "roles," allowing agents to convey abstract intent.
*   **Adaptive Neighborhood**: A Graph Attention Network (GATv2) allows agents to weigh the importance of messages from neighbors.
*   **Memory**: Features a dual memory system with a GRU-based internal belief state and a small, addressable external memory (NTM-like) that allows agents to perform gated reads from neighbors' memories.

#### MAAC (Multi-Agent Actor-Critic) Policy
A robust architecture designed around role specialization, where agents are assigned fixed roles (e.g., scout, collector) for an episode.
*   **Observation Handling**: It fuses multi-modal data using three distinct processing streams: an MLP for the self-observation vector, a multi-scale CNN for the map, and a GNN for the unified entity graph.
*   **Memory**: A custom `TitanTransformerEncoderLayer` implements episodic memory, allowing the policy to attend to a cached history of its own previous states for a longer temporal context.
*   **Communication**: Handled primarily by the GNN operating on the unified graph, enabling coordination based on both real-time and historical context.

#### SharedAgent Policy
The most feature-rich policy, integrating multiple advanced concepts for a high degree of emergent, decentralized coordination.
*   **Unique Communication Mechanisms**:
    *   **GNN for Local Exchange**: For direct, real-time message passing.
    *   **Decentralized Coordinator**: A dynamic "leader election" protocol where agents in a local "squad" elect a leader who broadcasts a shared context vector.
    *   **Information Trails (Stigmergy)**: Agents can leave persistent "information trails" (semantic vectors) in the environment that other agents can read, allowing for asynchronous communication.
*   **Memory and Control**: A high-level Transformer Encoder provides long-term temporal memory and produces a latent "plan" vector. This plan conditions a low-level controller that generates the final actions, creating a hierarchical control system.

## The Training Pipeline

The project provides a flexible training pipeline split between a general-purpose template (`Swarm2d/training`) for new users and an advanced pipeline (`Swarm2d/trainingCustom`) used for core research. Both are supported by a suite of powerful helper modules:

*   **Checkpointing (`checkpointing.py`)**: Manages saving and loading the complete state of a training session (model weights, optimizer states, replay buffers), allowing for the seamless resumption of interrupted runs.
*   **Curriculum Learning (`curriculum_learning.py`)**: Implements a phased approach where agents are progressively trained on more difficult tasks. The curriculum advances from basic survival to complex resource gathering and combat when an agent's performance on the current stage has plateaued.
*   **Logging Utilities (`log_utils.py`)**: A comprehensive logging solution that integrates with TensorBoard to track and visualize a wide array of metrics in real-time, such as rewards, losses, and episode lengths.
*   **Plateau Management (`PlateauManager.py`)**: Automates the training process by detecting when a team's performance has stalled over a window of episodes, serving as the primary trigger for advancing the curriculum stage.

## Observation Visualization Tools

To aid in debugging, analysis, and understanding agent behavior, the project includes a visualization script (`visualize_obs.py`). This script can generate detailed images and GIFs of an agent's complex observations, including the `raw_map`, `memory_map`, and `graph`. The `memory_map` visualization is particularly informative.

<img src="https://github.com/user-attachments/assets/b8352b02-5a41-4775-926e-4f1076af2dd3" width="700" alt="Memory Map Visualization" />

### Features of the Memory Map Visualization

The visualization for the `memory_map` is not just a direct printout of the tensor; it's a carefully rendered composite image designed to be intuitively understood.

-   **Persistent Ground:** The map displays a persistent ground layer for all areas the agent has ever explored. Unexplored areas remain dark, creating a natural "fog of war" effect.
-   **Time-Based Fading (Memory Decay):** The visualization directly reflects the agent's fading memory. The brightness of dynamic entities (like other agents and resources) is determined by the **age** of the observation. A recently seen enemy will appear bright and vibrant, while an old observation will appear dimmer and more transparent. This provides an immediate visual cue for the reliability of the information.
-   **Density Modulation (Coverage Map):** The `coverage_map` (representing entity density) is used to modulate the brightness of entities. Dense clusters of remembered objects will appear brighter than sparse, individual ones. This helps to highlight areas of high activity or strategic importance.
-   **Color-Coded Entities:** All entities are rendered with distinct, intuitive colors (e.g., red for enemies, blue for allies, green for resources), making the tactical situation easy to parse at a glance.
-   **Clear Agent Marker:** The observing agent is always clearly marked with a white pixel at its center, providing a constant point of reference.

These features combine to create a rich, informative visual representation that goes beyond simple debugging, offering deep insights into the agent's perceptual state and memory dynamics.


## Rendering and Visualization GUI

The Swarm2d environment features a dual-mode rendering system and a comprehensive Graphical User Interface (GUI) for configuration, interaction, and analysis.

#### Dual-Mode Rendering System
While the simulation leverages a 3D physics engine, agent logic is confined to a 2D plane. Two primary rendering options are available:
*   **PyBullet GUI (`render_mode='gui'`)**: Launches PyBullet's built-in GUI, providing a direct, top-down 3D view of the physics simulation. This is invaluable for debugging physical interactions.
*   **Pygame Display (`render_mode='human'`)**: Provides a clean, 2D, top-down schematic visualization. It renders agents as team-colored circles with health/energy bars, visualizes observation radii, and distinguishes between individual (green) and cooperative (orange) resources.

#### The Interactive Simulation GUI
Built with PyQt6, the GUI is the primary tool for exploring the environment's capabilities without modifying code.
*   **Deep Configuration**: Offers an extensive suite of options organized into tabs (General, Physics, Spawning, Constants, Observations, Rewards), allowing for customization of nearly every simulation parameter.
*   **Asymmetric Scenario Creation**: The "Team Setup" tab allows for per-team overrides of most parameters and policies, enabling the creation of customized scenarios pitting different agent types or strategies against each other.
*   **Simulation and Analysis**: Provides controls to run and manage the simulation, which operates in a separate thread for a responsive interface. Configurations can be saved to and loaded from JSON preset files. A real-time plotting panel displays crucial metrics for monitoring team performance.
*   **Policy Debugging**: While training is done headlessly for efficiency, the GUI is the ideal environment for visualizing, debugging, and verifying the behavior of trained policies.
<img src="https://github.com/user-attachments/assets/c1033646-2fc1-4073-ab5d-daa5129eb764" width="700" alt="PyBullet Rendering" />

<img src="https://github.com/user-attachments/assets/cc44c300-68eb-4ddf-9711-4f71a5d1f239" width="600" alt="GUI 2" />
<img src="https://github.com/user-attachments/assets/8ff01e13-a669-484d-9bf2-2114baa2eae0" width="600" alt="GUI 3" />




## Getting Started

This guide will walk you through setting up the Swarm2d environment on your local machine.

#### 1. Installation
First, clone the project repository from GitHub and install the required Python dependencies.

*   **Step 1: Clone the Repository**
    Open your terminal, navigate to your desired directory, and run:
    ```bash
    git clone https://github.com/your-username/Swarm2dSimple.git
    cd Swarm2dSimple
    ```

*   **Step 2: Install Dependencies**
    It is highly recommended to use a virtual environment. Install all dependencies from `requirements.txt`:
    ```bash
    pip install -r Swarm2d/requirements.txt
    ```

#### 2. Quick Start: The Simulation GUI
The easiest way to explore the environment is through the GUI.

*   **How to Run It:**
    Navigate to the `Swarm2d` directory and run the `simulation_gui.py` script:
    ```bash
    cd Swarm2d
    python simulation_gui.py
    ```
    This will launch the application, where you can configure and run simulations visually.

<img src="https://github.com/user-attachments/assets/f9808ba6-58fe-47a6-bd24-5521ec20e6ac" width="600" alt="GUI Metrics" />


#### 3. Example Script: `test_env.py`
This script demonstrates programmatic interaction with the environment without the GUI. Its purpose is to initialize the environment with a hard-coded configuration, run a simulation loop with random actions, and perform a deep inspection of the observation data structure, printing the output to the console and `test_output.log`.

*   **How to Run It:**
    From the `Swarm2d` directory, run the script:
    ```bash
    python test_env.py
    ```

#### 4. Training a New Policy
The primary entry point for training is the `main.py` script in the `Swarm2d/training/` directory.

*   **Training Loop Logic**:
    1.  **Configuration**: Training parameters are set via command-line arguments (e.g., `--num_episodes`, `--learning_rate`).
    2.  **Environment Setup**: Initializes a 2-team (e.g., 5v5) environment where Team 0 is controlled by the learning agents and Team 1 by a static opponent policy (e.g., `HeuristicPolicy`).
    3.  **Policy Initialization**: A learning policy (e.g., `SimpleRLPolicy`) is instantiated for Team 0.
    4.  **Episode Execution**: For each episode, the script collects trajectories of observations, actions, and rewards for the learning agents.
    5.  **Policy Update**: At the end of an episode, the policy's `update` method is called to compute a loss from the collected trajectories and perform a gradient update.
    6.  **Logging and Checkpointing**: Key metrics are logged, and the policy network is saved periodically, especially when the average reward improves.

*   **How to Run Training:**
    To start a training run for 5000 episodes with rendering enabled, navigate to the project's root directory and run:
    ```bash
    python Swarm2d/training/main.py --num_episodes 5000 --render
    ```

## Acknowledgements & Gratitude

This project stands on the shoulders of giants. The entire Swarm2d ecosystem would not be possible without the incredible work of the open-source community and the powerful, permissively licensed libraries that provide the foundation for modern scientific computing and machine learning research. I extend our deepest gratitude to the developers, maintainers, and contributors of these essential tools. I have made every effort to use these libraries in accordance with their licenses. Below is a list of the key projects that power Swarm2d:

*   **PyTorch & PyTorch Geometric**: These projects form the core of our reinforcement learning pipeline. `PyTorch` provides the fundamental tensor computation and automatic differentiation capabilities, while `PyTorch Geometric` enables the seamless implementation of graph neural networks, which are central to our agents' observation and decision-making processes.
    *   *License: Both are licensed under a permissive, BSD-style license.*

*   **PyBullet**: The simulation's physics, including all collisions, forces, and movements, are handled by the `PyBullet` physics engine. Its robust and high-performance capabilities allow for complex, realistic interactions between agents and objects in the environment.
    *   *License: PyBullet is provided under the liberal Zlib license.*

*   **Gymnasium**: Our environment is structured to be compliant with the `Gymnasium` API (the successor to OpenAI Gym). This ensures compatibility with the broader reinforcement learning ecosystem and provides a standardized interface for environment interaction.
    *   *License: Gymnasium is licensed under the MIT License.*

*   **NumPy & SciPy**: These libraries are the bedrock of numerical and scientific computing in Python. `NumPy` is used extensively for efficient array manipulation and mathematical operations, while `SciPy` provides a collection of algorithms for optimization and scientific computing.
    *   *License: Both projects are governed by a permissive BSD-style license.*

*   **PyQt6 & pyqtgraph**: The detailed and interactive graphical user interface (GUI) is built using `PyQt6`, a set of Python bindings for the powerful Qt application framework. The real-time plotting of metrics within the GUI is handled by `pyqtgraph`.
    *   *License: PyQt6 is available under the GNU General Public License (GPL), which means that any derivative work must also be open-sourced under the GPL. `pyqtgraph` is licensed under the MIT license.*

*   **Pygame**: The rendering of the environment, both in the GUI and in headless "human" render mode, is managed by `Pygame`. It provides a straightforward way to draw the state of the simulation to the screen.
    *   *License: Pygame is distributed under the GNU Lesser General Public License (LGPL), which allows it to be linked with projects that have different licenses.*

*   **Matplotlib & TQDM**: For generating plots for analysis and for providing clear, helpful progress bars during training and data processing, I rely on `Matplotlib` and `TQDM`, respectively.
    *   *License: Both Matplotlib and TQDM are available under permissive licenses (a BSD-style license for Matplotlib and the MIT License for TQDM).*

Thank you once again to the entire open-source community. Your collective effort makes complex research projects like this one achievable.



This repository contains a multi-agent reinforcement learning (MARL) environment and training scripts for swarm robotics scenarios.

## Project Structure

- **Swarm2d/**: The core environment and policy packages.
  - `env/`: Environment logic (physics, rendering, rewards).
  - `policies/`: Agent policies (RL, Heuristic).
  - `training/`: Training utilities and trainer classes.

- **scripts/**: Executable scripts for training, evaluation, and visualization.
  - `advanced_training/`: **Primary training scripts** for advanced models (NCA, MAAC, Shared GNN).
    - `train.py`: Main entry point for advanced CTDE training.
  - `simple_training/`: Training scripts for simpler map-based observations.
    - `train.py`: Entry point for map-based PPO training.
  - `legacy/`: Older training scripts (Graph, Single Agent, etc.).
  - `visualization/`: Scripts to visualize agent behavior.
    - `visualize_policy.py`: Watch a trained agent.
    - `generate_heatmap.py`: Generate spatial heatmaps of agent exploration.
    - `record_video.py`: Record gameplay videos.
  - `evaluation/`: Scripts for benchmarking and metrics.
    - `run_competition.py`: Run competitions between policies.
    - `benchmark_heuristics_fast.py`: Benchmark heuristic baselines.

- **results/**: Output directory for training logs, checkpoints, and media.
  - `logs/`: Training logs (CSV/JSON).
  - `media/`: Generated videos and plots.
  - `rl_training_results_*/`: Checkpoints from training runs.

## Usage

### Advanced Training (Recommended)

To train advanced agents (MAAC, NCA, Shared) with full observation modalities:
```bash
python scripts/advanced_training/train.py
```

### Simple Training

To train a baseline map-based agent:
```bash
python scripts/simple_training/train.py --scenario unified
```

### Visualization

To visualize a trained policy:
```bash
python scripts/visualization/visualize_policy.py --scenario resource --checkpoint results/rl_training_results_cnn_resource_fast_v4/checkpoint_latest.pt
```

### Evaluation

To run a heuristic benchmark:
```bash
python scripts/evaluation/benchmark_heuristics_fast.py
```

## Installation

Ensure you have the required dependencies installed (see `requirements.txt`).
Run scripts from the project root directory. The scripts automatically add the project root to `sys.path`.
