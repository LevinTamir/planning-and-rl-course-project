# Deep Reinforcement Learning for Real-World TurtleBot Navigation

This repository contains the implementation for a project focused on applying Deep Reinforcement Learning (DRL) to real-world mobile robot navigation using a TurtleBot and ROS2.

## Overview

The project explores sim-to-real transfer of DRL policies for indoor navigation tasks. A robot is trained in simulation to reach a goal location from arbitrary starting positions while avoiding obstacles. The trained policy is then deployed and evaluated on a physical TurtleBot platform.

## Methodology

1. **Simulation Training**  
   Training is performed in Gazebo Classic with ROS2 using DQN algorithm. The robot learns through trial-and-error interactions with the environment.

2. **Custom Environment Interface**  
   A Gym-like interface is created to expose sensor data (LiDAR, odometry) and accept discrete velocity commands. The reward function encourages goal-directed, collision-free, and smooth behavior.

3. **Sim-to-Real Transfer**  
   After training, the policy is transferred to the real TurtleBot and tested under real-world conditions. Fine-tuning may be performed to adapt to hardware-specific dynamics and noise.

## Setup Instructions

The following section provides steps to clone and run the project:

```bash
# Clone the repository
git clone <repo-url>
cd <repo-directory>

# Build or install dependencies (to be completed)

# Run the training or deployment scripts (to be completed)
```

More detailed instructions will be added as the implementation progresses.
