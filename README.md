# Deep Reinforcement Learning for Real-World TurtleBot Navigation

This repository contains the implementation for a project focused on applying Deep Reinforcement Learning (DRL) to real-world mobile robot navigation using a TurtleBot and ROS2.

## Overview

The project explores sim-to-real transfer of DRL policies for indoor navigation tasks. A robot is trained in simulation to reach a goal location from arbitrary starting positions while avoiding obstacles. The trained policy is then deployed and evaluated on a physical TurtleBot platform.

## Methodology

1. **Simulation Training**  
   Training is performed in Gazebo Classic with ROS2 using DRL algorithms (e.g., DQN or PPO). The robot learns through trial-and-error interactions with the environment.

2. **Custom Environment Interface**  
   A Gym-like interface is created to expose sensor data (LiDAR, odometry) and accept discrete velocity commands. The reward function encourages goal-directed, collision-free, and smooth behavior.

3. **Sim-to-Real Transfer**  
   After training, the policy is transferred to the real TurtleBot and tested under real-world conditions. Fine-tuning may be performed to adapt to hardware-specific dynamics and noise.

## Setup Instructions

### Clone the repository

```bash
git clone git@github.com:LevinTamir/planning-and-rl-course-project.git
cd planning-and-rl-course-project
```

### Add the following lines to your `.bashrc`

```bash
# Sourcing ROS2 & local workspace
source /opt/ros/humble/setup.bash
source <path to your ws>/install/setup.bash

# TurtleBot3 environment setup
export TURTLEBOT3_MODEL=waffle_pi
# export TURTLEBOT3_MODEL=burger

# ROS domain configuration
export ROS_DOMAIN_ID=30  # TURTLEBOT3

# Source Gazebo Classic environment
source /usr/share/gazebo/setup.bash

# Add Gazebo plugin path
export GAZEBO_PLUGIN_PATH=$HOME/ros2_workspaces/turtlebot3_drl_ws/build/turtlebot3_gazebo:$GAZEBO_PLUGIN_PATH
```

Apply the changes by running:

```bash
source ~/.bashrc
```

> More detailed instructions for training and running inference will be added as the implementation progresses.
