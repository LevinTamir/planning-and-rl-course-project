# Deep Reinforcement Learning for Real-World TurtleBot Navigation

This repository contains the project for "Planning and Reinforcement Learning 364-2-1031" course @ BGU.

## Overview

The project explores DRL policies for indoor navigation tasks. A robot is trained in simulation to reach a goal location from arbitrary starting positions while avoiding obstacles. The trained policy is then deployed and evaluated.

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

## Running the DRL Training and Testing

### Terminal #1: Launch Gazebo Stage Environment

First, bring up the Gazebo simulation environment with the stage map:

```bash
$ ros2 launch turtlebot3_gazebo turtlebot3_dqn_{stage_num}.launch.py
```

This launches the Gazebo world with the TurtleBot3 and sets up the environment for training or testing.

### Terminal #2: Run Gazebo Environment Node

Launch the Gazebo environment management node that handles goal regeneration and TurtleBot position initialization:

```bash
$ ros2 run turtlebot3_dqn dqn_gazebo {stage_num}
```

This node manages the Gazebo environment, regenerates goals, and initializes the TurtleBot's location when episodes start.

### Terminal #3: Run DQN Environment Node

Start the DQN environment node that calculates TurtleBot states and determines rewards, success, and failure conditions:

```bash
$ ros2 run turtlebot3_dqn dqn_environment
```

This node is responsible for state calculation and reward computation based on the TurtleBot's performance.

### Terminal #4: Training or Testing

#### For Training:
Run the DQN agent node to train the TurtleBot with calculated rewards:

```bash
$ ros2 run turtlebot3_dqn dqn_agent {stage_num} {max_training_episodes}
```

This trains the TurtleBot using DQN algorithm with the specified number of maximum training episodes.

#### For Testing:
After training, test the trained model by running the test node instead of the agent node:

```bash
$ ros2 run turtlebot3_dqn dqn_test {stage_num} {load_episode}
```

This loads a previously trained model from the specified episode and runs it in test mode.

### Terminal #5: Action Graph Visualization (Optional)

To visualize the TurtleBot's actions and rewards in real-time:

```bash
$ ros2 run turtlebot3_dqn action_graph
```

This opens the Action State window showing:
- Current TurtleBot actions (Left, Front, Right)
- Reward values for each action
- Total accumulated reward in the episode

The graph displays real-time feedback on the robot's decision-making process during training or testing.
