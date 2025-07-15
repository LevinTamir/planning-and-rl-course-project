#!/usr/bin/env python3

import collections
import json
import os
import random
import sys
import time
import datetime
import math
from std_msgs.msg import Float32MultiArray

from keras.api.layers import Dense
from keras.api.models import load_model
from keras.api.models import Sequential
from keras.api.optimizers import RMSprop
import numpy
import rclpy
from rclpy.node import Node
import tensorflow

from turtlebot3_msgs.srv import Dqn
LOGGING = True # flag for saving data from runs into logger
current_time = datetime.datetime.now().strftime("[%mm%dd-%H:%M]") # for saving data

# setting a class that track the loss function which is the score in our case 
# through the steps in each episode
class DQNMetric(tensorflow.keras.metrics.Metric):

    def __init__(self, name="dqn_metric"): # initialzition and definition function
        super(DQNMetric, self).__init__(name=name) # initialize as ROS node
        self.loss = self.add_weight(name="loss", initializer="zeros") # defining loss field, initialize it with zeros
        self.episode_step = self.add_weight(name="step", initializer="zeros") # defining episode step field, initialize it with zeros

    def update_state(self, y_true, y_pred=0, sample_weight=None): # updating loss and step each step
        self.loss.assign_add(y_true) # adding score into the loss field vairable
        self.episode_step.assign_add(1) # forwording the step

    def result(self): # returing avarge score by normalizing it by the steps
        return self.loss / self.episode_step

    def reset_states(self): # reseting for new episode (not for step)
        self.loss.assign(0)
        self.episode_step.assign(0)

# creating node for test model
class DQNTest(Node):

    def __init__(self, stage, load_episode):
        super().__init__("dqn_test") # initialize node
	#defining parameters for rl_agent progress including step, action and reward calculation
        self.stage = int(stage)
        self.state_size = 26
        self.action_size = 5
        self.episode_size = 3000

        self.discount_factor = 0.99 # needed for reward calculation
        self.learning_rate = 0.00025  # needed for optimizer in the build model
        self.epsilon = 1.0 # needed for small exploration in get action
        self.epsilon_decay = 0.99 
        self.epsilon_min = 0.05
        #self.batch_size = 64 # not needed
        #self.train_start = 64 # not needed

        #self.memory = collections.deque(maxlen=1000000) # not needeed

        self.model = self.build_model() # builidng model architecture, then inserting to it weights
        self.target_model = self.build_model()
	# direction to load model from
        self.load_model = True
        self.load_episode = int(load_episode)
        #self.model_dir_path = os.path.join(os.path.dirname(__file__), "..", "saved_model")
        self.model_dir_path = "/home/ido/ros2_workspaces/TurtleBot_DRL_Workspace/src/turtlebot3_machine_learning/turtlebot3_dqn/saved_model/learning_rate_0.002"
        self.model_path = os.path.join(
            self.model_dir_path, f"stage{self.stage}_episode{self.load_episode}.h5"
        )
	# loading model and weights
        if self.load_model:
            loaded_model = load_model(
                self.model_path,
                custom_objects={"mse": tensorflow.keras.losses.MeanSquaredError()},
            )
            self.model.set_weights(loaded_model.get_weights())
            with open(
                os.path.join(
                    self.model_dir_path,
                    "stage"
                    + str(self.stage)
                    + "_episode"
                    + str(self.load_episode)
                    + ".json",
                )
            ) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get("epsilon")
                # saving data into loggers into the desired dir
        if LOGGING:
            tensorboard_file_name = (
                current_time + " dqn_stage" + str(self.stage) + "_reward"
            )
            dqn_reward_log_dir = "logs/gradient_tape/" + tensorboard_file_name
            self.dqn_reward_writer = tensorflow.summary.create_file_writer(
                dqn_reward_log_dir
            )
            self.dqn_reward_metric = DQNMetric() # saving the class into logger including the score and steps
	# rl interface to opertae robot
        self.rl_agent_interface_client = self.create_client(Dqn, "rl_agent_interface")
        # publisher to plots
        self.action_pub = self.create_publisher(Float32MultiArray, "/get_action", 10) # publisher that send orders for get_action topic
        self.result_pub = self.create_publisher(Float32MultiArray, "result", 10) # publisher that send the score to result topic
	# starting 
        self.process()

    def process(self):
        global_step = 0
	# looping throught epsiodes and do action based on trained model
        for episode in range(self.load_episode + 1, self.episode_size):
            global_step += 1
            local_step = 0

            state = []
            next_state = []
            done = False
            init = True
            score = 0

            time.sleep(1.0)

            while not done:
                local_step += 1
                if local_step == 1: # in first step always get 0 angular velocity
                    action = 2
                else: # getting action from trained model
                    state = next_state
                    action = int(self.get_action(state))

                req = Dqn.Request() # getting action from rl service
                print(int(action))
                req.action = action
                req.init = init

                while not self.rl_agent_interface_client.wait_for_service( # waiting for service
                    timeout_sec=1.0
                ):
                    self.get_logger().info(
                        "rl_agent interface service not available, waiting again..."
                    )
                future = self.rl_agent_interface_client.call_async(req) # send action
                rclpy.spin_until_future_complete(self, future)
                  
              
                while rclpy.ok():
                    rclpy.spin_once(self)

                    if future.done():

                        if future.result() is not None: # updating the state, reward and environment after taking an action
                            # Next state and reward
                            next_state = future.result().state
                            reward = future.result().reward
                            done = future.result().done
                            score += reward
                            init = False
                            
                        else:
                            self.get_logger().error(
                                "Exception while calling service: {0}".format(
                                    future.exception()
                                )
                            )

                        break

                        # avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0
                        
                        

                time.sleep(0.01)
            # publishing msg to plot
            msg = Float32MultiArray()
            msg.data = [float(action), float(score), float(reward)]
            self.action_pub.publish(msg)
            avg_max_q = 1
            msg = Float32MultiArray()
            msg.data = [float(score), float(avg_max_q)]
            self.result_pub.publish(msg)
            if LOGGING:
                        self.dqn_reward_metric.update_state(score)
                        with self.dqn_reward_writer.as_default():
                            tensorflow.summary.scalar(
                                "dqn_reward",
                                self.dqn_reward_metric.result(),
                                step=episode,
                            )
                        self.dqn_reward_metric.reset_states()
    def build_model(self): # building model architecureand inserting to it the weights from trained model
        model = Sequential()
        model.add(
            Dense(
                512,
                input_shape=(self.state_size,),
                activation="relu",
                kernel_initializer="lecun_uniform",
            )
        )
        model.add(Dense(256, activation="relu", kernel_initializer="lecun_uniform"))
        model.add(Dense(128, activation="relu", kernel_initializer="lecun_uniform"))
        model.add(
            Dense(
                self.action_size,
                activation="linear",
                kernel_initializer="lecun_uniform",
            )
        )
        model.compile(
            loss="mse",
            optimizer=RMSprop(learning_rate=self.learning_rate, rho=0.9, epsilon=1e-06),
        )
        model.summary()

        return model

    def get_action(self, state): # function to get action from model prediction and with little exploration
        #if numpy.random.rand() <= self.epsilon: # CANCELING EXPLORATION
        #    return random.randrange(self.action_size)
        #else:
            state = numpy.asarray(state)
            q_value = self.model.predict(state.reshape(1, len(state)))
            print(numpy.argmax(q_value[0]))
            return numpy.argmax(q_value[0])


def main(args=None):
    if args is None:
        args = sys.argv
    stage = args[1] if len(args) > 1 else "1"
    load_episode = args[2] if len(args) > 2 else "600"
    rclpy.init(args=args)
    dqn_test = DQNTest(stage, load_episode)
    try:
        while rclpy.ok():
            rclpy.spin_once(dqn_test, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        dqn_test.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
