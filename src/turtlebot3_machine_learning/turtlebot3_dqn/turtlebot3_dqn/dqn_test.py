#!/usr/bin/env python3

import collections
import json
import os
import random
import sys
import time

from keras.api.layers import Dense
from keras.api.models import load_model
from keras.api.models import Sequential
from keras.api.optimizers import RMSprop
import numpy
import rclpy
from rclpy.node import Node
import tensorflow

from turtlebot3_msgs.srv import Dqn


class DQNTest(Node):

    def __init__(self, stage, load_episode):
        super().__init__("dqn_test")

        self.stage = int(stage)
        self.state_size = 26
        self.action_size = 5
        self.episode_size = 3000

        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        # self.epsilon = 1.0
        # self.epsilon_decay = 0.99
        # self.epsilon_min = 0.05
        # self.batch_size = 64 # Not needed
        # self.train_start = 64 # Not needed

        # self.memory = collections.deque(maxlen=1000000)

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.load_model = True
        self.load_episode = int(load_episode)
        self.model_dir_path = os.path.join(os.path.dirname(__file__), "..", "saved_model")
        self.model_path = os.path.join(
            self.model_dir_path, f"stage{self.stage}_episode{self.load_episode}.h5"
        )

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

        self.rl_agent_interface_client = self.create_client(Dqn, "rl_agent_interface")

        self.process()

    def process(self):
        global_step = 0

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
                if local_step == 1:
                    action = 2
                else:
                    state = next_state
                    action = int(self.get_action(state))

                req = Dqn.Request()
                print(int(action))
                req.action = action
                req.init = init

                while not self.rl_agent_interface_client.wait_for_service(
                    timeout_sec=1.0
                ):
                    self.get_logger().info(
                        "rl_agent interface service not available, waiting again..."
                    )
                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                while rclpy.ok():
                    rclpy.spin_once(self)

                    if future.done():

                        if future.result() is not None:
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

                time.sleep(0.01)

    # Only for creating the model, The weights are loaded later.
    def build_model(self):
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

    def get_action(self, state):
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
