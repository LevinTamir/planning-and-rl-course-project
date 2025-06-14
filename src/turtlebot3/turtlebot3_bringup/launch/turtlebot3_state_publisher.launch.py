#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    TURTLEBOT3_MODEL = os.environ["TURTLEBOT3_MODEL"]

    namespace = LaunchConfiguration("namespace")

    use_sim_time = LaunchConfiguration("use_sim_time", default="false")
    urdf_file_name = "turtlebot3_" + TURTLEBOT3_MODEL + ".urdf"

    print("urdf_file_name : {}".format(urdf_file_name))

    urdf = os.path.join(
        get_package_share_directory("turtlebot3_description"), "urdf", urdf_file_name
    )

    robot_desc = Command(
        [
            "xacro ",
            urdf,
            " namespace:=",
            PythonExpression(
                ['"', namespace, '" + "/" if "', namespace, '" != "" else ""']
            ),
        ]
    )

    # Major refactor of the robot_state_publisher
    # Reference page: https://github.com/ros2/demos/pull/426

    rsp_params = {"robot_description": robot_desc}

    # print (robot_desc) # Printing urdf information.

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation (Gazebo) clock if true",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[rsp_params, {"use_sim_time": use_sim_time}],
            ),
        ]
    )
