"""
This demo script demonstrates the various functionalities of each controller available within robosuite.

For a given controller, runs through each dimension and executes a perturbation "test_value" from its
neutral (stationary) value for a certain amount of time "steps_per_action", and then returns to all neutral values
for time "steps_per_rest" before proceeding with the next action dim.

Please reference the documentation of Controllers in the Modules section for an overview of each controller.
Controllers are expected to behave in a generally controlled manner, according to their control space.
"""

import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.utils.transform_utils import convert_quat, pose2mat, get_pose_error

def get_gripper_eef_pose(env, joint_angles):
    env.robots[0].set_robot_joint_positions(joint_angles)
    gripper_pose = (
        env.robots[0].sim.data.get_body_xpos('right_hand'),
        convert_quat(env.robots[0].sim.data.get_body_xquat('right_hand'), to='xyzw')
    )
    return gripper_pose

def get_jacobian(env):
    jacp = env.robots[0].sim.data.get_body_jacp('right_hand').reshape((3, -1))[:, env.robots[0]._ref_joint_vel_indexes]
    jacr = env.robots[0].sim.data.get_body_jacr('right_hand').reshape((3, -1))[:, env.robots[0]._ref_joint_vel_indexes]
    return np.concatenate([jacp, jacr], axis=0)

def inverse_kinematics(target_pose, env, max_error=1e-9):
    initial_joint_angles = env.robots[0]._joint_positions
    joint_angles = initial_joint_angles.copy()
    current_pose = get_gripper_eef_pose(env, joint_angles)
    pose_error = get_pose_error(pose2mat(target_pose), pose2mat(current_pose))
    error_mag = np.linalg.norm(pose_error)

    while error_mag > max_error:
        jacobian = get_jacobian(env)
        joint_angle_delta = np.matmul(np.linalg.pinv(jacobian), pose_error)
        joint_angles += joint_angle_delta
        current_pose = get_gripper_eef_pose(env, joint_angles)
        pose_error = get_pose_error(pose2mat(target_pose), pose2mat(current_pose))
        error_mag = np.linalg.norm(pose_error)

    return joint_angles

if __name__ == "__main__":
    # Original demo setup
    options = {}
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)
    options["env_name"] = choose_environment()

    if "TwoArm" in options["env_name"]:
        options["env_configuration"] = choose_multi_arm_config()
        if options["env_configuration"] == "bimanual":
            options["robots"] = "Baxter"
        else:
            options["robots"] = []
            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))
    else:
        options["robots"] = choose_robots(exclude_bimanual=True)

    joint_dim = 6 if options["robots"] == "UR5e" else 7
    controller_name = choose_controller()
    options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

    controller_settings = {
        "OSC_POSE": [6, 6, 0.1],
        "OSC_POSITION": [3, 3, 0.1],
        "IK_POSE": [6, 6, 0.01],
        "JOINT_POSITION": [joint_dim, joint_dim, 0.2],
        "JOINT_VELOCITY": [joint_dim, joint_dim, -0.1],
        "JOINT_TORQUE": [joint_dim, joint_dim, 0.25],
    }

    action_dim = controller_settings[controller_name][0]
    num_test_steps = controller_settings[controller_name][1]
    test_value = controller_settings[controller_name][2]
    steps_per_action = 75
    steps_per_rest = 75

    # Initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        horizon=(steps_per_action + steps_per_rest) * num_test_steps,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)
    n = 0
    gripper_dim = 0
    for robot in env.robots:
        gripper_dim = robot.gripper["right"].dof if isinstance(robot, Bimanual) else robot.gripper.dof
        n += int(robot.action_dim / (action_dim + gripper_dim))

    neutral = np.zeros(action_dim + gripper_dim)

    # Run through the predefined actions in the demo
    count = 0
    while count < num_test_steps:
        action = neutral.copy()
        for i in range(steps_per_action):
            if controller_name in {"IK_POSE", "OSC_POSE"} and count >= 3:
                vec = np.zeros(3)
                vec[count - 3] = test_value
                action[3:6] = vec
            else:
                action[count % action_dim] = test_value
            total_action = np.tile(action, n)
            env.step(total_action)
            env.render()
        for i in range(steps_per_rest):
            total_action = np.tile(neutral, n)
            env.step(total_action)
            env.render()
        count += 1

    # Perform peg-in-hole task using inverse kinematics after the demo actions
    target_poses = [
        (np.array([0.05, 0, 0.2]), np.array([1, 0, 0, 0])),  # Approach pose
        (np.array([0.05, 0, 0.15]), np.array([1, 0, 0, 0])),  # Alignment pose
        (np.array([0.05, 0, 0.1]), np.array([1, 0, 0, 0])),  # Insertion pose
    ]

    for pos, quat in target_poses:
        joint_angles = inverse_kinematics((pos, quat), env)
        env.robots[0].set_robot_joint_positions(joint_angles)
        env.step(joint_angles)
        env.render()

    env.close()