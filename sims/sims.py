"""
module to simulate movement
"""

import time
import numpy as np
import pybullet as p
import pybullet_data

def setup_simulation():
    """Sets up the PyBullet simulation environment."""
    p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")
    return plane_id

def load_robot():
    """Loads the R2D2 robot in the simulation."""
    robot_start_pos = [0, 0, 1]
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF("r2d2.urdf", robot_start_pos, robot_start_orientation)
    return robot_id

def create_cubes():
    """Creates colored cubes in the simulation."""
    colors = ["red", "green", "blue", "yellow"]
    rgba_colors = {
        "red": [1, 0, 0, 1],
        "green": [0, 1, 0, 1],
        "blue": [0, 0, 1, 1],
        "yellow": [1, 1, 0, 1]
    }
    cube_size = 2
    half_cube_size = cube_size / 2
    distance = 4
    start_pos_y = -1.5 * distance
    positions = {
        "red": (2, start_pos_y, 0),
        "green": (2, start_pos_y + distance, 0),
        "blue": (2, start_pos_y + 2 * distance, 0),
        "yellow": (2, start_pos_y + 3 * distance, 0)
    }
    for color in colors:
        pos = positions[color]
        rgba = rgba_colors[color]
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              halfExtents=[half_cube_size] * 3,
                                              rgbaColor=rgba)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=pos)

def move_to_target(robot_id, target_pos, num_targ, threshold=2.05):
    """Moves the robot to a target position."""
    old_distance = np.inf
    direction = 1
    current_velocity = 40
    consecutive_count = 0

    while True:
        current_pos, _ = p.getBasePositionAndOrientation(robot_id)
        distance = np.linalg.norm(np.array(target_pos[:2]) - np.array(current_pos[:2]))
        print(f"Distance to target: {distance}")

        if distance <= threshold:
            print(f"Target {num_targ} Reached")
            halt_robot(robot_id, direction)
            break
        else:
            consecutive_count, direction = adjust_direction(distance, old_distance, consecutive_count, direction)
            set_robot_velocity(robot_id, direction, current_velocity)

        old_distance = distance
        p.stepSimulation()
        time.sleep(1./240.)

def halt_robot(robot_id, direction):
    """Halts the robot at its current position."""
    halt_velocity = direction * 8
    for joint in [2, 3, 6, 7]:
        p.setJointMotorControl2(robot_id, joint, p.VELOCITY_CONTROL, targetVelocity=halt_velocity)
    end_time = time.time() + 10
    while time.time() < end_time:
        p.stepSimulation()
        time.sleep(1./240.)

def adjust_direction(distance, old_distance, consecutive_count, direction):
    """Adjusts the robot's direction based on distance measurements."""
    if round(old_distance, 3) < round(distance, 3):
        consecutive_count += 1
    else:
        consecutive_count = 0

    if consecutive_count == 3:
        direction *= -1
        consecutive_count = 0

    return consecutive_count, direction

def set_robot_velocity(robot_id, direction, current_velocity):
    """Sets the velocity for the robot."""
    velocity = direction * current_velocity
    wheel_joints = [2, 3, 6, 7]
    for joint in wheel_joints:
        p.setJointMotorControl2(robot_id, joint, p.VELOCITY_CONTROL, targetVelocity=velocity)

def main():
    """Main function to run the simulation."""
    setup_simulation()
    robot_id = load_robot()
    create_cubes()
    positions = {
        "red": (2, -6, 0),
        "green": (2, -2, 0),
        "blue": (2, 2, 0),
        "yellow": (2, 6, 0)
    }
    targ = 1
    move_to_target(robot_id, positions["green"], targ)

    print("Arrived at final position")
    print("------" * 10)

    for _ in range(3000):
        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    main()
