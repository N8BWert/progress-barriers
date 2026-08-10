"""
Two groups of robots trying to pass in a narrowing hallway. 

Two groups of robots start on either side of the robotarium and try to pass each other in a
narrow hallway.
"""

import os
import time
import numpy as np
import argparse

from rps.robotarium import Robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.controllers import create_si_position_controller

from matplotlib.axes import Axes
import matplotlib.patches as patches

from barrier_certificates import (
    LinearClassKFunction,
    DecentralizedCircularBarrierNarrowHallway,
    CentralizedCircularBarrierNarrowHallway,
    DecentralizedProgressPriorityBarrierNarrowHallway
)

# Is this submitted to the Robotarium? If so, save our output file to the appropriate directory
ROBOTARIUM_SUBMISSION: bool = False
# After 30 seconds of no movement, we assume the robots are deadlocked and terminate the experiment.
DEADLOCK_TIMEOUT_SECONDS: float = 30.0
# If the robots have moved less this distance we increment the deadlock timer
DEADLOCK_TIMEOUT_EPSILON: float = 0.01
# If the experiment has not completed within this time limit, we assume a livelock has occurred
# and terminate the experiment
LIVELOCK_TIMEOUT_SECONDS: float = 5 * 60.0
# The radius (in meters) of the circle that the robots will be placed on
CIRCLE_RADIUS: float = 1.0
# Goal tolerance for the experiment (in meters)
GOAL_TOLERANCE: float = 0.03
# The maximum distance you can be pushed from the goal before you are no longer complete (in meters)
GOAL_EPSILON: float = 0.1
# The safety radius for the barrier certificates (in meters)
SAFETY_RADIUS: float = 0.16
# The maximum priority radius for the progress barrier certificate (in meters)
MAXIMUM_PRIORITY_RADIUS: float = 0.3
# The communication radius for the local-increasing barrier certificate (in meters)
COMMS_RADIUS: float = 0.5

# The left goal line X
LEFT_GOAL_X: float = -1.0
# The right goal line X
RIGHT_GOAL_X: float = 1.0


def parse_cli_args() -> tuple[str, bool, bool, bool, int]:
    parser = argparse.ArgumentParser(
        description="Run the Circle Swap Experiment for the Progress Barrier Project"
    )
    parser.add_argument(
        "-b",
        "--barrier",
        type=str,
        default="progress",
        choices=["progress", "decentralized", "centralized", "double", "increasing", "local-increasing"],
        help="The barrier certificate to use for collision avoidance.",
    )
    parser.add_argument(
        "--hide-figure",
        action=argparse.BooleanOptionalAction,
        help="Whether to hide the figure of the experiment.",
        default=False
    )
    parser.add_argument(
        "--run-in-real-time",
        action=argparse.BooleanOptionalAction,
        help="Whether to run the experiment in real time.",
        default=False
    )
    parser.add_argument(
        "--skip-initialization",
        action=argparse.BooleanOptionalAction,
        help="Whether to skip the initialization of the Robotarium.",
        default=False
    )
    parser.add_argument("num_robots", type=int, help="The number of robots to use in the experiment.")
    args = parser.parse_args()
    return args.barrier, bool(args.hide_figure), bool(args.run_in_real_time), bool(args.skip_initialization), int(args.num_robots)

def generate_walls() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Generate the left and right walls of the hallway as lists of points.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists of points representing the left and right walls of the hallway.
    """
    return ([
            np.array([-1.6, 0.9]),
            np.array([-1.3, 0.9]),
            np.array([-1.0, 0.5]),
            np.array([1.0, 0.5]),
            np.array([1.3, 0.9]),
            np.array([1.6, 0.9])
        ], [
            np.array([-1.6, -0.9]),
            np.array([-1.3, -0.9]),
            np.array([-1.0, -0.5]),
            np.array([1.0, -0.5]),
            np.array([1.3, -0.9]),
            np.array([1.6, -0.9])
        ])

def draw_lines(axes: Axes, line: list[np.ndarray]):
    """
    Draw a line on the given axes.

    Args:
        axes (Axes): The axes to draw the line on.
        line (list[np.ndarray]): A list of points representing the line.
    """
    for i in range(len(line) - 1):
        axes.plot(
            [line[i][0], line[i + 1][0]],
            [line[i][1], line[i + 1][1]],
            color="black",
            linewidth=2
        )

def generate_start_and_goal_poses(num_robots: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the start and goal poses for the robots.

    Args:
        num_robots (int): The number of robots.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the start and goal poses for the robots.
    """
    if num_robots % 2 != 0:
        raise ValueError("Number of robots must be even for the narrow hallway experiment.")
    if num_robots > 18:
        raise ValueError("Number of robots must be less than or equal to 18 for the narrow hallway experiment.")
    robots_per_team = num_robots // 2
    start_poses = np.zeros((3, num_robots))
    goal_poses = np.zeros((3, num_robots))

    left_start_positions = [
        np.array([-1.5, 0.0]),
        np.array([-1.5, 0.3]),
        np.array([-1.5, -0.3]),
        np.array([-1.5, 0.6]),
        np.array([-1.5, -0.6]),
        np.array([-1.2, 0.15]),
        np.array([-1.2, -0.15]),
        np.array([-1.2, 0.45]),
        np.array([-1.2, -0.45]),
    ]
    right_start_positions = [
        np.array([1.5, 0.0]),
        np.array([1.5, 0.3]),
        np.array([1.5, -0.3]),
        np.array([1.5, 0.6]),
        np.array([1.5, -0.6]),
        np.array([1.2, 0.15]),
        np.array([1.2, -0.15]),
        np.array([1.2, 0.45]),
        np.array([1.2, -0.45]),
    ]

    # The first team starts on the left side of the hallway and moves to the right
    for i in range(robots_per_team):
        start_poses[0:2, i] = left_start_positions[i]
        start_poses[2, i] = 0.0  # orientation
        goal_poses[0:2, i] = right_start_positions[i]
        goal_poses[2, i] = np.pi  # orientation

    # The second team starts on the right side of the hallway and moves to the left
    for i in range(robots_per_team, num_robots):
        start_poses[0:2, i] = right_start_positions[i - robots_per_team]
        start_poses[2, i] = np.pi  # orientation
        goal_poses[0:2, i] = left_start_positions[i - robots_per_team]
        goal_poses[2, i] = 0.0  # orientation

    return start_poses, goal_poses

def draw_goal_patches(axes: Axes):
    """
    Draw goal patches on the given axes.

    Args:
        axes (Axes): The axes to draw the goal patches on.
    """
    left_goal_patch = patches.Polygon(
        np.array([
            [-1.6, 0.9],
            [-1.3, 0.9],
            [-1.0, 0.5],
            [-1.0, -0.5],
            [-1.3, -0.9],
            [-1.6, -0.9]
        ]),
        color="blue",
        alpha=0.5
    )
    axes.add_patch(left_goal_patch)

    right_goal_patch = patches.Polygon(
        np.array([
            [1.6, 0.9],
            [1.3, 0.9],
            [1.0, 0.5],
            [1.0, -0.5],
            [1.3, -0.9],
            [1.6, -0.9]
        ]),
        color="green",
        alpha=0.5
    )
    axes.add_patch(right_goal_patch)

def main():
    barrier, hide_figure, run_in_real_time, skip_initialization, num_robots = parse_cli_args()
    print(f"Using {num_robots} robots with the {barrier} barrier.")

    initial_poses, goal_poses = generate_start_and_goal_poses(num_robots)
    walls = generate_walls()
    r = Robotarium(
        number_of_robots = num_robots,
        show_figure = not hide_figure,
        sim_in_real_time = run_in_real_time,
        initial_conditions = initial_poses,
        skip_initialization = skip_initialization
    )
    axes = r._axes_handle
    draw_goal_patches(axes)
    draw_lines(axes, walls[0])
    draw_lines(axes, walls[1])

    controller = create_si_position_controller(velocity_magnitude_limit=0.15)
    si_to_uni, uni_to_si = create_si_to_uni_mapping()
    match barrier:
        case "progress" | "increasing" | "local-increasing":
            barrier_certificates = [
                DecentralizedProgressPriorityBarrierNarrowHallway(
                    safety_radius = SAFETY_RADIUS,
                    maximum_priority_radius = MAXIMUM_PRIORITY_RADIUS,
                    priority_levels = num_robots + 1,
                    priority_level = i,
                    magnitude_limit = 0.15,
                    class_k_function = LinearClassKFunction(k=1.0),
                    left_points = walls[0],
                    right_points = walls[1],
                )
                for i in range(num_robots)
            ]
        case "decentralized":
            barrier_certificates = [
                DecentralizedCircularBarrierNarrowHallway(
                    safety_radius = SAFETY_RADIUS,
                    magnitude_limit = 0.15,
                    class_k_function = LinearClassKFunction(k=1.0),
                    left_points = walls[0],
                    right_points = walls[1]
                )
                for _ in range(num_robots)
            ]
        case "centralized":
            barrier_certificates = [
                CentralizedCircularBarrierNarrowHallway(
                    safety_radius = SAFETY_RADIUS,
                    magnitude_limit = 0.15,
                    class_k_function = LinearClassKFunction(k=1.0),
                    left_points = walls[0],
                    right_points = walls[1]
                )
            ]
        case "double":
            barrier_certificates = [
                DecentralizedProgressPriorityBarrierNarrowHallway(
                    safety_radius = SAFETY_RADIUS,
                    maximum_priority_radius = MAXIMUM_PRIORITY_RADIUS,
                    priority_levels = num_robots + 1,
                    priority_level = i,
                    magnitude_limit = 0.15,
                    class_k_function = LinearClassKFunction(k=1.0),
                    left_points = walls[0],
                    right_points = walls[1],
                )
                for i in range(num_robots)
            ]
            centralized_barrier = CentralizedCircularBarrierNarrowHallway(
                safety_radius = SAFETY_RADIUS,
                magnitude_limit = 0.15,
                class_k_function = LinearClassKFunction(k=1.0),
                left_points = walls[0],
                right_points = walls[1]
            )
        case _:
            raise ValueError(f"Unknown barrier certificate: {barrier}")
        
    robots_per_team = num_robots // 2

    min_h = np.inf
    x = r.get_poses()
    r.step()
    x_si = uni_to_si(x)

    deadlock_check_poses = x_si.copy()
    deadlock_timer = 0.0
    deadlocked = False
    livelocked = False
    complete_ids = set()
    elapsed_time = 0.0
    last_time = time.time()
    completion_times = np.full(num_robots, np.nan)
    while not np.all(x_si[0, :robots_per_team] >= RIGHT_GOAL_X) or not np.all(x_si[0, robots_per_team:] <= LEFT_GOAL_X):
        x = r.get_poses()
        x_si = uni_to_si(x)

        # Check what robots have completed their goals
        distances = np.linalg.norm(x_si - goal_poses[:2, :], axis=0)
        completed_ids = np.where(distances <= GOAL_TOLERANCE)[0]
        for id in completed_ids:
            if id not in complete_ids:
                if barrier == "increasing":
                    for barrier_certificate in barrier_certificates:
                        barrier_certificate.increase_priority_level()
                if barrier == "local-increasing":
                    for robot_id in range(num_robots):
                        if robot_id != id:
                            distance = np.linalg.norm(x_si[:, id] - x_si[:, robot_id])
                            if distance < COMMS_RADIUS:
                                barrier_certificates[robot_id].increase_priority_level()
                completion_times[id] = elapsed_time
        complete_ids |= set(completed_ids)
        if barrier == "progress" or barrier == "double" or barrier == "increasing" or barrier == "local-increasing":
            for i in complete_ids:
                barrier_certificates[i].set_priority_level(num_robots)

        # Check for deadlock
        if np.linalg.norm(deadlock_check_poses - x_si) < DEADLOCK_TIMEOUT_EPSILON:
            if run_in_real_time:
                deadlock_timer += time.time() - last_time
            else:
                deadlock_timer += 0.033
            if deadlock_timer >= DEADLOCK_TIMEOUT_SECONDS:
                deadlocked = True
                r.step()
                break
        else:
            deadlock_timer = 0.0
            deadlock_check_poses = x_si.copy()

        # Check for livelock
        if elapsed_time > LIVELOCK_TIMEOUT_SECONDS:
            livelocked = True
            r.step()
            break

        # Find the minimum h value across the barrier certificates
        for i in range(num_robots):
            for j in range(i+1, num_robots):
                h = barrier_certificates[0].h(x_si[:, i], x_si[:, j])
                min_h = min(min_h, h)

        dxi = controller(x_si, goal_poses[:2, :])
        # Apply barriers
        match barrier:
            case "decentralized" | "double" | "increasing" | "local-increasing":
                for i in range(num_robots):
                    dxi[:, i] = barrier_certificates[i](
                        x_si[:, i],
                        x_si[:, [j for j in range(num_robots) if j != i]],
                        dxi[:, i]
                    )
                    barrier_certificates[i].show(x_si[:, i], axes)
            case "centralized":
                dxi = barrier_certificates[0](x_si, dxi)
                barrier_certificates[0].show(x_si, axes)
            case "progress":
                for i in range(num_robots):
                    dxi[:, i] = barrier_certificates[i](
                        x_si[:, i],
                        x_si[:, [j for j in range(num_robots) if j != i]],
                        dxi[:, i]
                    )
                    if np.linalg.norm(dxi[:, i]) < 0.01 and i in complete_ids and np.linalg.norm(x_si[:, i] - goal_poses[:2, i]) > GOAL_EPSILON:
                        print("REDISTRIBUTING")
                        complete_ids.remove(i)
                        barrier_certificates[i].set_priority_level(i)
                        dxi[:, i] = barrier_certificates[i](
                            x_si[:, i],
                            x_si[:, [j for j in range(num_robots) if j != i]],
                            dxi[:, i]
                        )
                    barrier_certificates[i].show(x_si[:, i], axes)
        if barrier == "double":
            dxi = centralized_barrier.apply(x_si, dxi)
        dx = si_to_uni(dxi, x)
        r.set_velocities(np.arange(num_robots), dx)
        r.step()
        if run_in_real_time:
            elapsed_time += time.time() - last_time
        else:
            elapsed_time += 0.033
        last_time = time.time()

    if deadlocked:
        print(f"Deadlock detected after {elapsed_time:.2f} seconds.")
    elif livelocked:
        print(f"Livelock detected after {elapsed_time:.2f} seconds.")
    else:
        print(f"Experiment completed successfully in {elapsed_time:.2f} seconds.")
    print(f"Minimum h value across all barrier certificates: {min_h:.4f}")
    if ROBOTARIUM_SUBMISSION:
        np.save("completion_times.npy", completion_times)
    else:
        os.makedirs(f"narrow_hallway/{barrier}/{num_robots}", exist_ok=True)
        np.save(f"narrow_hallway/{barrier}/{num_robots}/completion_times.npy", completion_times)
    r.debug()

if __name__ == "__main__":
    main()
