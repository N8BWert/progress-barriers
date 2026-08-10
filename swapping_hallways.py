"""
Robots are split into 4 groups organized in a '+' sign.  The robots must then traverse
a narrow hallway to swap sides of the '+' sign while avoiding collisions
"""

import os
import time
import numpy as np
import argparse
from rps.robotarium import Robotarium
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import create_si_position_controller

from matplotlib.axes import Axes
import matplotlib.patches as patches

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

# The center points of the four quadrants of the '+' sign
CENTER_POINTS = [
    np.array([-0.2, -0.2]),
    np.array([-0.2, 0.2]),
    np.array([0.2, 0.2]),
    np.array([0.2, -0.2])
]
# The lines that define the left side boundaries
LEFT_LINES = (
    [
        np.array([-0.8, 0.7]),
        np.array([-0.6, 0.2]),
        np.array([-0.2, 0.2])
    ],
    [
        np.array([-0.8, -0.7]),
        np.array([-0.6, -0.2]),
        np.array([-0.2, -0.2])
    ]
)
# The lines that define the top side boundaries
TOP_LINES = (
    [
        np.array([0.7, 0.8]),
        np.array([0.2, 0.6]),
        np.array([0.2, 0.2])
    ],
    [
        np.array([-0.7, 0.8]),
        np.array([-0.2, 0.6]),
        np.array([-0.2, 0.2])
    ]
)
# The lines that define the right side boundaries
RIGHT_LINES = (
    [
        np.array([0.8, -0.7]),
        np.array([0.6, -0.2]),
        np.array([0.2, -0.2])
    ],
    [
        np.array([0.8, 0.7]),
        np.array([0.6, 0.2]),
        np.array([0.2, 0.2])
    ]
)
# The lines that define the bottom side boundaries
BOTTOM_LINES = (
    [
        np.array([-0.7, -0.8]),
        np.array([-0.2, -0.6]),
        np.array([-0.2, -0.2]),
    ],
    [
        np.array([0.7, -0.8]),
        np.array([0.2, -0.6]),
        np.array([0.2, -0.2])
    ]
)
# The line to pass on the left side to reach the goal
LEFT_GOAL_X: float = -0.4
# The line to pass on the right side to reach the goal
RIGHT_GOAL_X: float = 0.4
# The line to pass on the top side to reach the goal
TOP_GOAL_Y: float = 0.4
# The line to pass on the bottom side to reach the goal
BOTTOM_GOAL_Y: float = -0.4

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

def draw_walls(axes: Axes):
    """
    Draw the walls of the '+' sign hallway 
    """
    for i in range(len(LEFT_LINES[0]) - 1):
        axes.plot(
            [LEFT_LINES[0][i][0], LEFT_LINES[0][i+1][0]],
            [LEFT_LINES[0][i][1], LEFT_LINES[0][i+1][1]],
            color="black",
            linewidth=2
        )
        axes.plot(
            [LEFT_LINES[1][i][0], LEFT_LINES[1][i+1][0]],
            [LEFT_LINES[1][i][1], LEFT_LINES[1][i+1][1]],
            color="black",
            linewidth=2
        )
    for i in range(len(TOP_LINES[0]) - 1):
        axes.plot(
            [TOP_LINES[0][i][0], TOP_LINES[0][i+1][0]],
            [TOP_LINES[0][i][1], TOP_LINES[0][i+1][1]],
            color="black",
            linewidth=2
        )
        axes.plot(
            [TOP_LINES[1][i][0], TOP_LINES[1][i+1][0]],
            [TOP_LINES[1][i][1], TOP_LINES[1][i+1][1]],
            color="black",
            linewidth=2
        )
    for i in range(len(RIGHT_LINES[0]) - 1):
        axes.plot(
            [RIGHT_LINES[0][i][0], RIGHT_LINES[0][i+1][0]],
            [RIGHT_LINES[0][i][1], RIGHT_LINES[0][i+1][1]],
            color="black",
            linewidth=2
        )
        axes.plot(
            [RIGHT_LINES[1][i][0], RIGHT_LINES[1][i+1][0]],
            [RIGHT_LINES[1][i][1], RIGHT_LINES[1][i+1][1]],
            color="black",
            linewidth=2
        )
    for i in range(len(BOTTOM_LINES[0]) - 1):
        axes.plot(
            [BOTTOM_LINES[0][i][0], BOTTOM_LINES[0][i+1][0]],
            [BOTTOM_LINES[0][i][1], BOTTOM_LINES[0][i+1][1]],
            color="black",
            linewidth=2
        )
        axes.plot(
            [BOTTOM_LINES[1][i][0], BOTTOM_LINES[1][i+1][0]],
            [BOTTOM_LINES[1][i][1], BOTTOM_LINES[1][i+1][1]],
            color="black",
            linewidth=2
        )
    for i in range(2):
        axes.plot(
            [LEFT_LINES[i][0][0], LEFT_LINES[i][0][0]-0.2],
            [LEFT_LINES[i][0][1], LEFT_LINES[i][0][1]],
            color="black",
            linewidth=2
        )
        axes.plot(
            [RIGHT_LINES[i][0][0], RIGHT_LINES[i][0][0]+0.2],
            [RIGHT_LINES[i][0][1], RIGHT_LINES[i][0][1]],
            color="black",
            linewidth=2
        )
        axes.plot(
            [TOP_LINES[i][0][0], TOP_LINES[i][0][0]],
            [TOP_LINES[i][0][1], TOP_LINES[i][0][1]+0.2],
            color="black",
            linewidth=2
        )
        axes.plot(
            [BOTTOM_LINES[i][0][0], BOTTOM_LINES[i][0][0]],
            [BOTTOM_LINES[i][0][1], BOTTOM_LINES[i][0][1]-0.2],
            color="black",
            linewidth=2
        )

def generate_start_and_goal_poses(num_robots: int) -> tuple[np.ndarray, np.ndarray]:
    # TODO:
    pass

def draw_goal_patches(axes: Axes):
    """
    Draw the goal patches for the experiment
    """
    # left goal patch
    axes.add_patch(patches.Polygon(
        np.array([
            [LEFT_LINES[0][0][0]-0.2, LEFT_LINES[0][0][1]],
            [LEFT_LINES[0][0][0], LEFT_LINES[0][0][1]],
            [LEFT_LINES[0][1][0], LEFT_LINES[0][1][1]],
            [LEFT_LINES[1][1][0], LEFT_LINES[1][1][1]],
            [LEFT_LINES[1][0][0], LEFT_LINES[1][0][1]],
            [LEFT_LINES[1][0][0]-0.2, LEFT_LINES[1][0][1]]
        ]),
        color="blue",
        alpha=0.5
    ))
    # top goal patch
    axes.add_patch(patches.Polygon(
        np.array([
            [TOP_LINES[0][0][0], TOP_LINES[0][0][1]+0.2],
            [TOP_LINES[0][0][0], TOP_LINES[0][0][1]],
            [TOP_LINES[0][1][0], TOP_LINES[0][1][1]],
            [TOP_LINES[1][1][0], TOP_LINES[1][1][1]],
            [TOP_LINES[1][0][0], TOP_LINES[1][0][1]],
            [TOP_LINES[1][0][0], TOP_LINES[1][0][1]+0.2]
        ]),
        color="green",
        alpha=0.5
    ))
    # right goal patch
    axes.add_patch(patches.Polygon(
        np.array([
            [RIGHT_LINES[0][0][0]+0.2, RIGHT_LINES[0][0][1]],
            [RIGHT_LINES[0][0][0], RIGHT_LINES[0][0][1]],
            [RIGHT_LINES[0][1][0], RIGHT_LINES[0][1][1]],
            [RIGHT_LINES[1][1][0], RIGHT_LINES[1][1][1]],
            [RIGHT_LINES[1][0][0], RIGHT_LINES[1][0][1]],
            [RIGHT_LINES[1][0][0]+0.2, RIGHT_LINES[1][0][1]]
        ]),
        color="red",
        alpha=0.5
    ))
    # bottom goal patch
    axes.add_patch(patches.Polygon(
        np.array([
            [BOTTOM_LINES[0][0][0], BOTTOM_LINES[0][0][1]-0.2],
            [BOTTOM_LINES[0][0][0], BOTTOM_LINES[0][0][1]],
            [BOTTOM_LINES[0][1][0], BOTTOM_LINES[0][1][1]],
            [BOTTOM_LINES[1][1][0], BOTTOM_LINES[1][1][1]],
            [BOTTOM_LINES[1][0][0], BOTTOM_LINES[1][0][1]],
            [BOTTOM_LINES[1][0][0], BOTTOM_LINES[1][0][1]-0.2]
        ]),
        color="yellow",
        alpha=0.5
    ))

def main():
    barrier, hide_figure, run_in_real_time, skip_initialization, num_robots = parse_cli_args()
    print(f"Using {num_robots} robots with the {barrier} barrier.")

    r = Robotarium(
        number_of_robots = num_robots,
        show_figure = not hide_figure,
        sim_in_real_time = run_in_real_time,
        skip_initialization = skip_initialization,
        show_arena_boundaries = False
    )
    axes = r._axes_handle
    draw_walls(axes)
    draw_goal_patches(axes)

    controller = create_si_position_controller(velocity_magnitude_limit=0.15)
    si_to_uni, uni_to_si = create_si_to_uni_mapping()
    # TODO: Create barrier    

    min_h = np.inf
    x = r.get_poses()
    r.step()
    x_si = uni_to_si(x)

    input()

    r.debug()

if __name__ == "__main__":
    main()
