"""
Generate the random goals for the random goals experiment
"""

import os
import numpy as np
from rps.utilities.misc import generate_random_poses
from rps.robotarium import Robotarium

# The random seed
SEED = 57
# The minimum spacing between robots
MIN_SPACING = 0.33
# The number of goals per experiment
NUM_GOALS = 2
# The number of random goal experiments to generate
NUM_EXPERIMENTS = 10
# The spacing between the boundary of the robotarium and the farthest point to place a robot
BOUNDARY_SPACING: float = 0.375

np.random.seed(SEED)

def main():
    os.makedirs("random_goals", exist_ok=True)
    for i in range(NUM_EXPERIMENTS):
        goal_poses = np.zeros((NUM_GOALS, 3, 20))
        for j in range(NUM_GOALS):
            goal_poses[j] = generate_random_poses(
                20,
                spacing=MIN_SPACING,
                width=Robotarium.BOUNDARIES[1] * 2 - 2 * BOUNDARY_SPACING,
                height=Robotarium.BOUNDARIES[3] * 2 - 2 * BOUNDARY_SPACING
            )
        np.save(f"random_goals/goal_poses_{i}.npy", goal_poses)

if __name__ == "__main__":
    main()
