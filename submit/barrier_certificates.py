"""
Barrier Certificates for use in the progress barriers project
"""

import math
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod
from cvxopt import matrix, sparse
from cvxopt.solvers import qp, options
from matplotlib.axes import Axes
import matplotlib.patches as patches

options["show_progress"] = False
options["reltol"] = 1e-6
options["feastol"] = 1e-6
options["maxiters"] = 100


class ClassKFunction(ABC):
    """
    Abstract base class for class kappa functions. A class kappa function is a continuous, strictly increasing
    function that maps from the non-negative reals to the non-negative reals and satisfies
    the property that f(0) = 0.
    """

    @abstractmethod
    def evaluate(self, x: float) -> float:
        """
        Evaluate the class kappa function at a given point x.

        Args:
            x (float): The input value at which to evaluate the function.
        Returns:
            float: The value of the class kappa function at x. 
        """
        ...

    def __call__(self, x: float) -> float:
        return self.evaluate(x)

class LinearClassKFunction(ClassKFunction):
    """
    A linear class kappa function of the form f(x) = k * x, where k is a positive constant.
    """

    def __init__(self, k: float):
        if k <= 0:
            raise ValueError("k must be a positive constant.")
        self.k = k

    def evaluate(self, x: float) -> float:
        return self.k * x

class CubicClassKFunction(ClassKFunction):
    """
    A cubic class kappa function of the form f(x) = k * x^3, where k is a positive constant.
    """

    def __init__(self, k: float):
        if k <= 0:
            raise ValueError("k must be a positive constant.")
        self.k = k

    def evaluate(self, x: float) -> float:
        return self.k * (x ** 3)


class BarrierCertificate(ABC):
    """
    Abstract base class for barrier certificates. A barrier certificate is a function that is used to ensure safety
    in control systems by providing a measure of how close the system is to violating safety constraints.
    """

    @abstractmethod
    def h(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> float:
        """
        Evaluate the barrier function h at the given states x1 and x2.

        Args:
            x1 (np.ndarray): The state of the first agent.
            x2 (np.ndarray): The state of the second agent.
        Returns:
            float: The value of the barrier function h at the given states.
        """
        ...

    @abstractmethod
    def grad_h(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the gradient of the barrier function h with respect to the states x1 and x2.

        Args:
            x1 (np.ndarray): The state of the first agent.
            x2 (np.ndarray): The state of the second agent.
        Returns:
            np.ndarray: The gradient of the barrier function h with respect to the states.
        """
        ...

    def _solve_qp(
        self,
        vhat: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Solve the quadratic program to find the optimal control input that satisfies the barrier constraints.

        Args:
            vhat (np.ndarray): The nominal control input. (2,N)
            A (np.ndarray): The matrix representing the linear constraints. (num_constraints, 2*N)
            b (np.ndarray): The vector representing the linear constraints. (num_constraints,)
        Returns:
            np.ndarray: The optimal control input that satisfies the barrier constraints.
        """
        N = vhat.shape[1]
        H = sparse(matrix(2.0 * np.eye(2 * N)))
        f = matrix(-2.0 * vhat.reshape(-1, order="F"))

        try:
            sol = qp(H, f, matrix(A), matrix(b))
            if sol["status"] == "optimal":
                return np.reshape(sol["x"], (2, N), order="F")
        except Exception:
            pass

        return None


class DecentralizedCircularBarrier(BarrierCertificate):
    """
    A decentralized circular barrier certificate for ensuring safety in multi-agent systems. This barrier certificate
    is based on the concept of control barrier functions and is designed to maintain a safe distance between agents.
    """

    def __init__(
        self,
        safety_radius: float = 0.15,
        magnitude_limit: float = 0.2,
        class_k_function: ClassKFunction = LinearClassKFunction(1.0)
    ):
        self.safety_radius = safety_radius
        self.magnitude_limit = magnitude_limit
        self.class_k_function = class_k_function
        self.patch = None

    def h(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> float:
        diff = x1[:2] - x2[:2]
        return np.dot(diff, diff) - self.safety_radius ** 2

    def grad_h(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> np.ndarray:
        diff = x1[:2] - x2[:2]
        return 2 * diff

    def show(
        self,
        pose: np.ndarray,
        axes: Axes,
    ):
        """
        Draw the zero level set of the barrier function as a circle around the agent's position.

        Args:
            pose (np.ndarray): The current pose of the agent. (3,)
            axes (Axes): The matplotlib axes on which to draw the circle.
        """
        if self.patch is None:
            self.patch = patches.Circle(
                (pose[0], pose[1]),
                radius=self.safety_radius,
                edgecolor="black",
                facecolor="red",
                alpha=0.3,
                zorder=3,
            )
            axes.add_patch(self.patch)
        self.patch.center = (pose[0], pose[1])

    def apply(
        self,
        pose: np.ndarray,
        obstacle_poses: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the decentralized circular barrier certificate to ensure safety in multi-agent systems.

        Args:
            pose (np.ndarray): The current pose of the agent. (3,)
            obstacle_poses (np.ndarray): The poses of the obstacles. (3, N)
            velocity (np.ndarray): The nominal control input for the agent. (2,)
        Returns:
            np.ndarray: The modified control input that satisfies the barrier constraints. (2,) 
        """
        num_constraints = obstacle_poses.shape[1] + 8
        A = np.zeros((num_constraints, 2))
        b = np.zeros(num_constraints)

        # Apply obstacle constraints
        for i in range(obstacle_poses.shape[1]):
            h = self.h(pose, obstacle_poses[:, i])
            grad_h = self.grad_h(pose, obstacle_poses[:, i])
            A[i, :] = -grad_h
            b[i] = self.class_k_function(h)

        # Apply Magnitude Constraints (8-sided approximation of the l2-norm)
        limit = self.magnitude_limit * np.cos(np.pi / 8)

        # vx <= magnitude_limit
        A[obstacle_poses.shape[1], 0] = 1.0
        b[obstacle_poses.shape[1]] = limit

        # 1/sqrt(2) * (vx + vy) <= magnitude_limit
        A[obstacle_poses.shape[1] + 1, :] = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
        b[obstacle_poses.shape[1] + 1] = limit

        # vy <= magnitude_limit
        A[obstacle_poses.shape[1] + 2, 1] = 1.0
        b[obstacle_poses.shape[1] + 2] = limit

        # 1/sqrt(2) * (-vx + vy) <= magnitude_limit
        A[obstacle_poses.shape[1] + 3, :] = [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
        b[obstacle_poses.shape[1] + 3] = limit

        # -vx <= magnitude_limit
        A[obstacle_poses.shape[1] + 4, 0] = -1.0
        b[obstacle_poses.shape[1] + 4] = limit

        # 1/sqrt(2) * (-vx - vy) <= magnitude_limit
        A[obstacle_poses.shape[1] + 5, :] = [-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]
        b[obstacle_poses.shape[1] + 5] = limit

        # -vy <= magnitude_limit
        A[obstacle_poses.shape[1] + 6, 1] = -1.0
        b[obstacle_poses.shape[1] + 6] = limit

        # 1/sqrt(2) * (vx - vy) <= magnitude_limit
        A[obstacle_poses.shape[1] + 7, :] = [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]
        b[obstacle_poses.shape[1] + 7] = limit

        safe_velocities = self._solve_qp(velocity.reshape(2, 1), A, b)
        if safe_velocities is None:
            return np.zeros(2)
        return safe_velocities.flatten()

    def __call__(
        self,
        pose: np.ndarray,
        obstacle_poses: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        return self.apply(pose, obstacle_poses, velocity)


class DecentralizedProgressPriorityBarrier(BarrierCertificate):
    """
    A decentralized barrier certificate using a priority-based outer progress barrier
    to ensure safety and progress in multi-agent systems. 
    """

    def __init__(
        self,
        safety_radius: float = 0.15,
        maximum_priority_radius: float = 0.25,
        priority_levels: int = 10,
        priority_level: int = 0,
        magnitude_limit: float = 0.2,
        class_k_function: ClassKFunction = LinearClassKFunction(1.0)
    ):
        """
        Create a new Decentralized Progress Priority barrier

        Args:
            safety_radius (float): The minimum distance to maintain between agents for safety.
            maximum_priority_radius (float): The radius of the largest barrier (i.e. the barrier with the
                least priority)
            priority_levels (int): The number of discrete priority levels for the progress barrier.
            priority_level (int): The specific priority level for this barrier instance.
            magnitude_limit (float): The maximum allowed magnitude of the control input.
            class_k_function (ClassKFunction): A class kappa function used in the barrier constraints. 
        """
        self.safety_radius = safety_radius
        self.maximum_priority_radius = maximum_priority_radius
        self.magnitude_limit = magnitude_limit
        self.class_k_function = class_k_function
        self.progress_radius = safety_radius + (maximum_priority_radius - safety_radius) / (priority_levels-1) * priority_level
        self.priority_levels = priority_levels        
        self.priority_level = priority_level
        print(f"Priority: {self.priority_level}, Progress Radius: {self.progress_radius}")
        self.safety_patch = None
        self.progress_patch = None

    def h(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    )-> float:
        diff = x1[:2] - x2[:2]
        return np.dot(diff, diff) - self.safety_radius ** 2

    def progress_h(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> float:
        """
        Calculate the h-value of the outer progress barrier

        Args:
            x1 (np.ndarray): The state of the first agent.
            x2 (np.ndarray): The state of the second agent.
        Returns:
            float: The value of the progress barrier function h at the given states.
        """
        diff = x1[:2] - x2[:2]
        return np.dot(diff, diff) - self.progress_radius ** 2

    def grad_h(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> np.ndarray:
        diff = x1[:2] - x2[:2]
        return 2 * diff

    def progress_grad_h(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the gradient of the h-value of the outer progress barrier

        Args:
            x1 (np.ndarray): The state of the first agent.
            x2 (np.ndarray): The state of the second agent.
        Returns:
            np.ndarray: The gradient of the progress barrier function h with respect to the states.
        """
        diff = x1[:2] - x2[:2]
        return 2 * diff

    def show(
        self,
        pose: np.ndarray,
        axes: Axes,
    ):
        """
        Draw the zero level set of the progress and safety barrier functions
        as circles around the agent's position. 
        """
        if self.progress_patch is None:
            self.progress_patch = patches.Circle(
                (pose[0], pose[1]),
                radius=self.progress_radius,
                edgecolor="black",
                facecolor="blue",
                alpha=0.3,
                zorder=3,
            )
            axes.add_patch(self.progress_patch)
        self.progress_patch.set_radius(self.progress_radius)
        self.progress_patch.center = (pose[0], pose[1])

        if self.safety_patch is None:
            self.safety_patch = patches.Circle(
                (pose[0], pose[1]),
                radius=self.safety_radius,
                edgecolor="black",
                facecolor="red",
                alpha=0.3,
                zorder=3,
            )
            axes.add_patch(self.safety_patch)
        self.safety_patch.center = (pose[0], pose[1])

    def set_priority_level(self, priority_level: int):
        """
        Set the priority level for the progress barrier. This will update the progress radius accordingly.

        Args:
            priority_level (int): The new priority level to set. Must be between 0 and the number of priority levels - 1.
        """
        if not (0 <= priority_level < self.priority_levels):
            raise ValueError(f"Priority level must be between 0 and {self.priority_levels - 1}")
        self.progress_radius = self.safety_radius + (self.maximum_priority_radius - self.safety_radius) / (self.priority_levels - 1) * priority_level
        self.priority_level = priority_level

    def increase_priority_level(self):
        """
        Increase the priority level for the progress barrier by one, if possible. This will update the progress radius accordingly.
        """
        if self.priority_level > 0 and self.priority_level < self.priority_levels - 1:
            self.priority_level -= 1
            self.progress_radius = self.safety_radius + (self.maximum_priority_radius - self.safety_radius) / (self.priority_levels - 1) * self.priority_level

    def apply(
        self,
        pose: np.ndarray,
        obstacle_poses: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the decentralized progress priority barrier certificate to ensure safety and progress in multi-agent systems.

        Args:
            pose (np.ndarray): The current pose of the agent. (3,)
            obstacle_poses (np.ndarray): The poses of the obstacles. (3, N)
            velocity (np.ndarray): The nominal control input for the agent. (2,)
        Returns:
            np.ndarray: The modified control input that satisfies the barrier constraints. (2,) 
        """
        num_constraints = 2 * obstacle_poses.shape[1] + 8
        A = np.zeros((num_constraints, 2))
        b = np.zeros(num_constraints)

        # Apply safety constraints
        for i in range(obstacle_poses.shape[1]):
            h = self.h(pose, obstacle_poses[:, i])
            grad_h = self.grad_h(pose, obstacle_poses[:, i])
            A[i, :] = -grad_h
            b[i] = self.class_k_function(h)

        # Apply progress constraints
        for i in range(obstacle_poses.shape[1]):
            h_progress = self.progress_h(pose, obstacle_poses[:, i])
            grad_h_progress = self.progress_grad_h(pose, obstacle_poses[:, i])
            A[obstacle_poses.shape[1] + i, :] = -grad_h_progress
            b[obstacle_poses.shape[1] + i] = self.class_k_function(h_progress)

        # Apply Magnitude Constraints (8-sided approximation of the l2-norm)
        limit = self.magnitude_limit * np.cos(np.pi / 8)

        # vx <= magnitude_limit
        A[2 * obstacle_poses.shape[1], 0] = 1.0
        b[2 * obstacle_poses.shape[1]] = limit

        # 1/sqrt(2) * (vx + vy) <= magnitude_limit
        A[2 * obstacle_poses.shape[1] + 1, :] = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
        b[2 * obstacle_poses.shape[1] + 1] = limit

        # vy <= magnitude_limit
        A[2 * obstacle_poses.shape[1] + 2, 1] = 1.0
        b[2 * obstacle_poses.shape[1] + 2] = limit

        # 1/sqrt(2) * (-vx + vy) <= magnitude_limit
        A[2 * obstacle_poses.shape[1] + 3, :] = [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
        b[2 * obstacle_poses.shape[1] + 3] = limit

        # -vx <= magnitude_limit
        A[2 * obstacle_poses.shape[1] + 4, 0] = -1.0
        b[2 * obstacle_poses.shape[1] + 4] = limit

        # 1/sqrt(2) * (-vx - vy) <= magnitude_limit
        A[2 * obstacle_poses.shape[1] + 5, :] = [-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]
        b[2 * obstacle_poses.shape[1] + 5] = limit

        # -vy <= magnitude_limit
        A[2 * obstacle_poses.shape[1] + 6, 1] = -1.0
        b[2 * obstacle_poses.shape[1] + 6] = limit

        # 1/sqrt(2) * (vx - vy) <= magnitude_limit
        A[2 * obstacle_poses.shape[1] + 7, :] = [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]
        b[2 * obstacle_poses.shape[1] + 7] = limit

        safe_velocities = self._solve_qp(velocity.reshape(2, 1), A, b)
        if safe_velocities is None:
            return np.zeros(2)
        return safe_velocities.flatten()

    def __call__(
        self,
        pose: np.ndarray,
        obstacle_poses: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        return self.apply(pose, obstacle_poses, velocity)


class CentralizedCircularBarrier(BarrierCertificate):
    """
    A centralized circular barrier certificate for ensuring safety in multi-agent systems. This barrier certificate
    is based on the concept of control barrier functions and is designed to maintain a safe distance between agents.
    """

    def __init__(
        self,
        safety_radius: float = 0.15,
        magnitude_limit: float = 0.2,
        class_k_function: ClassKFunction = LinearClassKFunction(1.0)
    ):
        self.safety_radius = safety_radius
        self.magnitude_limit = magnitude_limit
        self.class_k_function = class_k_function
        self.patches: list[patches.Circle] = []

    def h(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> float:
        diff = x1[:2] - x2[:2]
        return np.dot(diff, diff) - self.safety_radius ** 2

    def grad_h(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> np.ndarray:
        diff = x1[:2] - x2[:2]
        return 2 * diff

    def show(
        self,
        poses: np.ndarray,
        axes: Axes,
    ):
        """
        Draw the zero level set of the barrier function as circles around each agent's position.

        Args:
            poses (np.ndarray): The current poses of the agents. (3, N)
            axes (Axes): The matplotlib axes on which to draw the circles.
        """
        while len(self.patches) < poses.shape[1]:
            patch = patches.Circle(
                (0, 0),
                radius=self.safety_radius,
                edgecolor="black",
                facecolor="red",
                alpha=0.3,
                zorder=3,
            )
            axes.add_patch(patch)
            self.patches.append(patch)

        for i in range(poses.shape[1]):
            self.patches[i].center = (poses[0, i], poses[1, i])

    def apply(
        self,
        poses: np.array,
        velocities: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the centralized circular barrier certificate to ensure safety in multi-agent systems.

        Args:
            poses (np.ndarray): The current poses of the agents. (3, N)
            velocities (np.ndarray): The nominal control inputs for the agents. (2, N)
        Returns:
            np.ndarray: The modified control inputs that satisfy the barrier constraints. (2, N)
        """
        N = velocities.shape[1]
        num_constraints = math.comb(N, 2) + 8 * N
        A = np.zeros((num_constraints, 2 * N))
        b = np.zeros(num_constraints)

        # Apply Robot Constraints
        constraint = 0
        for i in range(N-1):
            for j in range(i+1, N):
                h = self.h(poses[:, i], poses[:, j])
                grad_h = self.grad_h(poses[:, i], poses[:, j])
                A[constraint, 2*i:2*i+2] = -grad_h
                A[constraint, 2*j:2*j+2] = grad_h
                b[constraint] = self.class_k_function(h)
                constraint += 1

        # Apply magnitude constraints
        for i in range(N):
            limit = self.magnitude_limit * np.cos(np.pi / 8)

            # vx <= magnitude_limit
            A[constraint, 2*i] = 1.0
            b[constraint] = limit
            constraint += 1

            # 1/sqrt(2) * (vx + vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
            b[constraint] = limit
            constraint += 1

            # vy <= magnitude_limit
            A[constraint, 2*i+1] = 1.0
            b[constraint] = limit
            constraint += 1

            # 1/sqrt(2) * (-vx + vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
            b[constraint] = limit
            constraint += 1

            # -vx <= magnitude_limit
            A[constraint, 2*i] = -1.0
            b[constraint] = limit
            constraint += 1

            # 1/sqrt(2) * (-vx - vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]
            b[constraint] = limit
            constraint += 1

            # -vy <= magnitude_limit
            A[constraint, 2*i+1] = -1.0
            b[constraint] = limit
            constraint += 1

            # 1/sqrt(2) * (vx - vy) <= magnitude_limit
            A[constraint, 2*i:2*i+2] = [1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]
            b[constraint] = limit
            constraint += 1

        safe_velocities = self._solve_qp(velocities, A, b)
        if safe_velocities is None:
            return np.zeros((2, N))
        return safe_velocities

    def __call__(
        self,
        poses: np.ndarray,
        velocities: np.ndarray,
    ) -> np.ndarray:
        return self.apply(poses, velocities)
