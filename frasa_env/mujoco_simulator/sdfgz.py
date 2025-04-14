import numpy as np


def normalize_joint_positions(joint_positions: np.ndarray,
                                     lower_limits: np.ndarray,
                                     upper_limits: np.ndarray) -> np.ndarray:
    """
    Normalize joint positions to the range [-1, 1] such that:
      - A joint position of 0 maps to 0.
      - The upper limit maps to +1.
      - The lower limit maps to -1.

    This function is applied element-wise and uses separate scales for the positive
    and negative ranges.

    Parameters
    ----------
    joint_positions : np.ndarray
        Array of actual joint positions.
    lower_limits : np.ndarray
        Array of lower limits for each joint (assumed to be <= 0).
    upper_limits : np.ndarray
        Array of upper limits for each joint (assumed to be >= 0).

    Returns
    -------
    np.ndarray
        Normalized joint positions with 0 corresponding to 0, +1 to the upper limit,
        and -1 to the lower limit.
    """
    norm = np.zeros_like(joint_positions)

    # For joint positions >= 0, divide by the positive (upper) limit.
    pos_mask = joint_positions >= 0
    # To avoid division by zero if some upper_limit is 0:
    norm[pos_mask] = joint_positions[pos_mask] / np.where(upper_limits[pos_mask] == 0,
                                                          1.0,
                                                          upper_limits[pos_mask])

    # For joint positions < 0, divide by the absolute value of the lower limit.
    neg_mask = joint_positions < 0
    norm[neg_mask] = joint_positions[neg_mask] / np.where(lower_limits[neg_mask] == 0,
                                                          1.0,
                                                          -lower_limits[neg_mask])
    return norm


def scale_normalized_actions(normalized_actions: np.ndarray,
                                    lower_limits: np.ndarray,
                                    upper_limits: np.ndarray) -> np.ndarray:
    """
    Scale normalized actions in the range [-1, 1] back to actual joint positions.

    The inverse mapping satisfies:
      - A value of 0 maps to 0.
      - A value of +1 yields the joint's upper limit.
      - A value of -1 yields the joint's lower limit.

    Parameters
    ----------
    normalized_actions : np.ndarray
        Normalized actions (or joint values) in the range [-1, 1].
    lower_limits : np.ndarray
        Array of lower joint limits (assumed to be <= 0).
    upper_limits : np.ndarray
        Array of upper joint limits (assumed to be >= 0).

    Returns
    -------
    np.ndarray
        Joint positions corresponding to the normalized values.
    """
    joint_positions = np.zeros_like(normalized_actions)

    # For values >= 0, use the upper limits.
    pos_mask = normalized_actions >= 0
    joint_positions[pos_mask] = normalized_actions[pos_mask] * upper_limits[pos_mask]

    # For values < 0, use the lower limits (note: lower_limits are negative).
    neg_mask = normalized_actions < 0
    joint_positions[neg_mask] = normalized_actions[neg_mask] * (-lower_limits[neg_mask])

    return joint_positions


# Example usage:
# Suppose we have 3 joints, with the following limits:
lower = np.array([-1, -3.14159, -2.79253])
upper = np.array([3.0, 0, 0.261799])

# An observation from the simulator (e.g., joint angles):
observed_joints = np.array([0.0, 0.0, 0.0])  # mid-range values

# A policy outputs normalized actions in [-1, 1].
normalized_action = np.array([0, -0.5, -0.5])

# Normalize the observation:
norm_obs = normalize_joint_positions(observed_joints, lower, upper)
print("Normalized observation:", norm_obs)
# Expected output: [0. 0. 0.] because 0 is the mid-range.

# Scale the normalized action to get target joint positions:
joint_targets = scale_normalized_actions(normalized_action, lower, upper)
print("Target joint positions:", joint_targets)
# For the first joint:
# mid = ( -1.0 + 1.0) / 2 = 0, half_range = 1, so target = 0.5*1 + 0 = 0.5.
