import math

def compute_vector(current_pose, goal_position, local_frame=True):
    """

    """
    # Unpack current pose and goal position
    x, y, theta = current_pose
    x_goal, y_goal = goal_position

    # Compute the global difference vector
    dx = x_goal - x
    dy = y_goal - y

    if not local_frame:
        # Use the global vector directly
        v_x, v_y = dx, dy
    else:
        # Transform the vector to the robot's local frame by rotating by -theta
        cos_theta = math.cos(-theta)
        sin_theta = math.sin(-theta)
        v_x = dx * cos_theta - dy * sin_theta
        v_y = dx * sin_theta + dy * cos_theta

    # norm = math.hypot(v_x, v_y)  # Computes sqrt(v_x**2 + v_y**2)
    # if norm != 0:
    #     v_x, v_y = v_x / norm, v_y / norm
    # else:
    #     # Return a zero vector if the computed vector is (0, 0)
    #     v_x, v_y = 0, 0

    return v_x, v_y

if __name__ == "__main__":
    # Example usage:
    current_pose = (0.0
                    , 0, 1)  # (x, y, theta in radians)
    goal_position = (1, 0)               # (x_goal, y_goal)

    # Compute the normalized 2D vector in the robot's local frame:
    local_normalized_vector = compute_vector(current_pose, goal_position, local_frame=True, )
    print("Normalized 2D vector in robot's local frame:", local_normalized_vector)

    # Compute the normalized 2D vector in the global coordinate frame:
    global_normalized_vector = compute_vector(current_pose, goal_position, local_frame=False, )
    print("Normalized 2D vector in global coordinates:", global_normalized_vector)
