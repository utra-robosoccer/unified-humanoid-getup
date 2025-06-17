#!/usr/bin/env python3
import subprocess
from functools import reduce

import numpy as np


def main():
    # List all the robot names you want to sweep over:
    # TODO make this better and easier to understnand
    multiplier1 = 7  # number of robots
    multiplier2 = 1  # number of seeds per row
    multiplier3 = 1  # number of rows
    multiplier = multiplier2 * multiplier3
    robot_names = [
        # s2
        [["bez1"]] * multiplier,
        [["op3_rot"]] * multiplier,
        [["bez2"]] * multiplier,
        [["bez3"]] * multiplier,
        [["sigmaban"]] * multiplier,
        [["wolfgang"]] * multiplier,
        [["nugus"]] * multiplier,
    ]
    robot_names = reduce(lambda x, y: x + y, robot_names)  # type: ignore[arg-type]
    ids = [9, 16, 23, 30] * 7
    ids = []
    for i in range(multiplier3):
        ids.append(np.linspace(63 + i, 72 + i, multiplier2, dtype=int))
    ids = np.array(ids * multiplier1).flatten()
    print(ids)
    print(robot_names)
    print(len(ids))
    print(len(robot_names))
    # ids = [49,56,50,57,51,58,52,59,53,60,54,61,55,62]
    #
    for idx, name in enumerate(robot_names):
        cmd = [
            "python3.10",
            "enjoy_sbx.py",
            "--algo",
            "crossq",
            "--env",
            "unified-humanoid-get-up-env-standup-v0",
            "--gym-packages",
            "unified_humanoid_get_up_env",
            "--folder",
            "logs/",
            "--load-best",
            "--exp-id",
            f"{ids[idx]}",
            "--no-render",
            "--env-kwargs",
            f"robot_name:{name}",
        ]
        print(f"\n=== Running training for robot_name='{name}' ===")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"✗ Training for '{name}' failed with exit code {e.returncode}")
            break
        else:
            print(f"✔ Finished training for '{name}'")


if __name__ == "__main__":
    main()
