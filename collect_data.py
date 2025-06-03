#!/usr/bin/env python3
import subprocess
from functools import reduce
def main():
    # List all the robot names you want to sweep over:
    multiplier = 7
    robot_names = [
        # s2
        [['bez1']]*multiplier,
        [['op3']]*multiplier,
        [['bez']]*multiplier,
        [['bez3']]*multiplier,
        [['sig']]*multiplier,
        [['bitbot']]*multiplier,
        [['nugus']]*multiplier,
    ]
    robot_names = reduce(lambda x,y :x+y ,robot_names)
    # ids = [13,20,27,34]*7
    ids = [35,36,37,38,39,40,41] * 7

    for idx, name in enumerate(robot_names):
        cmd = [
            "python3.10", "enjoy_sbx.py",
            "--algo", "crossq",
            "--env", "frasa-standup-v0",
            "--gym-packages", "frasa_env",
            "--folder", "logs/",
            "--load-best",
            "--exp-id",f"{ids[idx]}",
            "--env-kwargs", f"robot_name:{name}"
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
