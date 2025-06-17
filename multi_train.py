#!/usr/bin/env python3
import subprocess


def main():
    # List all the robot names you want to sweep over:
    robot_names = [
        # s2
        # ['sigmaban'],
        # ['wolfgang'],
        # ['nugus'],
        #
        # ['bez1'],
        # ['op3_rot'],
        # ['bez2'],
        # ['bez3'],
        #
        # ['sigmaban'],
        # ['wolfgang'],
        # ['nugus'],
        # all 7
        # [ 'bez1','op3_rot','bez2','bez3', 'sigmaban', 'wolfgang', 'nugus'],
        # s1
        # ['op3_rot', 'bez2', 'bez3', 'sigmaban', 'wolfgang', 'nugus'],
        # ['bez1', 'bez2', 'bez3', 'sigmaban', 'wolfgang', 'nugus'],
        # ['bez1', 'op3_rot', 'bez3', 'sigmaban', 'wolfgang', 'nugus'],
        # ['bez1', 'op3_rot', 'bez2', 'sigmaban', 'wolfgang', 'nugus'],
        # ['bez1', 'op3_rot', 'bez2', 'bez3', 'wolfgang', 'nugus'],
        # ['bez1', 'op3_rot', 'bez2', 'bez3', 'sigmaban', 'nugus'],
        # ['bez1', 'op3_rot', 'bez2', 'bez3', 'sigmaban', 'wolfgang'],
        # # s3
        # ['bez3', 'op3_rot',  'wolfgang',],
        # [ 'bez2', 'bez3','op3_rot', 'wolfgang', 'nugus'],
        # ['bez1',  'bez3', 'nugus'],
        # s4
        # ['bez2', 'sigmaban'],
        # ['op3_rot', 'bez2', 'sigmaban'],
        ["bez1", "op3_rot", "bez2", "sigmaban"],
        # ['bez1', 'op3_rot', 'bez2', 'sigmaban', 'nugus'],
    ]

    for i in range(7):
        for name in robot_names:
            cmd = [
                "python",
                "train_sbx.py",
                "--algo",
                "crossq",
                "--env",
                "unified-humanoid-get-up-env-standup-v0",
                "--conf",
                "hyperparams/crossq.yml",
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
