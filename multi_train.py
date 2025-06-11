#!/usr/bin/env python3
import subprocess

def main():
    # List all the robot names you want to sweep over:
    robot_names = [
        # s2

        # ['sig'],
        # ['bitbot'],
        # ['nugus'],
        #
        # ['bez1'],
        # ['op3'],
        # ['bez'],
        # ['bez3'],
        #
        # ['sig'],
        # ['bitbot'],
        # ['nugus'],
        # all 7
        # [ 'bez1','op3','bez','bez3', 'sig', 'bitbot', 'nugus'],
        # s1
        # ['op3', 'bez', 'bez3', 'sig', 'bitbot', 'nugus'],
        # ['bez1', 'bez', 'bez3', 'sig', 'bitbot', 'nugus'],
        # ['bez1', 'op3', 'bez3', 'sig', 'bitbot', 'nugus'],
        # ['bez1', 'op3', 'bez', 'sig', 'bitbot', 'nugus'],
        # ['bez1', 'op3', 'bez', 'bez3', 'bitbot', 'nugus'],
        # ['bez1', 'op3', 'bez', 'bez3', 'sig', 'nugus'],
        # ['bez1', 'op3', 'bez', 'bez3', 'sig', 'bitbot'],

        # # s3
        # ['bez3', 'op3',  'bitbot',],
        # [ 'bez', 'bez3','op3', 'bitbot', 'nugus'],
        # ['bez1',  'bez3', 'nugus'],
        # s4
        # ['bez', 'sig'],
        # ['op3', 'bez', 'sig'],
        ['bez1', 'op3', 'bez', 'sig'],
        # ['bez1', 'op3', 'bez', 'sig', 'nugus'],
    ]

    for i in range(7):
        for name in robot_names:
            cmd = [
                "python", "train_sbx.py",
                "--algo", "crossq",
                "--env", "frasa-standup-v0",
                "--conf", "hyperparams/crossq.yml",
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
