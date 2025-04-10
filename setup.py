from setuptools import find_packages, setup

setup(
    name="frasa_env",
    packages=[package for package in find_packages() if package.startswith("frasa_env")],
    version="1.0",
    description="FRASA RL Environment",
    install_requires=[
        "gymnasium>=0.29.1,<1.1.0",
        "numpy>=1.20.0",
        "stable_baselines3>=2.1.0",
        "sb3-contrib>=2.1.0",
        "mujoco>=3.1.5",
        "meshcat>=0.3.2",
        "sbx-rl>=0.17.0",
    ],
)
