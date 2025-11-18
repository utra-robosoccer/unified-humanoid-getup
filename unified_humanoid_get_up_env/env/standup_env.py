import math
import os
import pickle
import random
import warnings
from typing import Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from unified_humanoid_get_up_env.mujoco_simulator.simulator import Simulator, tf


class StandupEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(
        self,
        robot_name: list = ["uw"],
        render_mode="none",
        options: Optional[dict] = None,
        evaluation: bool = False,
        **kwargs,
    ):
        self.options = {
            # Duration of the stabilization pre-simulation (waiting for the gravity to stabilize the robot) [s]
            "stabilization_time": 2.0,
            # Duration of the episode before truncation [s]
            "truncate_duration": 5.0,
            # Period for the agent to apply control [s]
            "dt": 0.05,
            # Maximum command angular velocity [rad/s]
            "vmax": 2 * np.pi,
            # Is the render done in "realtime"
            "render_realtime": True,
            # Target robot state (q_motors, tilt) [rad^6]
            # [elbow, shoulder_pitch, hip_pitch, knee, ankle_pitch, IMU_pitch]
            # "desired_state": np.deg2rad([-49.5, 19.5, -52, 79, -36.5, -8.5]),
            # "desired_state": np.deg2rad([49.5, 19.5, 52, -82.15, 36.5, 0.65374085940294579533,0.65374085940294579533]),
            # "desired_state": np.deg2rad([0.65374085940294579533, 29.13318501]),
            # "desired_state": np.deg2rad([29.13318501]),
            "desired_state": np.deg2rad([38.525680187183610315]),  # 57.29 38.525680187183610315  22.9
            # "desired_state": np.deg2rad([31.455094774119341849,0]),
            # Probability of seeding the robot in finale position
            "reset_final_p": 0.1,
            # Termination conditions
            "terminate_upside_down": True,
            "terminate_gyro": True,
            "terminate_shock": False,
            # Randomization
            "random_angles": 1.5,  # [+- deg]
            "random_time_ratio": [0.98, 1.02],  # [+- ratio]
            "random_friction": [0.25, 1.5],  # [friction]
            "random_body_mass": [0.8, 1.2],  # [+- ratio]
            "random_body_com_pos": 5e-3,  # [+- m]
            "random_damping": [0.8, 1.2],  # [+- ratio]
            "random_frictionloss": [0.8, 1.2],  # [+- ratio]
            "random_v": [13.8, 16.8],  # [volts]
            "nominal_v": 15,  # [volts]
            # Control type (position, velocity or error)
            "control": "velocity",
            "interpolate": True,
            # Delay for velocity [s]
            "qdot_delay": 0.030,
            "tilt_delay": 0.050,
            "dtilt_delay": 0.050,
            # Previous actions
            "previous_actions": 1,
        }

        self.options.update(options or {})
        self.render_mode = render_mode

        print(robot_name)
        self.folder_name = robot_name
        # self.folder_name = ["bez1"]  # robot_name
        # self.folder_name = ["op3_rot"]
        # self.folder_name = ["bez2"]
        # self.folder_name = ["bez3"]
        # self.folder_name = ["sigmaban"]
        # self.folder_name = ["wolfgang"]
        # self.folder_name = ["nugus"]

        self.multi = False
        self.record = False
        self.stand = False
        self.stand_time = float("inf")

        self.pose_init = "none"

        self.stand_success_count = [[], [], []]  # all, Front, back, side
        self.n_ep = -1
        self.tot_time_to_stand = [[], [], []]
        self.init_pose = [0, 0, 0]  # Front, back, side
        self.hei = {
            "sigmaban": 0.67,
            "bez2": 0.54,
            "bez3": 0.62,
            "op3_rot": 0.49199,
            "bez1": 0.48083660868911876,
            "wolfgang": 0.7673033792122936,
            "nugus": 0.8086855785416924,
            "uw": 0.48083660868911876,
        }
        self.desired_height = self.hei[self.folder_name[0]]

        self.count = [0] * len(self.folder_name)
        self.current_index = 0
        self.sim = Simulator(scene_name=f"scene_{self.folder_name[self.current_index]}.xml")

        # Loading initial configuration cache
        self.initial_config = None
        for name in self.folder_name:
            initial_config_path = self.get_initial_config_filename(name)
            if os.path.exists(initial_config_path):
                if self.initial_config is None:
                    self.initial_config = []
                # print(f"Loading initial configurations from {initial_config_path}")
                with open(initial_config_path, "rb") as f:
                    self.initial_config.append(pickle.load(f))

        # assert len(self.initial_config) == len(self.folder_name)

        # Degrees of freedom involved
        self.dofs = ["elbow", "shoulder_pitch", "hip_pitch", "knee", "ankle_pitch"]
        # Action space is q
        max_variation = self.options["dt"] * self.options["vmax"]
        self.delta_max = np.array([max_variation] * len(self.dofs))

        self.get_indexes()

        # Window for qdot delay simulation
        self.q_history = []
        self.q_history_size = max(1, round(self.options["qdot_delay"] / self.sim.dt))

        # Window for tilt delay simulation
        self.tilt_history = []
        self.tilt_history_size = max(1, round(self.options["tilt_delay"] / self.sim.dt))

        self.height_history = []
        self.height_history_size = max(1, round(self.options["tilt_delay"] / self.sim.dt))

        self.dtilt_history = []
        self.dtilt_history_size = max(1, round(self.options["dtilt_delay"] / self.sim.dt))

        self.accel_history = []
        self.accel_history_size = max(1, round(self.options["tilt_delay"] / self.sim.dt))
        # self.options["dt"] = 0.01

    def get_indexes(self):
        self.ranges = [self.sim.model.actuator(f"left_{dof}").ctrlrange for dof in self.dofs]

        # Pre-fetching indexes and sites for faster evaluation
        self.trunk_site = self.sim.data.site("trunk")
        self.left_actuators_indexes = [self.sim.get_actuator_index(f"left_{dof}") for dof in self.dofs]
        self.right_actuators_indexes = [self.sim.get_actuator_index(f"right_{dof}") for dof in self.dofs]
        self.range_low = np.array([r[0] for r in self.ranges])
        self.range_high = np.array([r[1] for r in self.ranges])

        if self.options["control"] == "position":
            self.action_space = spaces.Box(
                np.array(self.range_low),
                np.array(self.range_high),
                dtype=np.float32,
            )
        elif self.options["control"] == "velocity":
            self.action_space = spaces.Box(
                np.array([-self.options["vmax"]] * len(self.dofs), dtype=np.float32),
                np.array([self.options["vmax"]] * len(self.dofs), dtype=np.float32),
                dtype=np.float32,
            )
        elif self.options["control"] == "error":
            self.action_space = spaces.Box(
                np.array([-np.pi / 4] * len(self.dofs), dtype=np.float32),
                np.array([np.pi / 4] * len(self.dofs), dtype=np.float32),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown control type: {self.options['control']}")

        # Observation is q_pitch, dqpitch, tilt, dtilt
        self.observation_space = spaces.Box(
            np.array(
                [
                    # q (0-4)
                    *(-np.pi * np.ones(len(self.dofs))),
                    # dq (5-9)
                    *(-10 * np.ones(len(self.dofs))),
                    # ctrl (10-14)
                    *self.range_low,
                    # tilt (15)
                    *np.array([-np.pi, -np.pi, -np.pi]),
                    # dtilt (16)
                    *np.array([-10, -10, -10]),
                    # accel
                    # *np.array([-40, -40, -40]),
                    # height (17)
                    0,
                    # Previous action
                    *(list(self.action_space.low) * self.options["previous_actions"]),
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    # q (0-4)
                    *(np.pi * np.ones(len(self.dofs))),
                    # dq (5-9)
                    *(10 * np.ones(len(self.dofs))),
                    # ctrl (10-14)
                    *self.range_high,
                    # tilt (15)
                    *np.array([np.pi, np.pi, np.pi]),
                    # dtilt (16)
                    *np.array([10, 10, 10]),
                    # accel
                    # *np.array([40, 40, 40]),
                    # height (17)
                    1,
                    # Previous action
                    *(list(self.action_space.high) * self.options["previous_actions"]),
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Values for randomization
        self.trunk_body = "torso_2023"
        self.trunk_mass = self.sim.model.body(self.trunk_body).mass.copy()
        self.trunk_ipos = self.sim.model.body(self.trunk_body).ipos.copy()
        self.gainprm_original = self.sim.model.actuator_gainprm.copy()
        self.biasprm_original = self.sim.model.actuator_biasprm.copy()
        self.damping_original = self.sim.model.dof_damping.copy()
        self.frictionloss_original = self.sim.model.dof_frictionloss.copy()
        self.forcerange_original = self.sim.model.actuator_forcerange.copy()
        self.body_quat_original = self.sim.model.body_quat.copy()

    def get_initial_config_filename(self, folder: str) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), f"{folder}/standup_initial_configurations.pkl")

    def apply_control(self, q: np.ndarray, reset: bool = False) -> None:
        self.sim.data.ctrl[self.left_actuators_indexes] = q
        self.sim.data.ctrl[self.right_actuators_indexes] = q

        if reset:
            for k, dof in enumerate(self.dofs):
                self.sim.set_control(f"left_{dof}", q[k], True)
                self.sim.set_control(f"right_{dof}", q[k], True)

    def get_tilt(self):
        R = self.trunk_site.xmat
        return -np.arctan2(R[7], R[8])

    def get_observation(self) -> np.ndarray:
        # Retrieving joints
        q = self.q_history[-1]
        q_dot = (np.array(self.q_history[-1]) - np.array(self.q_history[0])) / (self.q_history_size * self.sim.dt)

        # Current control
        ctrl = [self.sim.get_control(f"left_{dof}") for dof in self.dofs]
        ctrl = np.array(ctrl)

        # Tilt, dtilt
        tilt = self.tilt_history[0]
        dtilt = self.dtilt_history[0]
        accel = self.accel_history[0]

        height = self.height_history[0]

        return np.array(
            [
                *q,
                *q_dot,
                *ctrl,
                *tilt,
                *dtilt,
                # *accel,
                height,
                *(np.array(self.previous_actions).flatten()),
            ],
            dtype=np.float32,
        )

    def step(self, action):
        action = np.array(action)
        # Current control
        start_ctrl = [self.sim.get_control(f"left_{dof}") for dof in self.dofs]

        # Applying the action
        if self.options["control"] == "position":
            # Action is a q
            target_ctrl_unclipped = action
        elif self.options["control"] == "velocity":
            # Action is a q variation
            target_ctrl_unclipped = start_ctrl + action * self.options["dt"]
        elif self.options["control"] == "error":
            # Action is an error w.r.t the read configuration
            start_q = [self.sim.get_q(f"left_{dof}") for dof in self.dofs]
            target_ctrl_unclipped = np.array(start_q) + action

        # Limiting the control
        target_ctrl = np.clip(
            target_ctrl_unclipped,
            start_ctrl - self.delta_max * self.time_ratio,
            start_ctrl + self.delta_max * self.time_ratio,
        )

        # Limiting the control to the range
        target_ctrl = np.clip(target_ctrl, self.range_low, self.range_high)
        # print(np.rad2deg(self.sim.get_rpy()))
        # Not terminating episode
        done = False
        shock = False

        # Step the simulation
        timesteps = round(self.time_ratio * self.options["dt"] / self.sim.dt)
        for k in range(timesteps):
            if self.options["interpolate"]:
                alpha = (k + 1) / timesteps
                self.apply_control(start_ctrl + alpha * (target_ctrl - start_ctrl))
            else:
                self.apply_control(target_ctrl)
            self.sim.step()

            if self.options["terminate_shock"]:
                centroidal_force = self.sim.centroidal_force()
                if centroidal_force - self.centroidal_force > 200.0:
                    shock = True
                self.centroidal_force = centroidal_force

            q = [self.sim.get_q(f"left_{dof}") for dof in self.dofs]
            self.q_history.append(q)
            self.tilt_history.append(self.sim.get_rpy())
            self.dtilt_history.append(self.sim.get_gyro())
            self.accel_history.append(self.sim.get_accel())
            self.height_history.append(self.sim.get_head_height())

            if self.render_mode == "human":
                self.sim.render(self.options["render_realtime"])
        # Terminating in case of upside down robot
        if self.options["terminate_upside_down"]:
            tilt = self.get_tilt()
            if np.rad2deg(np.abs(tilt)) > 135:
                done = True
                # print("terminate_upside_down")

        # Penalizing high gyro
        if self.options["terminate_gyro"]:
            gyro = self.sim.get_gyro()
            if abs(gyro[1]) > 7.5:
                done = True
                # print("terminate_gyro")

        # Shock termination
        if self.options["terminate_shock"] and shock:
            done = True
            # print("terminate_shock")

        self.q_history = self.q_history[-self.q_history_size :]
        self.tilt_history = self.tilt_history[-self.tilt_history_size :]
        self.dtilt_history = self.dtilt_history[-self.dtilt_history_size :]
        self.height_history = self.height_history[-self.height_history_size :]
        self.accel_history = self.accel_history[-self.accel_history_size :]
        # print(f"Height: {self.height_history[-1]},  Pitch: {self.sim.get_rpy()[1]}")
        if self.record:
            if ((abs(self.desired_height - self.height_history[-1]) / self.desired_height) * 100) < 20:
                if not self.stand:
                    self.stand_time = self.sim.t
                self.stand = True
            elif self.stand and ((abs(self.desired_height - self.height_history[-1]) / self.desired_height) * 100) > 20:
                self.stand = False
                self.stand_time = float("inf")

        # Extracting observation
        obs = self.get_observation()
        state_current = [self.height_history[-1]]
        reward = np.exp(-10 * (np.linalg.norm(np.array(state_current) - np.array(self.options["desired_state"])) ** 2))
        desired_height = 0.34  # self.desired_height[self.current_index]
        if (
            state_current[0] >= desired_height
        ):  # ((abs(desired_height - state_current[0]) / desired_height) * 100) < 10:
            reward += np.exp(-10 * (np.linalg.norm(np.array([self.tilt_history[-1][1]]) - np.array([0])) ** 2))
        # print(f"PITCH: {np.rad2deg(self.tilt_history[-1][1])} Pitch2: {np.rad2deg(self.get_tilt())}")

        action_variation = np.abs(action - self.previous_actions[-1])
        self.previous_actions.append(action)
        self.previous_actions = self.previous_actions[-self.options["previous_actions"] :]

        # Penalizing velocities
        if self.options["control"] == "position":
            # Penalizing control velocity
            ctrl_velocity = (target_ctrl - start_ctrl) / self.options["dt"]
            reward += np.exp(-np.linalg.norm(ctrl_velocity)) * 1e-1

            # Penalizing action variation
            reward += np.exp(-np.linalg.norm(action_variation)) * 5e-2
        elif self.options["control"] == "velocity":
            # Penalizing action
            reward += np.exp(-np.linalg.norm(action)) * 1e-1

            # Penalizing action variation
            reward += np.exp(-np.linalg.norm(action_variation)) * 5e-2
        elif self.options["control"] == "error":
            # Penalizing action
            reward += np.exp(-np.linalg.norm(action)) * 1e-1

            # Penalizing action variation
            reward += np.exp(-np.linalg.norm(action_variation)) * 5e-2

        # No self collisions reward
        reward += np.exp(-self.sim.self_collisions()) * 1e-1

        # Terminating the episode after the trucate_duration
        # print(self.options["truncate_duration"])
        # terminated = self.sim.t > 10
        terminated = self.sim.t > self.options["truncate_duration"]
        # print(self.get_tilt())
        return obs, reward, done, terminated, {}

    def apply_angular_offset(self, joint: str, offset: float):
        body_id = self.sim.model.joint(joint).bodyid
        quat = self.sim.model.body_quat[body_id][0]
        quat = tf.quaternion_multiply(quat, tf.quaternion_about_axis(offset, [0, 0, 1]))
        self.sim.model.body_quat[body_id] = quat

    def apply_randomization(self):
        # Randomizing the time ratio
        self.time_ratio = self.np_random.uniform(*self.options["random_time_ratio"])

        # Randomizing the floor friction
        self.sim.set_floor_friction(self.np_random.uniform(*self.options["random_friction"]))

        # Randomizing the mass
        self.sim.model.body(self.trunk_body).mass = self.trunk_mass * self.np_random.uniform(
            *self.options["random_body_mass"]
        )

        # Randomizing the center of mass
        self.sim.model.body(self.trunk_body).ipos = self.trunk_ipos + self.np_random.uniform(
            -self.options["random_body_com_pos"],
            self.options["random_body_com_pos"],
            size=3,
        )

        # Randomizing the voltage
        volts = self.np_random.uniform(*self.options["random_v"])
        volts_ratio = volts / self.options["nominal_v"]
        self.sim.model.actuator_gainprm = self.gainprm_original * volts_ratio
        self.sim.model.actuator_biasprm = self.biasprm_original * volts_ratio
        self.sim.model.actuator_forcerange = self.forcerange_original * volts_ratio

        # Randomize the damping
        damping_ratio = self.np_random.uniform(*self.options["random_damping"])
        self.sim.model.dof_damping = self.damping_original * damping_ratio

        frictionloss_ratio = self.np_random.uniform(*self.options["random_frictionloss"])
        self.sim.model.dof_frictionloss = self.frictionloss_original * frictionloss_ratio

        # Restoring body quats
        self.sim.model.body_quat = self.body_quat_original.copy()

        # Adding error angles
        err_rad = np.deg2rad(self.options["random_angles"])
        for dof in self.sim.dof_names():
            self.apply_angular_offset(dof, self.np_random.uniform(-err_rad, err_rad))

    def randomize_fall(self, target: bool = False):
        # Decide if we will use the target
        my_target = np.copy(self.options["desired_state"])
        if target is False:
            target = self.np_random.random() < self.options["reset_final_p"]

        # Selecting a random configuration
        initial_q = self.np_random.uniform(low=-np.pi, high=np.pi, size=(len(self.dofs),))

        # If target, we will use the q_target
        if target:
            initial_q = my_target[: len(self.dofs)]
            offset = self.np_random.uniform(-0.1, 0.1)
            initial_q[2] -= offset
            initial_q[3] += offset * 2
            initial_q[4] -= offset

        initial_q = np.clip(initial_q, self.range_low, self.range_high)
        self.apply_control(initial_q)

        # Set the robot initial pose
        self.sim.step()
        initial_tilt = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        if target:
            initial_tilt = my_target[-1]
        T_world_trunk = tf.rotation_matrix(initial_tilt, [0, 1, 0])
        T_world_trunk[:3, 3] = [0, 0, 0.4]

        self.sim.set_T_world_site("trunk", T_world_trunk)

        # Wait for the robot to stabilize
        for _ in range(round(self.options["stabilization_time"] / self.sim.dt)):
            self.sim.step()

    def reset(
        self,
        seed: int = None,  # type: ignore[assignment]
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if self.record:

            if self.stand:
                if self.pose_init == "front":
                    self.tot_time_to_stand[0].append(self.stand_time)
                elif self.pose_init == "back":
                    self.tot_time_to_stand[1].append(self.stand_time)
                elif self.pose_init == "up":
                    self.tot_time_to_stand[2].append(self.stand_time)

            if self.pose_init == "front":
                self.stand_success_count[0].append(int(self.stand))
            elif self.pose_init == "back":
                self.stand_success_count[1].append(int(self.stand))
            elif self.pose_init == "up":
                self.stand_success_count[2].append(int(self.stand))
            # print(self.init_pose)
            if self.n_ep == 99:
                print(f"Ep: {self.n_ep}")
                print(self.folder_name[0])
                tot = []
                for i in self.stand_success_count:
                    for j in i:
                        tot.append(j)
                tot2 = []
                for i in self.tot_time_to_stand:
                    for j in i:
                        tot2.append(j)
                print(f"Stand: {self.stand}, Stand time: {self.stand_time}")
                print(f"{np.mean(tot)},{np.std(tot)},{sum(tot)},{sum(self.init_pose)},{np.mean(tot2)},{np.std(tot2)}")
                with open(f"{self.folder_name[0]}_tmp.txt", "a") as file:
                    file.write(
                        f"{np.mean(tot)},{np.std(tot)},{sum(tot)},{sum(self.init_pose)},{np.mean(tot2)},{np.std(tot2)}\n"
                    )
                # if self.init_pose[0] > 0:
                #     print(f"{np.mean(self.stand_success_count[0])},"
                #           f"{np.std(self.stand_success_count[0])},{sum(self.stand_success_count[0])},{self.init_pose[0]},"
                #           f"{np.mean(self.tot_time_to_stand[0])},{np.std(self.tot_time_to_stand[0])}")
                # if self.init_pose[1] > 0:
                #     print(f"{np.mean(self.stand_success_count[1])},"
                #           f"{np.std(self.stand_success_count[1])},{sum(self.stand_success_count[1])},{self.init_pose[1]},"
                #           f"{np.mean(self.tot_time_to_stand[1])},{np.std(self.tot_time_to_stand[1])}")
                # if self.init_pose[2] > 0:
                #     print(f"{np.mean(self.stand_success_count[2])},"
                #           f"{np.std(self.stand_success_count[2])},{sum(self.stand_success_count[2])},{self.init_pose[2]},"
                #           f"{np.mean(self.tot_time_to_stand[2])},{np.std(self.tot_time_to_stand[2])}")

            self.n_ep += 1
        if self.multi:
            self.current_index = random.choice(range(len(self.folder_name)))
            self.count[self.current_index] += 1
            # print(self.count)
            # self.current_index = 6
            # self.current_index = (self.current_index + 1) % len(self.scene_names)
            new_scene = f"scene_{self.folder_name[self.current_index]}.xml"

            self.sim.close_viewer()
            self.sim = Simulator(scene_name=new_scene)
            self.sim.reset()
            self.get_indexes()
        else:
            self.sim.reset()

        # TODO add here code for randomizing the selection check
        options = options or {}
        target = options.get("target", False)
        use_cache = options.get("use_cache", True)

        if use_cache and self.initial_config[self.current_index] is None:
            warnings.warn(
                "use_cache=True but no initial config file could be loaded. Did you run standup_generate_initial.py?"
            )

        # Initial robot configuration
        if use_cache and self.initial_config[self.current_index] is not None:
            qpos, ctrl = random.choice(self.initial_config[self.current_index])
            self.sim.data.qpos = qpos
            self.sim.data.ctrl = ctrl
            self.sim.data.qvel *= 0
            self.sim.step()
        else:
            self.randomize_fall(target)

        # Randomization
        # self.apply_randomization()
        self.time_ratio = self.np_random.uniform(*self.options["random_time_ratio"])

        self.sim.reset_render()

        # Taking arms a little bit inside
        self.sim.set_control("left_shoulder_roll", -np.deg2rad(5))
        self.sim.set_control("right_shoulder_roll", np.deg2rad(5))

        # Initializing the history
        q = [self.sim.get_q(f"left_{dof}") for dof in self.dofs]
        self.q_history = [q] * self.q_history_size

        # Initializing the tilt history
        self.tilt_history = [self.sim.get_rpy()] * self.tilt_history_size
        self.dtilt_history = [self.sim.get_gyro()] * self.dtilt_history_size
        self.height_history = [self.sim.get_head_height()] * self.height_history_size
        self.accel_history = [self.sim.get_accel()] * self.accel_history_size
        # Initializing the previous action
        self.previous_actions = [np.zeros(len(self.dofs))] * self.options["previous_actions"]

        self.centroidal_force = self.sim.centroidal_force()

        if self.record:
            pitch = np.rad2deg(self.sim.get_rpy()[1])
            if pitch > 30:
                self.init_pose[0] += 1  # [0, 0, 0]  # Front, back, up
                self.pose_init = "front"
            elif pitch < -30:
                self.init_pose[1] += 1  # [0, 0, 0]  # Front, back, up
                self.pose_init = "back"
            else:
                self.init_pose[2] += 1
                self.pose_init = "up"

        return self.get_observation(), {}

    def render(self):
        self.render_mode = "human"
