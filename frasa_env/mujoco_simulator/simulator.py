import os
import time
from typing import Optional

import meshcat.transformations as tf
import mujoco
import mujoco.viewer
import numpy as np


class Simulator:
    def __init__(self, model_dir: Optional[str] = None, scene_name:str = "scene_sig.xml"):
        # If model_dir is not provided, use the current directory
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__) + "/model/")
        self.model_dir = model_dir

        # Load the model and data
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(f"{model_dir}/{scene_name}")
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        # Retrieve the degrees of freedom id/name pairs
        joints = len(self.model.jnt_pos)
        self.dofs = [[k, self.model.jnt(k).name] for k in range(1, joints)]
        self.dofs_to_index = {dof: k for k, dof in self.dofs}

        self.viewer = None
        self.t: float = 0.0
        self.dt: float = self.model.opt.timestep
        self.frame: int = 0
        self.data.ctrl[:] = 0

    def set_floor_friction(self, friction: float) -> None:
        self.model.geom("floor").friction[0] = friction
        self.model.geom("floor").priority = 1

    def self_collisions(self) -> float:
        forcetorque = np.zeros(6)
        contacts = self.data.contact
        selector = (contacts.geom[:, 0] != 0) * (contacts.geom[:, 1] != 0)
        forces = 0.0
        for id in np.argwhere(selector):
            mujoco.mj_contactForce(self.model, self.data, id, forcetorque)
            forces += np.linalg.norm(forcetorque[:3])

        return forces

    def centroidal_force(self) -> float:
        return np.linalg.norm(self.data.qfrc_constraint[3:])

    def dof_names(self) -> list:
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

    def reset_render(self) -> None:
        self.t = 0
        self.frame = 0
        self.viewer_start = time.time()

    def get_range(self, name: str) -> np.ndarray:
        return self.model.joint(name).range

    def get_q(self, name: str) -> float:
        """
        Gets the position of a given joint.

        Args:
            name (str): joint name

        Returns:
            float: joint position
        """
        addr = self.model.jnt_qposadr[self.dofs_to_index[name]]
        return self.data.qpos[addr]

    def get_qdot(self, name: str) -> float:
        """
        Gets the velocity of a given joint.

        Args:
            name (str): joint name

        Returns:
            float: joint velocity
        """
        addr = self.model.jnt_dofadr[self.dofs_to_index[name]]
        return self.data.qvel[addr]

    def set_q(self, name: str, value: float) -> None:
        """
        Sets a value of a given joint.

        Args:
            name (str): joint name
            value (float): target value
        """
        addr = self.model.jnt_qposadr[self.dofs_to_index[name]]
        self.data.qpos[addr] = value

    def get_control(self, name: str) -> None:
        """
        Gets the control for a given actuator

        Args:
            name (str): actuator name
        """
        actuator_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return self.data.ctrl[actuator_idx]

    def get_actuator_index(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def set_control(self, name: str, value: float, reset: bool = False) -> None:
        """
        Sets the control for a given actuator.
        If the actuator is a position actuator, the value is the desired position.
        If the actuator is a motor actuator, the value is the desired torque.

        Args:
            name (str): actuator name
            value (float): target value
        """
        actuator_idx = self.get_actuator_index(name)
        self.data.ctrl[actuator_idx] = value

        if reset:
            self.set_q(name, value)

    def get_gyro(self) -> np.ndarray:
        """
        Gets the gyroscope data.

        Returns:
            np.ndarray: gyroscope data
        """
        return self.data.sensor("gyro").data
    def get_accel(self) -> np.ndarray:
        """
        Gets the gyroscope data.

        Returns:
            np.ndarray: gyroscope data
        """
        return self.data.sensor("accelerometer").data

    def get_T_world_body(self, body_name: str) -> np.ndarray:
        """
        Gets the transformation from world to body frame.

        Args:
            body_name (str): body name
        """
        T = np.eye(4)
        body = self.data.body(body_name)
        T[:3, :3] = body.xmat.reshape(3, 3)
        T[:3, 3] = body.xpos
        return T

    def get_T_world_site(self, site_name: str) -> np.ndarray:
        """
        Gets the transformation from world to site frame.

        Args:
            site_name (str): site name
        """
        T = np.eye(4)
        site = self.data.site(site_name)
        T[:3, :3] = site.xmat.reshape(3, 3)
        T[:3, 3] = site.xpos
        return T

    def get_T_world_fbase(self, root_name:str = "root") -> np.ndarray:
        """
        Gets the transformation from world to floating base frame.
        """
        data = self.data.joint(root_name).qpos
        quat = data[3:]
        pos = data[:3]

        T = tf.quaternion_matrix(quat)
        T[:3, 3] = pos
        return T

    def set_T_world_fbase(self, T: np.ndarray, root_name:str = "root") -> None:
        """
        Updates the floating base so that a body transformation match the target one

        Args:
            T (np.ndarray): target transformation
        """
        joint = self.data.joint(root_name)

        quat = tf.quaternion_from_matrix(T)
        pos = T[:3, 3]

        joint.qpos[:] = [*pos, *quat]
        self.reset_velocity()

    def reset_velocity(self) -> None:
        """
        Resets the velocity of all the joints
        """
        self.data.qvel[:] = 0

    def set_T_world_body(self, body_name: str, T_world_bodyTarget: np.ndarray, root_name:str = "root") -> None:
        """
        Updates the floating base so that a body transformation match the target one

        Args:
            body_name (str): body name
        """
        T_world_fbase = self.get_T_world_fbase(root_name)
        T_world_body = self.get_T_world_body(body_name)
        T_body_fbase = np.linalg.inv(T_world_body) @ T_world_fbase

        self.set_T_world_fbase(T_world_bodyTarget @ T_body_fbase)

    def set_T_world_site(self, site_name: str, T_world_siteTarget: np.ndarray, root_name:str = "root") -> None:
        """
        Updates the floating base so that a site transformation match the target one

        Args:
            site_name (str): site name
        """
        T_world_fbase = self.get_T_world_fbase(root_name)
        T_world_site = self.get_T_world_site(site_name)
        T_site_fbase = np.linalg.inv(T_world_site) @ T_world_fbase

        self.set_T_world_fbase(T_world_siteTarget @ T_site_fbase, root_name)

    def get_pressure_sensors(self) -> dict:
        left_pressures = [
            -self.data.sensor("left_foot_cleat_front_right").data[2],
            -self.data.sensor("left_foot_cleat_front_left").data[2],
            -self.data.sensor("left_foot_cleat_back_right").data[2],
            -self.data.sensor("left_foot_cleat_back_left").data[2],
        ]
        right_pressures = [
            -self.data.sensor("right_foot_cleat_front_right").data[2],
            -self.data.sensor("right_foot_cleat_front_left").data[2],
            -self.data.sensor("right_foot_cleat_back_right").data[2],
            -self.data.sensor("right_foot_cleat_back_left").data[2],
        ]

        return {"left": left_pressures, "right": right_pressures}

    def step(self) -> None:
        self.t = self.frame * self.dt
        mujoco.mj_step(self.model, self.data)
        self.frame += 1

    def set_gravity(self, gravity: np.ndarray) -> None:
        """
        Sets the gravity vector.

        Args:
            gravity (np.ndarray): gravity vector
        """
        self.model.opt.gravity[:] = gravity

    def render(self, realtime: bool = True):
        """
        Renders the visualization of the simulation.

        Args:
            realtime (bool, optional): if True, render will sleep to ensure real time viewing. Defaults to True.
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.reset_render()
        if not hasattr(self, "viewer_start"):
            self.reset_render()
        # This snippet shows how you might reload the model
        # after a certain time. In a switching mechanism, you would reinstantiate
        # a new Simulator instead.
        if realtime:
            current_ts = self.viewer_start + self.frame * self.dt
            to_sleep = current_ts - time.time()
            if to_sleep > 0:
                time.sleep(to_sleep)

        self.viewer.sync()
    def get_head_height(self):
        left_foot = self.get_T_world_site('left_foot')[0:3][:, 3]
        right_foot = self.get_T_world_site('right_foot')[0:3][:, 3]
        foot = (left_foot + right_foot) / 2
        head_height = self.get_T_world_site('camera')[2][3] - foot[2]
        if self.get_T_world_site('camera')[2][3] < foot[2]:
            head_height = -10
        return head_height


    def get_rpy(self, site:str = "trunk") -> np.ndarray:
        R = self.data.site(site).xmat
        # pitch = np.arctan2(-R[6], np.sqrt(R[0] ** 2 + R[3] ** 2)) #
        pitch =-np.arctan2(R[6], R[8])
        roll = np.arctan2(R[7], R[8])  # atan2(R[2,1], R[2,2])
        yaw = np.arctan2(R[3], R[0])
        # fused, tilt = self.fused_tilt_from_xmat(R)
        # psi, theta, phi, h = fused

        # return np.array([phi, psi, theta])
        return np.array([roll, pitch, yaw])

    def close_viewer(self) -> None:
        """
        Closes the current viewer if one is open.
        """
        if self.viewer is not None:
            try:
                # Try to call a close or finish method if available.
                if hasattr(self.viewer, "close"):
                    self.viewer.close()
                elif hasattr(self.viewer, "finish"):
                    self.viewer.finish()
            except Exception as e:
                print(f"Error closing viewer: {e}")
            finally:
                self.viewer = None

if __name__ == "__main__":
    sim = Simulator(scene_name="scene_sig.xml")
    # sim = Simulator(scene_name="scene_bez.xml")

    # sim = Simulator(scene_name="scene_bez3.xml")
    # sim = Simulator(scene_name="scene_bez1.xml")

    # sim = Simulator(scene_name="scene_bitbot.xml")

    # sim = Simulator(scene_name="scene_op3.xml")
    # sim = Simulator(scene_name="scene_nugus.xml") #euler="-1.57  0 0.2 "  left_hip_pitch
    sim.step()
    x = np.eye(4)
    x[:3, 3] = [0, 0, 0]
    sim.set_T_world_site("b1_left_foot", x, root_name="b1_root")
    x = np.eye(4)
    x[:3, 3] = [0, 0.275, 0]
    sim.set_T_world_site("op3_left_foot", x, root_name="op3_root")
    x = np.eye(4)
    x[:3, 3] = [0, 0.575, 0]
    sim.set_T_world_site("b2_left_foot", x, root_name="b2_root")
    x = np.eye(4)
    x[:3, 3] = [0, 0.9, 0]
    sim.set_T_world_site("b3_left_foot", x, root_name="b3_root")
    x = np.eye(4)
    x[:3, 3] = [0, 1.275, 0]
    sim.set_T_world_site("left_foot",x)
    x = np.eye(4)
    x[:3, 3] = [0, 1.7, 0]
    sim.set_T_world_site("bb_left_foot", x, root_name="bb_root")
    x = np.eye(4)
    x[:3, 3] = [0, 2.05, 0]
    sim.set_T_world_site("nug_left_foot", x, root_name="nug_root")

    sim.step()
    start = time.time()
    model_dir = os.path.join(os.path.dirname(__file__) + "/model/")
    once = True
    ran = np.linspace(-np.pi, np.pi, 100)
    count = 0
    ti = 0
    while True:
        sim.render(True)
        # print(sim.t)
        # sim.set_control("right_elbow", -1.2653655)
        dofs = ["elbow", "shoulder_pitch", "hip_pitch", "knee", "ankle_pitch"]
        tt = "ankle_pitch"
        # sim.set_control("right_"+tt, ran[count])
        # sim.set_control("left_"+tt, ran[count])
        # print(ran[count])
        ti += sim.dt
        if ti >= 0.03 and count < 99 and sim.t > 2:
            count+=1
            ti = 0



        # sim.set_T_world_site("left_foot", np.eye(4))
        # sim.set_control("left_elbow", 0.863938)
        # sim.set_control("right_elbow", 0.863938)
        # sim.set_control("right_shoulder_pitch", 0.3403392)
        # sim.set_control("left_shoulder_pitch", 0.3403392)
        # sim.set_control("right_hip_pitch", 0.907571)
        # sim.set_control("left_hip_pitch", 0.907571)
        # sim.set_control("right_knee", -1.4338038100846235)
        # sim.set_control("left_knee", -1.4338038100846235)
        # sim.set_control("right_ankle_pitch", 0.6370452)
        # sim.set_control("left_ankle_pitch", 0.6370452)

        # sim.set_control("left_elbow", 0)
        # sim.set_control("right_elbow", 0)
        # sim.set_control("right_shoulder_pitch", 0)
        # sim.set_control("left_shoulder_pitch", 0)
        # sim.set_control("right_hip_pitch", 0)
        # sim.set_control("left_hip_pitch", 0)
        # sim.set_control("right_knee", 0)
        # sim.set_control("left_knee", 0)
        # sim.set_control("right_ankle_pitch", 0)
        # sim.set_control("left_ankle_pitch", 0)

        #sig
        # sim.set_control("left_elbow", -0.863938)
        # sim.set_control("right_elbow", -0.863938)
        # sim.set_control("right_shoulder_pitch", 0.3403392)
        # sim.set_control("left_shoulder_pitch", 0.3403392)
        # sim.set_control("right_hip_pitch", -0.907571)
        # sim.set_control("left_hip_pitch", -0.907571)
        # sim.set_control("right_knee", 1.37881)
        # sim.set_control("left_knee", 1.37881)
        # sim.set_control("right_ankle_pitch", -0.6370452)
        # sim.set_control("left_ankle_pitch", -0.6370452)
        # R = sim.data.site("trunk").xmat
        # pitch = np.arctan2(R[6], R[8])
        # # print(R)
        #
        # x = [sim.get_actuator_index(f"left_{dof}") for dof in dofs]
        # # print(x)
        # # print(sim.get_T_world_site('camera')[0:3][:,3])
        # # print(sim.t)
        # # if sim.t > 5 and once:
        # #     sim.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(f"{model_dir}/scene_sig.xml")
        # #     sim.data: mujoco.MjData = mujoco.MjData(sim.model)
        # #     once = False
        # # print(sim.get_T_world_site('left_foot')[0:3][:,3])
        # # print(sim.get_T_world_site('right_foot')[0:3][:, 3])
        # print(f"Pitch2: {np.rad2deg(sim.get_rpy())}")
        #
        # left_foot = sim.get_T_world_site('left_foot')[2][3]
        # right_foot = sim.get_T_world_site('right_foot')[2][3]
        # foot = (left_foot+right_foot)/2
        # print(sim.get_rpy())
        # print(sim.get_head_height())
        # print(foot)
        # head_height = np.linalg.norm(sim.get_T_world_site('camera')[0:3][:,3] - foot)
        # print(sim.get_T_world_site('ball')[0:3][:,3])
        # print(head_height * (1-abs(pitch)) )
        # dofs = ["elbow", "shoulder_pitch", "hip_pitch", "knee", "ankle_pitch"]
        # ctrl = [sim.get_control(f"left_{dof}") for dof in dofs]
        # print(ctrl)

        sim.step()

        elapsed = time.time() - start
        frames = sim.frame
        print(f"Elapsed: {elapsed:.2f}, Frames: {frames}, FPS: {frames / elapsed:.2f}")
