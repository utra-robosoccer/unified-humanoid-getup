import os
import time
from typing import Optional
import meshcat.transformations as tf
import mujoco
import mujoco.viewer
import numpy as np


class Simulator:
    def __init__(self, model_dir: Optional[str] = None, scene_name: str = "scene_sig.xml"):
        # If model_dir is not provided, use the current directory + /model/
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "model")
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
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                for i in range(self.model.nu)]

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

    def reset_render(self) -> None:
        self.t = 0
        self.frame = 0
        self.viewer_start = time.time()

    def get_range(self, name: str) -> np.ndarray:
        return self.model.joint(name).range

    def get_q(self, name: str) -> float:
        addr = self.model.jnt_qposadr[self.dofs_to_index[name]]
        return self.data.qpos[addr]

    def get_qdot(self, name: str) -> float:
        addr = self.model.jnt_dofadr[self.dofs_to_index[name]]
        return self.data.qvel[addr]

    def set_q(self, name: str, value: float) -> None:
        addr = self.model.jnt_qposadr[self.dofs_to_index[name]]
        self.data.qpos[addr] = value

    def get_control(self, name: str) -> float:
        actuator_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return self.data.ctrl[actuator_idx]

    def get_actuator_index(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def set_control(self, name: str, value: float, reset: bool = False) -> None:
        actuator_idx = self.get_actuator_index(name)
        self.data.ctrl[actuator_idx] = value
        if reset:
            self.set_q(name, value)

    def get_gyro(self) -> np.ndarray:
        return self.data.sensor("gyro").data

    def get_accel(self) -> np.ndarray:
        return self.data.sensor("accelerometer").data

    def get_T_world_body(self, body_name: str) -> np.ndarray:
        T = np.eye(4)
        body = self.data.body(body_name)
        T[:3, :3] = body.xmat.reshape(3, 3)
        T[:3, 3] = body.xpos
        return T

    def get_T_world_site(self, site_name: str) -> np.ndarray:
        T = np.eye(4)
        site = self.data.site(site_name)
        T[:3, :3] = site.xmat.reshape(3, 3)
        T[:3, 3] = site.xpos
        return T

    def get_T_world_fbase(self) -> np.ndarray:
        data = self.data.joint("root").qpos
        quat = data[3:]
        pos = data[:3]
        T = tf.quaternion_matrix(quat)
        T[:3, 3] = pos
        return T

    def set_T_world_fbase(self, T: np.ndarray) -> None:
        joint = self.data.joint("root")
        quat = tf.quaternion_from_matrix(T)
        pos = T[:3, 3]
        joint.qpos[:] = [*pos, *quat]
        self.reset_velocity()

    def reset_velocity(self) -> None:
        self.data.qvel[:] = 0

    def set_T_world_body(self, body_name: str, T_world_bodyTarget: np.ndarray) -> None:
        T_world_fbase = self.get_T_world_fbase()
        T_world_body = self.get_T_world_body(body_name)
        T_body_fbase = np.linalg.inv(T_world_body) @ T_world_fbase
        self.set_T_world_fbase(T_world_bodyTarget @ T_body_fbase)

    def set_T_world_site(self, site_name: str, T_world_siteTarget: np.ndarray) -> None:
        T_world_fbase = self.get_T_world_fbase()
        T_world_site = self.get_T_world_site(site_name)
        T_site_fbase = np.linalg.inv(T_world_site) @ T_world_fbase
        self.set_T_world_fbase(T_world_siteTarget @ T_site_fbase)

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
        self.model.opt.gravity[:] = gravity

    def render(self, realtime: bool = True):
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
        left_foot = self.get_T_world_site('left_foot')[0:3, 3]
        right_foot = self.get_T_world_site('right_foot')[0:3, 3]
        foot = (left_foot + right_foot) / 2
        head_height = self.get_T_world_site('camera')[2, 3] - foot[2]
        if self.get_T_world_site('camera')[2, 3] < foot[2]:
            head_height = -10
        return head_height

    def get_rpy(self, site: str = "trunk") -> np.ndarray:
        R = self.data.site(site).xmat
        # Using the approach that returns a continuous pitch beyond +/-90°.
        pitch = -np.arctan2(R[6], R[8])
        roll = np.arctan2(R[7], R[8])
        yaw = np.arctan2(R[3], R[0])
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


class SwappableSimulator:
    """
    Wraps multiple robot models so that only one is active in the scene at any time.
    When a swap is needed, the simulator reinitializes the scene with a new XML model.
    """

    def __init__(self, model_dir: Optional[str] = None, scene_names: Optional[list] = None):
        if scene_names is None:
            scene_names = ["scene_bez.xml", "scene_sig.xml", "scene_bez3.xml"]
        self.scene_names = scene_names
        self.current_index = 0  # Start with the first model

        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), "model")
        # Instantiate the initial simulator.
        self.sim = Simulator(model_dir=self.model_dir, scene_name=self.scene_names[self.current_index])
        # Optionally, initialize a viewer.
        self.sim.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
        self.sim.step()
        self.sim.set_T_world_site("left_foot", np.eye(4))
        self.last_switch_time = time.time()

    def swap_model(self):
        """
        Swap the current model with the next one in the scene_names list.
        This reinitializes the Simulator so that only one robot model is in the scene.
        """
        # Increment and wrap around the index.
        self.current_index = (self.current_index + 1) % len(self.scene_names)
        new_scene = self.scene_names[self.current_index]
        print(f"Swapping model to: {new_scene}")
        # Reinstantiate the Simulator with the new XML file.
        self.sim.close_viewer()
        self.sim = Simulator(model_dir=self.model_dir, scene_name=new_scene)
        # Re-launch the viewer for the new simulation.
        self.sim.viewer = mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
        self.sim.reset_render()
        self.sim.step()
        self.sim.set_T_world_site("left_foot", np.eye(4))
        self.last_switch_time = time.time()

    def run(self, run_time: float = 30.0, switch_interval: float = 5.0):
        """
        Run the simulation for a given time, swapping the active robot at the given interval.
        """
        start = time.time()
        while time.time() - start < run_time:
            # Check if it's time to swap.
            if time.time() - self.last_switch_time > switch_interval:
                self.swap_model()
            self.sim.step()
            self.sim.render(realtime=True)
            # Optional: print some feedback.
            self.sim.set_control("left_elbow", np.sin(self.sim.t))
            rpy = self.sim.get_rpy("trunk")
            print(f"Time: {self.sim.t:.2f}, rpy (deg): {np.rad2deg(rpy)}")


if __name__ == "__main__":
    # Instantiate SwappableSimulator with two model XML files.
    swapper = SwappableSimulator(scene_names=["scene_sig.xml", "scene_bez.xml", "scene_bez3.xml"])
    swapper.run(run_time=30.0, switch_interval=5.0)
