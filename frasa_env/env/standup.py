import numpy as np

from .standup_env import StandupEnv


class FRASAEnv(StandupEnv):
    def __init__(self, evaluation=False, render_mode="none", options=None):
        options = options or {}
        options["stabilization_time"] = 2.0
        options["truncate_duration"] = 5.0
        options["dt"] = 0.05
        options["vmax"] = 2 * np.pi
        options["reset_final_p"] = 0.1

        super().__init__(evaluation=evaluation, render_mode=render_mode, options=options)
