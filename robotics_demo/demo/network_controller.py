import numpy as np
import torch

from robotics_demo.configs import ModelDefinitionConfig
from robotics_demo.demo.app import Demo
from robotics_demo.network import UVNetInDeptCombined


class DemoNetworkController(Demo):
    def __init__(
        self,
        robot: ModelDefinitionConfig,
        model_path,
        lower_limits,
        upper_limits,
        total_steps: int,
        dt: float,
        target_robot: ModelDefinitionConfig = None,
        target_q: np.ndarray = None,
        port: int = 7000,
    ):
        super().__init__(
            robot,
            lower_limits,
            upper_limits,
            dt,
            target_robot,
            target_q,
            port,
        )
        self.n_steps = total_steps
        self.net = UVNetInDeptCombined.load(model_path).get_control_network()

    def apply_control(self, x: np.ndarray):
        current_time = 0.0
        time_to_go = self.n_steps * self.dt
        self.record_a_step(current_time, x[:7])  # record initial state
        for n in range(self.n_steps):
            # get u from network u(time_to_go, x): use remaining time
            state = torch.tensor(
                np.concatenate([[time_to_go], x], axis=0), dtype=torch.float32
            )
            u = self.net(state).detach().numpy()
            x = self.dynamics.step_forward(x, u, self.dt)
            q = x[:7]
            current_time += self.dt
            time_to_go -= self.dt

            self.record_a_step(current_time, q)
