import numpy as np
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.controllers import InverseDynamicsController

from robotics_demo.configs import ModelDefinitionConfig
from robotics_demo.robot import construct_model


class InverseDynamicsPIDController:
    def __init__(self, model_conf: ModelDefinitionConfig, kp, kd, ki):
        plant = MultibodyPlant(0.0)
        construct_model(model_conf=model_conf, plant=plant)
        plant.Finalize()
        self.controller = InverseDynamicsController(
            plant, kp=kp, kd=kd, ki=ki, has_reference_acceleration=False
        )
        self.context = self.controller.CreateDefaultContext()
        self.input_current_state = self.controller.get_input_port_estimated_state()
        self.input_desired_state = self.controller.get_input_port_desired_state()
        self.control_output = self.controller.get_output_port()

    def get_control(
        self, current_state: np.ndarray, desired_state: np.ndarray
    ) -> np.ndarray:
        self.input_current_state.FixValue(self.context, current_state)
        self.input_desired_state.FixValue(self.context, desired_state)
        return self.control_output.Eval(self.context)
