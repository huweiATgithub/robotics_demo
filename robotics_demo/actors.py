import copy

import numpy as np
import pinocchio
from pydrake.geometry import Meshcat

from robotics_demo.robot import Manipulator


class ButtonActor:
    def __init__(self, meshcat: Meshcat, button_name, actor):
        self.meshcat = meshcat
        self.meshcat.AddButton(button_name)
        self.name = button_name
        self.click = 0
        self.actor = actor

    def is_new_click(self):
        click = self.meshcat.GetButtonClicks(self.name)
        if click > self.click:
            self.click += 1
            return True
        return False

    def act(self):
        if self.is_new_click():
            self.actor()
            return True
        return False


class PositionSliderManager:
    def __init__(
        self,
        meshcat: Meshcat,
        manipulator: Manipulator,
        manipulator_context,
        manipulator_pin: pinocchio.Model = None,
        *,
        lower_limits: np.ndarray = None,
        upper_limits: np.ndarray = None,
    ):
        self.meshcat = meshcat
        self.manipulator = manipulator
        self.manipulator_context = manipulator_context
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        if self.lower_limits is None and manipulator_pin is not None:
            self.lower_limits = manipulator_pin.model.lowerPositionLimit
        if self.upper_limits is None and manipulator_pin is not None:
            self.upper_limits = manipulator_pin.model.upperPositionLimit
        self.sliders = self.setup_sliders()

    def setup_sliders(self):
        joints = self.manipulator.get_revolute_joint_indices()
        current_angles = self.manipulator.get_iiwa_position(self.manipulator_context)
        # default_angles = self.manipulator.get_default_joint_angles()
        sliders = {}
        for i, (j, a) in enumerate(zip(joints, current_angles)):
            name = str(j)
            sliders[i] = name
            self.meshcat.AddSlider(
                value=a,
                name=name,
                min=self.lower_limits[i],
                max=self.upper_limits[i],
                step=0.05,
            )
        return sliders

    def remove_sliders(self):
        for s in self.sliders.values():
            self.meshcat.DeleteSlider(s)

    def update_robot(self) -> bool:
        """Return True if robot is updated."""
        old_positions = self.manipulator.get_iiwa_position(
            self.manipulator_context,
        )
        positions = copy.deepcopy(old_positions)
        for i, s in self.sliders.items():
            positions[i] = self.meshcat.GetSliderValue(s)
        if not np.array_equal(old_positions, positions):
            self.manipulator.set_iiwa_positions(self.manipulator_context, positions)
            return True
        return False

    def update_slider(self, q: np.ndarray):
        for i, s in self.sliders.items():
            self.meshcat.SetSliderValue(s, q[i])

    def get_slider_values(self):
        return np.array(
            [self.meshcat.GetSliderValue(s) for i, s in self.sliders.items()]
        )
