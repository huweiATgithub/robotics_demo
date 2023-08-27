import abc
import time
from typing import Optional

import numpy as np
import requests
from pydrake.geometry import Meshcat, MeshcatVisualizer
from pydrake.systems.framework import DiagramBuilder

from robotics_demo.actors import PositionSliderManager, ButtonActor
from robotics_demo.configs import ModelDefinitionConfig
from robotics_demo.robot import Manipulator, ManipulatorDynamics
from robotics_demo.utils import get_ip_addr, setup_drake_meshcat_camera


class StageController:
    def __init__(self, enter, loop, exi):
        self._enter = enter
        self._loop = loop
        self._exit = exi

        self.inited = False
        self.should_end = False

    def enter(self):
        if self.inited:
            return
        self.should_end = False
        for e in self._enter:
            e()
        self.inited = True

    def loop(self) -> bool:
        final_loop_result = False
        for lo in self._loop:
            final_loop_result = lo()
        self.should_end = final_loop_result
        return self.should_end

    def exit(self):
        if not self.should_end:
            return
        self.inited = False
        next_stage = None
        for ex in self._exit:
            next_stage = ex()
        return next_stage


class Demo:
    def __init__(
        self,
        robot: ModelDefinitionConfig,
        lower_limits,
        upper_limits,
        dt: float,
        target_robot: ModelDefinitionConfig = None,
        target_q: np.ndarray = None,
        port: int = 7000,
    ):
        builder = DiagramBuilder()
        self.manipulator: Manipulator = builder.AddSystem(
            Manipulator(0.0),
        )
        self.manipulator.add_iiwa(robot)
        if target_robot:
            instance = self.manipulator.add_static_iiwa(target_robot)
        self.manipulator.finalize()

        self.meshcat = Meshcat(port)
        try:
            ip_addr = get_ip_addr()
            print(f"Meshcat listening for connections at: http://{ip_addr}:{port}")
        except requests.exceptions.HTTPError:
            print(f"Cannot determine IP address. Please lookout your IP address.")
        setup_drake_meshcat_camera(
            self.meshcat,
        )

        # noinspection PyTypeChecker
        self.viz = MeshcatVisualizer.AddToBuilder(
            builder,
            self.manipulator.get_output_port_geometry_query(),
            self.meshcat,
        )
        self.viz.set_name("meshcat_visualizer")

        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.manipulator_context = self.manipulator.GetMyContextFromRoot(self.context)

        if target_q is not None and target_robot:
            # noinspection PyUnboundLocalVariable
            self.manipulator.set_iiwa_positions_on_model(
                self.manipulator_context, target_q, instance.model_instance
            )

        self.diagram.ForcedPublish(self.context)

        self.sliders: Optional[PositionSliderManager] = None
        self.release_button = None
        self.reset_button = None
        self.stop_button = ButtonActor(self.meshcat, "Stop", actor=None)

        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

        self.drag_stage = StageController(
            enter=[
                self.add_slider,
                self.add_release_button,
            ],
            loop=[self.slider_loop, self.release_loop],
            exi=[
                self.remove_release_button,
                self.remove_slider,
                lambda: self.play_stage,
            ],
        )
        self.play_stage = StageController(
            enter=[self.add_reset_button, self.viz.PublishRecording],
            loop=[self.reset_loop],
            exi=[
                # This will only delete the recording buffer, but not the recordings in frontend.
                self.viz.DeleteRecording,
                # Let us Publish an empty recording to clear the frontend. self.meshcat.Delete works only for objects
                self.viz.PublishRecording,
                self.remove_reset_button,
                lambda: self.drag_stage,
            ],
        )

        self.current_stage = self.drag_stage
        self.dynamics = ManipulatorDynamics(robot)
        self.dt = dt

    def add_slider(self):
        self.sliders = PositionSliderManager(
            self.meshcat,
            self.manipulator,
            self.manipulator_context,
            lower_limits=self.lower_limits,
            upper_limits=self.upper_limits,
        )

    def add_release_button(self):
        self.release_button = ButtonActor(
            self.meshcat, "Release", actor=self.apply_control_and_record
        )

    def remove_release_button(self):
        self.meshcat.DeleteButton(self.release_button.name)
        self.release_button = None

    def add_reset_button(self):
        self.reset_button = ButtonActor(self.meshcat, "Reset", actor=self.reset)

    def remove_reset_button(self):
        self.meshcat.DeleteButton(self.reset_button.name)
        self.reset_button = None

    def remove_slider(self):
        self.sliders.remove_sliders()
        self.sliders = None

    def apply_control_and_record(self):
        self.viz.DeleteRecording()
        self.viz.StartRecording()
        q = self.manipulator.get_iiwa_position(self.manipulator_context)
        x = np.r_[q, np.zeros_like(q)]
        self.apply_control(x)
        self.viz.StopRecording()

    def record_a_step(self, t, q):
        self.manipulator.set_iiwa_positions(self.manipulator_context, q)
        self.context.SetTime(t)
        self.diagram.ForcedPublish(self.context)

    @abc.abstractmethod
    def apply_control(self, x: np.ndarray):
        pass

    def reset(self):
        pass

    def slider_loop(self) -> bool:
        is_updated = self.sliders.update_robot()
        if is_updated:
            self.diagram.ForcedPublish(self.context)
        return is_updated

    def release_loop(self) -> bool:
        return self.release_button.act()

    def reset_loop(self) -> bool:
        return self.reset_button.act()

    def run(self):
        # In a stage, it should only respond to that stage (also only those resources are available)
        while not self.stop_button.is_new_click():
            self.current_stage.enter()
            if self.current_stage.loop():
                self.current_stage = self.current_stage.exit()

            time.sleep(0.1)
