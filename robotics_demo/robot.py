import abc
import dataclasses
from typing import Optional, Union

import numpy as np
import pinocchio as pin
from pydrake.geometry import SceneGraph
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex, Frame
from pydrake.systems.framework import Diagram, DiagramBuilder, OutputPort
from pydrake.systems.primitives import Demultiplexer

from robotics_demo.configs import ModelDefinitionConfig


@dataclasses.dataclass
class ModelInfo:
    model_path: str
    model_instance: ModelInstanceIndex
    child_frame: Frame
    X_PC: RigidTransform
    do_weld: bool
    name: str


# noinspection PyArgumentList
def construct_model(
    model_conf: Union[str, ModelDefinitionConfig],
    plant: MultibodyPlant,
    name: str = "iiwa",
) -> ModelInfo:
    parser = Parser(plant)
    parser.package_map().Add(model_conf.pkg_name, model_conf.pkg_dir)
    model = parser.AddModelFromFile(model_conf.model_file_path, name)
    if model_conf.weld_info is not None:
        weld_info = model_conf.weld_info
        child_frame = plant.GetFrameByName(weld_info.child, model)
        parent_frame = (
            plant.GetFrameByName(weld_info.parent, model)
            if weld_info.parent
            else plant.world_frame()
        )
        X_PC = RigidTransform.Identity()
        if weld_info.xyz_rpy is not None:
            xyz_rpy = weld_info.xyz_rpy
            rpy = RollPitchYaw(*xyz_rpy[3:])
            X_PC = RigidTransform(rpy=rpy, p=xyz_rpy[:3])
        plant.WeldFrames(parent_frame, child_frame, X_PC)
        iiwa_model = ModelInfo(
            model_conf.model_file_path,
            model,
            child_frame,
            X_PC,
            do_weld=True,
            name=name,
        )
    else:
        # noinspection PyTypeChecker
        iiwa_model = ModelInfo(
            model_conf.model_file_path, model, None, None, False, name=name
        )
    return iiwa_model


class Plant(Diagram):
    """Plant is a class that wraps a MultibodyPlant and a SceneGraph,
    and expose several useful methods to manipulate them.
    Set with_geometry to False for dynamics-only simulation: SceneGraph will not be created.
    """

    plant_name = "plant"
    scene_graph_name = "scene_graph"
    geometry_output_port_name = "query_object"

    def __init__(self, time_step: float, with_geometry: bool = True):
        super().__init__()
        self.plant = MultibodyPlant(time_step)
        self.with_geometry = with_geometry
        if with_geometry:
            self.scene_graph = SceneGraph()
            self.plant.RegisterAsSourceForSceneGraph(self.scene_graph)
            self.plant.set_name(self.plant_name)
            self.scene_graph.set_name(self.scene_graph_name)

    @abc.abstractmethod
    def finalize(self):
        pass

    def _get_builder_with_plant(self):
        builder = DiagramBuilder()
        builder.AddSystem(self.plant)
        return builder

    def _connect_scene_graph(self, builder):
        assert self.with_geometry, "Must enable geometry to connect it."
        builder.AddSystem(self.scene_graph)
        builder.Connect(
            self.plant.get_geometry_poses_output_port(),
            self.scene_graph.get_source_pose_port(self.plant.get_source_id()),
        )
        builder.Connect(
            self.scene_graph.get_query_output_port(),
            self.plant.get_geometry_query_input_port(),
        )
        return builder

    def get_builder(self):
        builder = self._get_builder_with_plant()
        if self.with_geometry:
            builder = self._connect_scene_graph(builder)
            builder.ExportOutput(
                self.scene_graph.get_query_output_port(), self.geometry_output_port_name
            )
        return builder

    def get_output_port_geometry_query(self):
        return self.GetOutputPort(self.geometry_output_port_name)


DEFAULT_q0 = np.array([-1.57, 0.1, 0.0, -1.2, 0.0, 1.6, 0])


class Manipulator(Plant):
    """ManipulatorOnly means the plant only contains the manipulator.
    If one needs to attach a controller, use Manipulator instead.
    """

    def __init__(
        self,
        time_step: float,
        q0=None,
        with_geometry: bool = True,
        keep_default_angle: bool = False,
    ):
        super().__init__(time_step, with_geometry=with_geometry)

        self.set_name("manipulator")

        self.iiwa_model: Optional[ModelInfo] = None
        if keep_default_angle:
            assert q0 is None
            self._q0 = None
        else:
            self._q0 = q0 if q0 is not None else DEFAULT_q0

    def add_iiwa(self, model_def):
        self.iiwa_model = construct_model(model_def, self.plant)

    def add_static_iiwa(self, model_def):
        return construct_model(model_def, self.plant, name="static_iiwa")

    def finalize(self):
        self.plant.Finalize()
        builder = self.get_builder()
        self._set_default_joint_angles()
        iiwa_state_output_port = self.plant.get_state_output_port(
            self.iiwa_model.model_instance
        )
        builder.ExportOutput(
            iiwa_state_output_port,
            "iiwa_state",
        )
        builder.ExportInput(
            self.plant.get_actuation_input_port(self.iiwa_model.model_instance),
            "iiwa_actuation",
        )
        builder.BuildInto(self)

    def get_revolute_joint_indices(self):
        iiwa_joint_indices = self.plant.GetJointIndices(self.iiwa_model.model_instance)
        revolute_joint_indices = []
        for j in iiwa_joint_indices:
            joint = self.plant.get_joint(j)
            if joint.type_name() == "revolute":
                revolute_joint_indices.append(j)
                continue
            if joint.num_positions() != 0:
                raise ValueError(f"There are joints {str(joint)} which not supported.")
        return revolute_joint_indices

    def get_num_iiwa_positions(self):
        return self.plant.num_positions(self.iiwa_model.model_instance)

    def get_num_iiwa_states(self):
        return self.plant.num_multibody_states(self.iiwa_model.model_instance)

    def get_num_iiwa_actuated_dofs(self):
        return self.plant.num_actuated_dofs(self.iiwa_model.model_instance)

    @property
    def num_states(self):
        return self.get_num_iiwa_states()

    @property
    def num_actuators(self):
        return self.get_num_iiwa_actuated_dofs()

    @property
    def num_positions(self):
        return self.get_num_iiwa_positions()

    def _set_default_joint_angles(self):
        q0_iiwa = self._q0
        if q0_iiwa is None:
            return
        iiwa_joint_indices_revolute = self.get_revolute_joint_indices()
        assert len(iiwa_joint_indices_revolute) == len(q0_iiwa), (
            f"Number of default angles ({len(q0_iiwa)}) "
            f"does NOT match number of revolute joint ({len(iiwa_joint_indices_revolute)})"
        )
        for index, angle in zip(iiwa_joint_indices_revolute, q0_iiwa):
            joint = self.plant.get_mutable_joint(index)
            if joint:
                joint.set_default_angle(angle)

    def get_default_joint_angles(self):
        angles = []
        for j_id in self.get_revolute_joint_indices():
            joint = self.plant.get_mutable_joint(j_id)
            angles.append(joint.get_default_angle())
        return np.array(angles)

    def _add_positions_velocities_output(self, builder, state_output_port: OutputPort):
        demux = builder.AddSystem(Demultiplexer(self.num_states, self.num_positions))
        builder.Connect(
            state_output_port,
            demux.get_input_port(0),
        )
        demux.set_name("Split states")

        builder.ExportOutput(demux.get_output_port(0), "iiwa_position")
        builder.ExportOutput(demux.get_output_port(1), "iiwa_velocity")
        return builder

    def print_body_info(self):
        body_indexes = self.plant.GetBodyIndices(self.iiwa_model.model_instance)
        for index in body_indexes:
            body = self.plant.get_body(index)
            print(body)

    def set_iiwa_states(self, my_context, q_v):
        plant_context = self.GetMutableSubsystemContext(self.plant, my_context)
        self.plant.SetPositionsAndVelocities(
            plant_context, self.iiwa_model.model_instance, q_v
        )

    def set_iiwa_positions(self, my_context, q):
        self.set_iiwa_positions_on_model(my_context, q, self.iiwa_model.model_instance)

    def set_iiwa_positions_on_model(self, my_context, q, model_instance):
        plant_context = self.GetMutableSubsystemContext(self.plant, my_context)
        self.plant.SetPositions(plant_context, model_instance, q)

    def get_iiwa_position(self, my_context):
        plant_context = self.GetSubsystemContext(self.plant, my_context)
        return self.plant.GetPositions(plant_context, self.iiwa_model.model_instance)

    def get_multibody_plant(self):
        return self.plant

    def get_iiwa_ee_default_pose(self, ee_body_name):
        temp_context = self.CreateDefaultContext()
        temp_plant_context = self.plant.GetMyContextFromRoot(temp_context)
        return self.plant.EvalBodyPoseInWorld(
            temp_plant_context, self.plant.GetBodyByName(ee_body_name)
        )

    def get_output_port_iiwa_position(self):
        return self.GetOutputPort("iiwa_position")

    def get_output_port_iiwa_velocity(self):
        return self.GetOutputPort("iiwa_velocity")

    def get_input_port_actuation(self):
        return self.GetInputPort("iiwa_actuation")


class ManipulatorDynamics:
    def __init__(self, model_def: ModelDefinitionConfig):
        self.model = pin.buildModelsFromUrdf(
            model_def.model_file_path,
            package_dirs=[model_def.dir_containing_pkg],
        )[0]
        self.data = pin.createDatas(self.model)[0]
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nu = self.nv

    def get_control_gravity_compensation(self, x: np.ndarray) -> np.ndarray:
        u_g = pin.computeGeneralizedGravity(
            self.model,
            self.data,
            x[: self.nq],
        )
        return u_g

    def step_forward(self, x, u, dt) -> np.ndarray:
        q, v = x[: self.nq], x[self.nq :]
        a = pin.aba(self.model, self.data, q, v, u)
        v_next = v + a * dt
        q_next = q + v_next * dt
        return np.r_[q_next, v_next]
