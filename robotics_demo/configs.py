import dataclasses
import pathlib
from typing import Tuple

import numpy as np

PKG_PATH = pathlib.Path(__file__).parent
PROJECT_PATH = PKG_PATH.parent
RESOURCES_PATH = PROJECT_PATH.joinpath("resources")
MODEL_PATH = RESOURCES_PATH.joinpath("models")

Q0 = np.array(
    [
        1.67994461,
        1.25010269,
        2.4427646,
        -1.26689386,
        -0.97780785,
        1.1235575,
        -1.35749704,
    ]
)
Q_GOAL = np.array(
    [
        2.77363793,
        0.58416175,
        1.54129758,
        -1.702825,
        -2.1665327,
        0.08465574,
        -2.57643323,
    ]
)


@dataclasses.dataclass
class WeldInfo:
    child: str
    xyz_rpy: Tuple[float, float, float, float, float, float] = None
    _parent: str = None  # None means WorldBody

    @property
    def parent(self):
        if self._parent is not None:
            raise ValueError(
                "Welding to frame other than world frame is not supported."
            )
        return self._parent


@dataclasses.dataclass
class ModelDefinitionConfig:
    # the path to model description file
    model_file_path: str
    # this is used by Pinocchio: it will search package
    # (i.e. folder whose name is the same as package name) in this directory
    dir_containing_pkg: str
    # To load this model in Drake, one should configure the Parser as:
    # parser.package_map().Add(model_conf.pkg_name, model_conf.pkg_dir)
    pkg_dir: str  # the path to the package, used by Drake
    pkg_name: str  # This is the package name in model description file when referring to meshes
    ee_name: str
    warn_msg: str = None
    weld_info: WeldInfo = None


IIWA14 = ModelDefinitionConfig(
    str(MODEL_PATH.joinpath("iiwa14", "urdf", "iiwa14_no_collision.urdf")),
    str(MODEL_PATH),
    pkg_dir=str(MODEL_PATH.joinpath("iiwa14")),
    pkg_name="iiwa14",
    ee_name="iiwa_link_ee_kuka",
    weld_info=WeldInfo(
        "base",
    ),
)
IIWA14_ALPHA = ModelDefinitionConfig(
    str(MODEL_PATH.joinpath("iiwa14", "urdf", "iiwa14_no_collision_alpha.urdf")),
    str(MODEL_PATH),
    pkg_dir=str(MODEL_PATH.joinpath("iiwa14")),
    pkg_name="iiwa14",
    ee_name="iiwa_link_ee_kuka",
    weld_info=WeldInfo(
        "base",
    ),
)
