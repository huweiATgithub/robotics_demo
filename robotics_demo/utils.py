import numpy as np
from pydrake.geometry import Meshcat


def setup_drake_meshcat_camera(meshcat_drake: Meshcat):
    from pydrake.math import RigidTransform, RotationMatrix
    from pydrake.common.eigen_geometry import AngleAxis

    trans_rigid = RigidTransform([0.0, 1.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    rot_rigid = RigidTransform(
        RotationMatrix(AngleAxis(-3.14 / 2, axis / np.linalg.norm(axis)))
    )
    meshcat_drake.SetTransform(
        "/Cameras/default",
        rot_rigid @ trans_rigid,
    )
    meshcat_drake.SetProperty("/Cameras/default/rotated/<object>", "zoom", 1.5)


def get_ip_addr() -> str:
    import requests

    response = requests.get("https://ifconfig.me/ip/", timeout=10)
    if response.status_code != 200:
        raise requests.exceptions.HTTPError(
            f"{response.status_code} - {response.reason}"
        )
    return response.text
