import io
import abc
import pickle
import torch
import numpy as np
from typing import Union


def convert_array_to_str(arr: Union[np.ndarray, torch.Tensor], precision=3):
    arr = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else arr
    return np.array_str(arr, precision=precision, suppress_small=True)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


class SaveLoad:
    """This class defines save and load which handles both file path and file buffer.
    It leaves the detailed implementation to save_impl and load_impl.
    """

    @abc.abstractmethod
    def save_impl(self, buffer, *args, **kwargs):
        pass

    def save(self, save_path_or_buffer, *args, **kwargs) -> None:
        if hasattr(save_path_or_buffer, "write"):
            self.save_impl(save_path_or_buffer, *args, **kwargs)
        else:
            with open(save_path_or_buffer, "wb") as f:
                self.save_impl(f, *args, **kwargs)

    @classmethod
    @abc.abstractmethod
    def load_impl(cls, buff, *args, **kwargs):
        pass

    @classmethod
    def load(cls, load_path_or_buffer, *args, **kwargs):
        if hasattr(load_path_or_buffer, "read") and hasattr(
            load_path_or_buffer, "readline"
        ):
            return cls.load_impl(load_path_or_buffer, *args, **kwargs)
        else:
            with open(load_path_or_buffer, "rb") as f:
                return cls.load_impl(f, *args, **kwargs)


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module

        if module.startswith("manipulator"):
            if module.endswith("qr_net"):
                renamed_module = "robotics_demo.network"
            else:
                renamed_module = module.replace("manipulator", "robotics_demo")

        return super(RenameUnpickler, self).find_class(renamed_module, name)


class AutoTorchDeviceUnpickler(RenameUnpickler):
    def find_class(self, __module_name: str, __global_name: str):
        if not torch.cuda.is_available():
            if __module_name == "torch.storage" and __global_name == "_load_from_bytes":
                return lambda b: torch.load(
                    io.BytesIO(b), map_location=torch.device("cpu")
                )
        return super().find_class(__module_name, __global_name)


class GeneralizedLogisticFunction:
    """
    This function computes:
        u_min + (u_max - u_min)/(1+c1 exp(-c2(u-uf)))
    It maps [-inf, inf] to [u_min, u_max].
    c1, c2 is chosen such that:
        - It equals to uf at uf.
        - It has derivative equals to 1 at uf.

    """

    def __init__(self, u_min: torch.Tensor, u_max: torch.Tensor, u_f: torch.Tensor):
        self.u_min = (
            torch.tensor(u_min)
            if not isinstance(u_min, torch.Tensor)
            else u_min.detach().clone()
        )
        self.u_max = (
            torch.tensor(u_max)
            if not isinstance(u_max, torch.Tensor)
            else u_max.detach().clone()
        )
        self.u_f = (
            torch.tensor(u_f)
            if not isinstance(u_f, torch.Tensor)
            else u_f.detach().clone()
        )
        self.c1 = (self.u_max - self.u_f) / (self.u_f - self.u_min)
        self.c2 = (self.u_max - self.u_min) / (
            (self.u_max - self.u_f) * (self.u_f - self.u_min)
        )

    def to(self, *args, **kwargs):
        self.u_min = self.u_min.to(*args, **kwargs)
        self.u_max = self.u_max.to(*args, **kwargs)
        self.u_f = self.u_f.to(*args, **kwargs)
        self.c1 = self.c1.to(*args, **kwargs)
        self.c2 = self.c2.to(*args, **kwargs)
        return self

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.u_min + (self.u_max - self.u_min) / (
            1 + self.c1 * torch.exp(-self.c2 * (inputs - self.u_f))
        )
