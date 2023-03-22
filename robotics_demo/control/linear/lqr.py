import io
import pickle
from typing import Union

import numpy as np
import torch

from robotics_demo.network import AutoTorchDeviceUnpickler, SaveLoad


class LinearQuadraticControllerTorch(SaveLoad):
    def get_data_dict(self):
        return {
            "t0": self.t0,
            "tf": self.tf,
            "dt": self.dt,
            "Kt": self.Kt,
            "k0t": self.k0t,
        }

    def save_impl(self, buffer, *args, **kwargs):
        pickle.dump(self.get_data_dict(), buffer)

    @classmethod
    def load_impl(cls, buff, *args, **kwargs) -> "LinearQuadraticControllerTorch":
        # return cls(**pickle.load(buff))
        return cls(**AutoTorchDeviceUnpickler(buff).load())

    def __init__(
        self,
        t0: float,
        tf: float,
        dt: float,
        Kt: torch.Tensor,
        k0t: torch.Tensor,
    ):
        self.tf = torch.scalar_tensor(tf)
        self.t0 = torch.scalar_tensor(t0)
        self.dt = torch.scalar_tensor(dt)
        self.Kt = Kt
        self.k0t = k0t

    def to(self, *args, **kwargs):
        self.Kt = self.Kt.to(*args, **kwargs)
        self.k0t = self.k0t.to(*args, **kwargs)
        self.tf = self.tf.to(*args, **kwargs)
        self.t0 = self.t0.to(*args, **kwargs)
        self.dt = self.dt.to(*args, **kwargs)
        return self

    def from_time_to_n(self, t: Union[float, torch.Tensor]) -> int:
        t = torch.scalar_tensor(t)
        return torch.round((t - self.t0) / self.dt).long()

    def from_time_to_n_batch(self, t: torch.Tensor) -> torch.LongTensor:
        return torch.round((t - self.t0) / self.dt).long().flatten()

    def get_control(
        self, t: Union[float, torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """x: (n_batch, nx)
        K: (nu, nx), k0: (nu, 1)
        (K*x.T): (nu, n_batch)
        """
        n = self.from_time_to_n(t)
        K, k0 = self.Kt[n], self.k0t[n]
        return (-torch.matmul(K, x.T) - k0).T

    def get_control_batch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The differentiability through t is broken.
        It can only be resolved by recovering the formulae of K(t), k0(t),
         which can be done through taking each polynomial of K, k0. (there are thousands segments)
         t: (n_batch,) x: (n_batch, nx)
        """
        indices = self.from_time_to_n_batch(t)
        Ks, k0s = (
            self.Kt[indices],
            self.k0t[indices],
        )  # shape: (n_batch, nu, nx), (n_batch, nu, 1)
        return (-torch.bmm(Ks, x.unsqueeze(dim=-1)) - k0s).squeeze()

    def convert_time_to_go_to_time(self, time_to_go: torch.Tensor) -> torch.Tensor:
        """Assuming time_to_go=time to go to tf,
        t0 ... t <time to go> tf: time_to_go = tf - t
        """
        return self.tf - time_to_go

    def __eq__(self, other: "LinearQuadraticControllerTorch"):
        if self.t0 == other.t0 and self.tf == other.tf and self.dt == other.dt:
            if torch.norm(self.k0t - other.k0t) <= 1e-15:
                return torch.norm(self.Kt - other.Kt) <= 1e-15
        return False


class ShiftLinearQuadraticControllerAtFixedPointTorch(SaveLoad):
    def __init__(
        self,
        x0: Union[np.ndarray, torch.Tensor],
        u0: Union[np.ndarray, torch.Tensor],
        controller,
    ):
        self.controller = controller
        self.x0 = (
            torch.tensor(x0)
            if not isinstance(x0, torch.Tensor)
            else x0.detach().clone()
        )
        self.u0 = (
            torch.tensor(u0)
            if not isinstance(u0, torch.Tensor)
            else u0.detach().clone()
        )

    def save_impl(self, buffer, *args, **kwargs):
        controller_buffer = io.BytesIO()
        controller_buffer.seek(0)
        self.controller.save(controller_buffer)
        data_dict = {
            "x0": self.x0,
            "u0": self.u0,
            "controller_cls": self.controller.__class__,
            "controller_buff": controller_buffer,
        }
        pickle.dump(data_dict, buffer)

    @classmethod
    def load_impl(cls, buff, *args, **kwargs):
        # data_dict = pickle.load(buff)
        data_dict = AutoTorchDeviceUnpickler(buff).load()
        controller_cls = data_dict.pop("controller_cls")
        controller_buff = data_dict.pop("controller_buff")
        controller_buff.seek(0)
        controller = controller_cls.load(controller_buff)
        return cls(
            **data_dict,
            controller=controller,
        )

    def to(self, *args, **kwargs):
        self.controller.to(*args, **kwargs)
        self.x0 = self.x0.to(*args, **kwargs)
        self.u0 = self.u0.to(*args, **kwargs)
        return self

    def get_control(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """x: (n_batch, nx)
        self.x0: (nx,)
        u_out: (n_batch, nu)
        self.u0: (nu,)
        """
        return self.controller.get_control(t, x - self.x0) + self.u0

    def get_control_batch(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.controller.get_control_batch(t, x - self.x0) + self.u0

    def convert_time_to_go_to_time(self, time_to_go: torch.Tensor) -> torch.Tensor:
        """t0 ... tf: time_to_go = tf - t"""
        return self.controller.convert_time_to_go_to_time(time_to_go)


class AtFixedPointLinearQuadraticControllerTorch(
    ShiftLinearQuadraticControllerAtFixedPointTorch
):
    """This is the LQR controller for using.
     Previous controller assumes expansion at origin while this class also do the shifting.
    If the linear dynamics is also needed (e.g. step_forward, get_value, etc.),
    use LinearDynamicsAtFixedPointWithControllerTorch instead
    """

    @classmethod
    def load_impl(cls, buff, *args, **kwargs):
        # data_dict = pickle.load(buff)
        data_dict = AutoTorchDeviceUnpickler(buff).load()
        # Support for models saved by old api
        if "controller" in data_dict:
            controller = LinearQuadraticControllerTorch(**data_dict.pop("controller"))
            return cls(
                **data_dict,
                controller=controller,
            )
        buff.seek(0)
        return ShiftLinearQuadraticControllerAtFixedPointTorch.load(buff)
