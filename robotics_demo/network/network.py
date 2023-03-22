""" Unifed Models defined here to support unified signatures.
    Network outputs should match the first input of loss functions.
    The second input of loss functions is always from a dataloader which constructed from dataset, i.e. a single tensor.
"""


import io
import logging
import pickle
import warnings
from typing import Iterator

import torch
import torch.nn as nn
from torch.nn import Parameter


from robotics_demo.network.base import (
    convert_array_to_str,
    renamed_load,
    SaveLoad,
    AutoTorchDeviceUnpickler,
    GeneralizedLogisticFunction,
)


class UVNetInDeptCombined(nn.Module, SaveLoad):
    """This class combined two independent networks into one network.
    It outputs the concatenation of the two networks' outputs.
    """

    def __init__(self, u_net, v_net) -> None:
        super().__init__()

        self.u_net = u_net
        self.v_net = v_net

    def to(self, *args, **kwargs):
        self.u_net = self.u_net.to(*args, **kwargs)
        self.v_net = self.v_net.to(*args, **kwargs)
        return self

    def forward(self, x):
        u = self.u_net(x)
        v = self.v_net(x)
        return u, v

    @staticmethod
    def save_to_buff(net):
        net_buff = io.BytesIO()
        net_buff.seek(0)
        if hasattr(net, "save"):
            net.save(net_buff)
        else:
            torch.save(net, net_buff)
        return net_buff

    @staticmethod
    def load_from_buff(buff, cls):
        buff.seek(0)
        if hasattr(cls, "load"):
            return cls.load(buff)
        else:
            return torch.load(
                buff,
                **({"map_location": "cpu"} if not torch.cuda.is_available() else {}),
            )

    def save_impl(self, buffer, *args, **kwargs):
        u_net_buff = self.save_to_buff(self.u_net)
        v_net_buff = self.save_to_buff(self.v_net)

        data_dict = {
            "u_net_buff": u_net_buff,
            "v_net_buff": v_net_buff,
            "u_net_cls": self.u_net.__class__,
            "v_net_cls": self.v_net.__class__,
        }
        pickle.dump(data_dict, buffer)

    @classmethod
    def load_impl(cls, buff, *args, **kwargs):
        data_dict = renamed_load(buff)
        u_net_buff = data_dict["u_net_buff"]
        v_net_buff = data_dict["v_net_buff"]
        u_net_cls = data_dict["u_net_cls"]
        v_net_cls = data_dict["v_net_cls"]
        u_net = cls.load_from_buff(u_net_buff, u_net_cls)
        v_net = cls.load_from_buff(v_net_buff, v_net_cls)
        return cls(u_net, v_net)

    def get_control_network(self):
        return self.u_net

    def get_value_network(self):
        return self.v_net


class EnsembleNet(nn.Module):
    """
    This class gets a list of network, and outputs the mean of their outputs as an ensemble network.
    """

    def __init__(self, net_list) -> None:
        super().__init__()
        self.net_list = nn.ModuleList(net_list)

    def to(self, *args, **kwargs):
        self.net_list = self.net_list.to(*args, **kwargs)
        return self

    def forward(self, x):
        out_list = []
        for net in self.net_list:
            out_list.append(net(x).unsqueeze(-1))
        out_all = torch.cat(out_list, dim=-1)
        return out_all.mean(-1)


class QRNet(nn.Module, SaveLoad):
    def __init__(self, net, x_f, u_f, u_min, u_max, lqr_controller):
        super().__init__()

        logging.info(
            "The QRNet has been constructed with:\n"
            "min %s\n"
            "max %s\n"
            "ref %s" % tuple(map(convert_array_to_str, (u_min, u_max, u_f)))
        )
        self.net: nn.Module = net
        self.x_f = (
            torch.tensor(x_f)
            if not isinstance(x_f, torch.Tensor)
            else x_f.detach().clone()
        )

        self.nx = len(x_f)
        self.lqr_controller = lqr_controller
        self.sigma = GeneralizedLogisticFunction(u_min=u_min, u_max=u_max, u_f=u_f)

    def to(self, *args, **kwargs):
        self.x_f = self.x_f.to(*args, **kwargs)
        self.net.to(*args, **kwargs)
        self.lqr_controller.to(*args, **kwargs)
        self.sigma.to(*args, **kwargs)
        return self

    def save_impl(self, buffer, *args, **kwargs):
        net_buffer = io.BytesIO()
        net_buffer.seek(0)
        torch.save(self.net, net_buffer)

        controller_buffer = io.BytesIO()
        controller_buffer.seek(0)
        self.lqr_controller.save(controller_buffer)

        data_dict = {
            "x_f": self.x_f,
            "u_f": self.sigma.u_f,
            "u_min": self.sigma.u_min,
            "u_max": self.sigma.u_max,
            "net_buffer": net_buffer,
            "controller_buffer": controller_buffer,
            "controller_cls": self.lqr_controller.__class__,
        }
        pickle.dump(data_dict, buffer)

    @classmethod
    def load_impl(cls, buff, lqr_controller_cls=None, *args, **kwargs):
        data_dict = AutoTorchDeviceUnpickler(buff).load()
        net_buffer = data_dict.pop("net_buffer")
        net_buffer.seek(0)
        net = torch.load(
            net_buffer,
            **({"map_location": "cpu"} if not torch.cuda.is_available() else {}),
        )
        if lqr_controller_cls is None:
            assert (
                "controller_cls" in data_dict
            ), "Cannot infer LQR controller class from buffer"
            lqr_controller_cls = data_dict.pop("controller_cls")
        if "controller_cls" in data_dict:
            lqr_controller_cls_buff = data_dict.pop("controller_cls")
            if lqr_controller_cls_buff != lqr_controller_cls:
                warnings.warn(
                    f"LQR controller class in buffer ({lqr_controller_cls_buff}) "
                    f"does not match that in the argument."
                    f"Will use the argument {lqr_controller_cls}"
                )

        controller_buffer = data_dict.pop("controller_buffer")
        controller_buffer.seek(0)
        controller = lqr_controller_cls.load(controller_buffer)
        logging.info(f"Controller of class: {lqr_controller_cls} is being loaded")

        return cls(net, lqr_controller=controller, **data_dict)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.net.parameters(recurse)

    def train(self, mode: bool = True):
        self.net.train(mode)
        return super().train(mode)

    def eval(self):
        self.net.eval()
        return super().eval()

    def forward(self, x: torch.Tensor):
        single_sample = x.ndim == 1
        if single_sample:
            x = x.view(1, -1)
        t, xv = torch.narrow(x, dim=-1, start=0, length=1), torch.narrow(
            x, dim=-1, start=1, length=self.nx
        )
        t = self.lqr_controller.convert_time_to_go_to_time(t)
        lqr_control = self.lqr_controller.get_control_batch(t, xv)  # (n_batch, nu)
        u = (
            lqr_control
            + self.net(x)
            - self.net(torch.cat([x[:, :1], self.x_f.repeat(x.shape[0], 1)], dim=-1))
        )
        uu = self.sigma(u)
        if single_sample:
            return uu.flatten()
        else:
            return uu
