from typing import Dict, Union

import torch
from torch._dynamo.device_interface import (
    DeviceInterface,
    caching_worker_current_devices,
    caching_worker_device_properties,
)
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream

_device_t = Union[torch.device, str, int, None]


class XPUInterface(DeviceInterface):
    device = torch.xpu.device
    Event = torch.xpu.Event
    Stream = torch.xpu.Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["xpu"] = device

        @staticmethod
        def current_device() -> int:
            if "xpu" in caching_worker_current_devices:
                return caching_worker_current_devices["xpu"]
            return torch.xpu.current_device()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "xpu"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = XPUInterface.Worker.current_device()

            if "xpu" not in caching_worker_device_properties:
                device_prop = [
                    torch.xpu.get_device_properties(i)
                    for i in range(torch.xpu.device_count())
                ]
                caching_worker_device_properties["xpu"] = device_prop

            return caching_worker_device_properties["xpu"][device]

    current_device = staticmethod(torch.xpu.current_device)
    set_device = staticmethod(torch.xpu.set_device)
    device_count = staticmethod(torch.xpu.device_count)
    stream = staticmethod(torch.xpu.stream)
    current_stream = staticmethod(torch.xpu.current_stream)
    set_stream = staticmethod(torch.xpu.set_stream)
    _set_stream_by_id = staticmethod(torch.xpu._set_stream_by_id)
    synchronize = staticmethod(torch.xpu.synchronize)
    get_device_properties = staticmethod(torch.xpu.get_device_properties)
    get_raw_stream = staticmethod(get_xpu_stream)

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t = None) -> dict:
        # currently, torch.xpu.get_device_capability returns a dict
        dev_property = torch.xpu.get_device_properties(device)
        print(dev_property)

        dev_capability = {}
        dev_capability["dev_type"] = dev_property.dev_type
        dev_capability["dev_name"] = dev_property.name
        dev_capability["vendor"] = dev_property.vendor
        dev_capability["driver_version"] = dev_property.driver_version
        dev_capability["version"] = dev_property.version
        dev_capability["backend_version"] = dev_property.backend_version
        dev_capability["is_available"] = dev_property.is_available
        dev_capability["global_mem_size"] = dev_property.total_memory
        dev_capability["max_compute_units"] = dev_property.max_compute_units
        dev_capability["gpu_eu_count"] = dev_property.gpu_eu_count
        dev_capability["gpu_subslice_count"] = dev_property.gpu_subslice_count
        dev_capability["max_work_group_size"] = dev_property.max_work_group_size
        dev_capability["max_num_sub_groups"] = dev_property.max_num_sub_groups
        dev_capability["sub_group_sizes"] = dev_property.sub_group_sizes
        dev_capability["support_fp64"] = dev_property.support_fp64

        return dev_capability
