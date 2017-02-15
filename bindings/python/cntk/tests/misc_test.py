# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import cntk
import pytest
from cntk.device import *

cntk_py.always_allow_setting_default_device()

from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(1)

def is_locked(device):
    future = pool.submit(device.is_locked)
    return future.result()

def test_callstack1():
    with pytest.raises(ValueError) as excinfo:
        cntk.device.gpu(99999)
    assert '[CALL STACK]' in str(excinfo.value)

def test_callstack2():
    with pytest.raises(ValueError) as excinfo:
        cntk.io.MinibatchSource(cntk.io.CTFDeserializer("", streams={}))
    assert '[CALL STACK]' in str(excinfo.value)

def test_cpu_and_gpu_devices():
    device = cpu()
    assert DeviceKind(device.type()) == DeviceKind.CPU
    assert device.id() == 0
    for i in range(len(all_devices()) - 1):
        device = gpu(i)
        assert DeviceKind(device.type()) == DeviceKind.GPU
        assert device.id() == i
        

def test_all_devices():
    assert len(all_devices()) > 0
    assert cpu() in all_devices()
    if (len(all_devices()) > 1):
        assert gpu(0) in all_devices()

def test_gpu_properties():
    for device in all_devices():
        if (DeviceKind(device.type()) != DeviceKind.GPU):
            continue
        props =  get_gpu_properties(device)
        assert props.device_id == 0
        assert props.cuda_cores > 0
        assert props.total_memory > 0
        assert props.version_major > 0

def test_use_default_device():
    device = use_default_device()
    if (DeviceKind(device.type()) != DeviceKind.GPU):
        assert not is_locked(device)
    else:
        assert is_locked(device)
    
def test_set_cpu_as_default_device():
    device = cpu()
    assert not is_locked(device)
    assert not try_set_default_device(device, True)
    assert not is_locked(device)
    assert try_set_default_device(device, False)
    assert not is_locked(device)
    assert device == use_default_device()

def test_set_cpu_as_default_device():
  if len(all_devices()) == 1: 
      return;
  # this will realease any previous device lock
  try_set_default_device(cpu(), False)
  for i in range(len(all_devices()) - 1):
    device = gpu(i)
    assert try_set_default_device(device, False)
    assert not is_locked(device)
    assert device == use_default_device()
    if not device.is_locked():
        assert not is_locked(device)
        assert try_set_default_device(device, True)
        assert device == use_default_device()
        assert is_locked(device)

def test_set_excluded_devices():
  if len(all_devices()) == 1: 
      return;
  assert try_set_default_device(cpu(), False)
  assert try_set_default_device(gpu(0), False)
  set_excluded_devices([cpu()])
  assert not try_set_default_device(cpu(), False)
  set_excluded_devices([])
  assert try_set_default_device(cpu(), False)