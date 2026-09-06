import os
from collections import OrderedDict
import torch

# Environment flag
_enable_env = 'ENABLE_RECORDING'

# Global state
_global_seq = 0
_global_values = {}
_global_seq_map = {}

# informational prints (enabled/cleared/collected); scripts that want quiet
# output set recorder.VERBOSE = False (e.g. via run_model_on_single_sequence)
VERBOSE = True


def enable_recording():
    """Turn on recording globally."""
    os.environ[_enable_env] = '1'
    if VERBOSE:
        print("Recording enabled globally.\n")


def disable_recording():
    """Turn off recording globally."""
    os.environ.pop(_enable_env, None)
    if VERBOSE:
        print("Recording disabled globally.\n")


def empty():
    """Reset sequence and clear all records."""
    global _global_seq, _global_values, _global_seq_map
    _global_seq = 0
    _global_values.clear()
    _global_seq_map.clear()

    if VERBOSE:
        print(
            "All recordings cleared. \n")


def get_global_recordings():
    """Return a deep copy of all recorded tensors."""
    copied_global_values = {k: v for k, v in _global_values.items()}
    return copied_global_values


def get_global_recordings_order():
    """Return a deep copy of the recording order map."""
    copied_global_seq_map = {k: v for k, v in _global_seq_map.items()}
    return copied_global_seq_map


def generic_record(uid, verbose=False, **kwargs):
    """
    Record named tensors under a unique identifier `uid`.
    Unique key is '{uid}__{var_name}', with an incremental sequence.
    """
    if os.environ.get(_enable_env) != '1':
        return

    global _global_seq, _global_values, _global_seq_map
    for var_name, tensor in kwargs.items():
        if tensor is None:
            continue

        seq = _global_seq
        key = f"{uid}__{var_name}"

        if key not in _global_values:
            _global_seq_map[key] = seq
            _global_seq += 1
            _global_values[key] = []

        # record a clone of the tensor
        recorded = tensor.clone()
        _global_values[key].append(recorded)

        count = len(_global_values[key])

        if verbose:
            print(f"{_global_seq_map[key]}: '{key}' recorded ({count} items, shape = {tuple(recorded.shape)})")


def record(parent_object, **kwargs):
    """
    Convenience wrapper: use module instance as uid.

    Example:
        record(self, hidden_states=hidden, outputs=out)
    """

    if isinstance(parent_object, torch.nn.Module):
        uid = hex(id(parent_object))
    elif isinstance(parent_object, str):
        uid = parent_object
    else:
        uid = str(parent_object)

    generic_record(uid, **kwargs)


def collect_model_recordings(root_module, use_long_name=False, stack_all=False, verbose=False):
    """
    Traverse all submodules of `root_module` and merge together
    any global recordings, renaming each key to include its full module path under 'model'.

    Args:
        root_module (nn.Module): the top-level model
        use_long_name (bool): include index, type, and ptr in key
        stack_all (bool): if True, stack all recorded tensors; otherwise take last only

    Returns:
        recorded_tensors (OrderedDict): mapping of new key -> tensor
        recorded_tensors_order (OrderedDict): mapping of sequence index -> new key
    """
    recorded_tensors = OrderedDict()
    recorded_tensors_order = OrderedDict()

    recordings = get_global_recordings()
    recordings_order = get_global_recordings_order()

    for unique_key, val_list in recordings.items():
        id_str, var = unique_key.split('__', 1)
        idx = recordings_order[unique_key]

        # default (to be overridden):
        module_full_path = f"model.{id_str}"
        model_type = ""

        # find module path by matching id
        for module_path, module in root_module.named_modules():
            if hex(id(module)) == id_str:
                module_full_path = f"model.{module_path}" if module_path else "model"
                model_type = type(module).__name__
                break

        short_name = f"{module_full_path}__{var}"
        long_name = f"{idx}__{module_full_path}__{var}"
        key = long_name if use_long_name else short_name

        # select tensor
        if stack_all:
            tensor = torch.stack(val_list)
        else:
            tensor = val_list[-1] if val_list else torch.tensor()
        recorded_tensors[key] = tensor.detach().cpu()
        recorded_tensors_order[idx] = key

        if verbose:
            print(f"{key}: {recorded_tensors[key].numpy().shape}")

    if VERBOSE:
        print("\nAll recordings collected.\n")

    return recorded_tensors, recorded_tensors_order
