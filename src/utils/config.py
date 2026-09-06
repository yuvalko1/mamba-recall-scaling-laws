from typing import Any

from mqar import MqarDimensions


available_mamba_architectures = []
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig
    available_mamba_architectures.extend(['mamba_ssm'])
except Exception:
    pass
from mamba_tiny.model import ModelArgs, Mamba as MambaTiny
available_mamba_architectures.extend(['mamba_tiny'])





def get_model_from_config(
        dims: MqarDimensions,
        model_config: dict[str, Any],
):
    # extract sub-configs
    model_class = model_config['class']
    model_variant = model_config['variant']  # only for debug/naming
    model_args = model_config['args']
    model_kwargs = model_config.get('kwargs', {})


    if model_class not in available_mamba_architectures:
        raise ValueError(
            f"{model_class = } is currently unavailable; "
            f"choose from: {available_mamba_architectures}\n"
        )

    # set model dimensions, and init weights if requested
    if model_class == 'mamba_ssm':

        # set dims
        model_args['vocab_size'] = dims.V
        model_args['d_model'] = dims.D
        model_args['ssm_cfg']['d_state'] = dims.N
        model_args['n_layer'] = dims.Lambda

        # instantiate
        model = MambaLMHeadModel(config=MambaConfig(**model_args), **model_kwargs)

    elif model_class == 'mamba_tiny':

        # set dims
        model_args['vocab_size'] = dims.V
        model_args['d_model'] = dims.D
        model_args['d_state'] = dims.N
        model_args['n_layers'] = dims.Lambda
        model_args['n_heads'] = dims.M

        # instantiate
        args = ModelArgs(**model_args)
        model = MambaTiny(args=args, **model_kwargs)

    else:
        raise ValueError(f"unknown {model_class = }")


    return model


def get_dataset_config_by_split_from_dataset_config(dataset_config: str, seed: int = None):

    splits = ["train", "val", "test"]
    dataset_config_by_split: dict[str, dict[str, Any]] = {}

    for i, k in enumerate(splits):

        cfg = {}

        cfg.update(dataset_config.get("kwargs", {}))

        cfg["V"] = dataset_config["V"]
        cfg["L"] = dataset_config["L"]
        cfg["N_facts"] = dataset_config["N_facts"]
        cfg["batch_size"] = dataset_config["batch_size"]
        cfg["dataset_size"] = dataset_config["split_size"][k]

        # set a different seed for each split
        cfg["seed"] = (seed + i) if seed is not None else None

        dataset_config_by_split[k] = cfg

    return dataset_config_by_split
