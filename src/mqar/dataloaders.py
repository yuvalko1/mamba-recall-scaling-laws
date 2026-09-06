import inspect

from torch.utils.data import IterableDataset, DataLoader
from typing import Callable

import random

from mqar.generators import generate_mqar_batch


class DynamicMQARBatchDataset(IterableDataset):
    """
    IterableDataset that generates MQAR samples on the fly in batches.
    Batches do not repeat and randomness is reproducible via a global seed.
    """
    def __init__(
        self,
        V: int,
        L: int | None,
        N_facts: int | list[int],
        dataset_size: int,
        batch_size: int,
        seed: int = 0,
        set_special_tokens_to_0: bool = True,
        power_a: float = 1.0,  # for non-uniform distribution set to 0.01 or other values
        random_non_queries: bool = False,

        # our extensions
        num_key_repeats: int = 1,
        num_query_repeats: int = 1,
        reduce_to_AR: bool = False,  # AR instead of MQAR

        include_slices: bool = False,  # unused
):

        self.V = V
        self.L = L
        self.N_facts = N_facts

        self.dataset_size = dataset_size
        self.batch_size = min(batch_size, self.dataset_size)

        self.seed = seed

        self.power_a = power_a
        self.set_special_tokens_to_0 = set_special_tokens_to_0
        self.random_non_queries = random_non_queries

        self.include_slices = include_slices

        self.num_batches = (self.dataset_size + self.batch_size - 1) // self.batch_size

        # our extensions
        self.num_key_repeats = num_key_repeats
        self.num_query_repeats = num_query_repeats
        self.reduce_to_AR = reduce_to_AR

    def __iter__(self):

        # generate one batch of data
        def _generate_batch(n_facts: int | list[int]):
            return generate_mqar_batch(
                V=self.V,
                L=self.L,
                N_facts=n_facts,
                batch_size=self.batch_size,
                seed=random.randint(0, 2 ** 31),
                power_a=self.power_a,
                random_non_queries=self.random_non_queries,
                include_slices=self.include_slices,
                reduce_to_AR=self.reduce_to_AR,
                num_key_repeats=self.num_key_repeats,
                num_query_repeats=self.num_query_repeats,
            )

        for batch_idx in range(self.num_batches):
            batch = _generate_batch(self.N_facts)
            yield batch

    def __len__(self):
        return self.num_batches


def get_mqar_dynamic_dataloader(
    V: int,
    L: int,
    N_facts: int,
    dataset_size: int,
    batch_size: int,
    **kwargs
) -> DataLoader:
    """
    Returns a DataLoader that yields dynamically generated MQAR batches.

    Additional DataLoader keyword arguments (e.g., num_workers) can be passed via dataloader_kwargs.
    Batch outputs are dicts with tensor entries for each field.
    """

    dataset = DynamicMQARBatchDataset(
        V=V,
        L=L,
        N_facts=N_facts,
        dataset_size=dataset_size,
        batch_size=batch_size,
        **kwargs,
    )
    # Use batch_size=None so DataLoader yields the full batch dict
    return DataLoader(dataset, batch_size=None)


def _get_kwargs(function: Callable, config: dict):
    kwargs = {k: config[k] for k in inspect.signature(function).parameters if k in config}
    return kwargs


def get_mqar_dynamic_dataloaders_by_split(dataset_config_by_split: dict[str, dict]) -> dict[str, DataLoader]:
    """
    Load or generate dynamic MQAR DataLoaders from utils.config.
    Returns train/val/test splits of dynamic DataLoaders,
    with sizes determined by train_frac/val_frac/test_frac in config.
    """

    def _build_loader(dataset_config: dict):
        kwargs = _get_kwargs(DynamicMQARBatchDataset.__init__, dataset_config)
        return get_mqar_dynamic_dataloader(**kwargs)

    return {k: _build_loader(dataset_config=config) for k, config in dataset_config_by_split.items()}
