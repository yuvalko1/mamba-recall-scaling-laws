import dataclasses

import numpy as np
import torch

import mqar_zoology.associative_recall as zoology


@dataclasses.dataclass
class MqarBatch:
    x_ids: torch.Tensor
    y_true_ids: torch.Tensor
    V: int
    L: int
    N_facts: int
    size: int


# Wrappers

def generate_mqar_batch(
    V: int,
    L: int,
    N_facts: int | list[int],
    batch_size: int,
    seed: int = None,

    # our extensions
    num_key_repeats: int = 1,
    num_query_repeats: int = 1,
    reduce_to_AR: bool = False,  # AR instead of MQAR

    # additional kwargs: (zoology)
    power_a: float = 1.0,  # for non-uniform distribution set to 0.01 or other values
    random_non_queries: bool = False,
    include_slices: bool = False,  # unused

) -> MqarBatch:


    rng = np.random.RandomState(seed)

    pseudorandom_seed = rng.randint(2 ** 31)

    if reduce_to_AR:
        L = int(4 * N_facts)  # minimum possible for zoology implementation; temporary

    data = zoology.multiquery_ar(
        vocab_size=V,
        num_examples=batch_size,
        input_seq_len=L,
        num_kv_pairs=N_facts,
        power_a=power_a,
        random_non_queries=random_non_queries,
        num_key_repeats=num_key_repeats,
        num_query_repeats=num_query_repeats,
        seed=pseudorandom_seed,
        include_slices=include_slices,
    )

    x = data.inputs
    y = data.labels

    if reduce_to_AR:
        L = int(2 * N_facts + 1)  # actual length for AR (fact pairs + a single query)
        x = x[:, :L]
        y = y[:, :L]

    batch = MqarBatch(
        x_ids=x,
        y_true_ids=y,
        V=V,
        L=L,
        N_facts=N_facts,
        size=batch_size,
    )

    return batch
