import torch

from mqar.dataloaders import DynamicMQARBatchDataset, get_mqar_dynamic_dataloaders_by_split
from utils.common import set_seed
from utils.config import get_dataset_config_by_split_from_dataset_config


def _dataset_config(train_size=64, val_size=32, test_size=32, batch_size=8):
    return {
        'V': 64,
        'L': 32,
        'N_facts': 4,
        'kwargs': {
            'power_a': 1.0,
            'random_non_queries': True,
            'reduce_to_AR': False,
            'num_key_repeats': 1,
            'num_query_repeats': 1,
        },
        'batch_size': batch_size,
        'split_size': {'train': train_size, 'val': val_size, 'test': test_size},
    }


def test_dataset_len_is_ceil_of_size_over_batch():
    dataset = DynamicMQARBatchDataset(V=64, L=32, N_facts=4, dataset_size=100, batch_size=8)
    assert len(dataset) == 13  # ceil(100 / 8)
    # batch_size is clamped to the dataset size
    tiny = DynamicMQARBatchDataset(V=64, L=32, N_facts=4, dataset_size=4, batch_size=8)
    assert tiny.batch_size == 4
    assert len(tiny) == 1


def test_iteration_yields_len_batches_of_expected_shape():
    dataset = DynamicMQARBatchDataset(V=64, L=32, N_facts=4, dataset_size=24, batch_size=8)
    batches = list(dataset)
    assert len(batches) == len(dataset) == 3
    for batch in batches:
        assert batch.x_ids.shape == (8, 32)


def test_reproducible_via_global_seed():
    # per-batch seeds come from the *global* `random` module (the stored dataset
    # seed is not used for batch seeding), so set_seed governs reproducibility
    dataset = DynamicMQARBatchDataset(V=64, L=32, N_facts=4, dataset_size=16, batch_size=8)
    set_seed(0)
    first = [batch.x_ids.clone() for batch in dataset]
    set_seed(0)
    second = [batch.x_ids.clone() for batch in dataset]
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_split_configs_get_distinct_consecutive_seeds():
    config_by_split = get_dataset_config_by_split_from_dataset_config(_dataset_config(), seed=10)
    assert set(config_by_split) == {'train', 'val', 'test'}
    assert [config_by_split[k]['seed'] for k in ('train', 'val', 'test')] == [10, 11, 12]
    # generator kwargs and dims are copied into every split config
    for cfg in config_by_split.values():
        assert cfg['V'] == 64 and cfg['L'] == 32 and cfg['N_facts'] == 4
        assert cfg['num_key_repeats'] == 1


def test_dataloaders_by_split_have_expected_lengths():
    config_by_split = get_dataset_config_by_split_from_dataset_config(_dataset_config(), seed=0)
    dataloaders = get_mqar_dynamic_dataloaders_by_split(config_by_split)
    assert set(dataloaders) == {'train', 'val', 'test'}
    assert len(dataloaders['train']) == 8   # 64 / 8
    assert len(dataloaders['val']) == 4     # 32 / 8
    batch = next(iter(dataloaders['val']))
    assert batch.x_ids.shape == (8, 32)
