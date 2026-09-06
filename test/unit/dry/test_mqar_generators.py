import pytest
import torch

from mqar.generators import generate_mqar_batch


V, L, N_FACTS, BATCH = 64, 32, 4, 16


def test_batch_shapes_and_vocab_range():
    batch = generate_mqar_batch(V=V, L=L, N_facts=N_FACTS, batch_size=BATCH, seed=0)
    assert batch.x_ids.shape == (BATCH, L)
    assert batch.y_true_ids.shape == (BATCH, L)
    assert batch.V == V and batch.L == L and batch.N_facts == N_FACTS and batch.size == BATCH
    assert batch.x_ids.dtype == torch.int64
    assert int(batch.x_ids.max()) < V
    assert int(batch.x_ids.min()) >= 0


def test_deterministic_for_fixed_seed():
    a = generate_mqar_batch(V=V, L=L, N_facts=N_FACTS, batch_size=BATCH, seed=123)
    b = generate_mqar_batch(V=V, L=L, N_facts=N_FACTS, batch_size=BATCH, seed=123)
    c = generate_mqar_batch(V=V, L=L, N_facts=N_FACTS, batch_size=BATCH, seed=124)
    assert torch.equal(a.x_ids, b.x_ids)
    assert torch.equal(a.y_true_ids, b.y_true_ids)
    assert not torch.equal(a.x_ids, c.x_ids)


def test_reduce_to_AR_truncates_to_pairs_plus_query():
    # AR reduction ignores the passed L: sequence becomes the N_facts kv pairs
    # followed by a single query token
    batch = generate_mqar_batch(V=V, L=500, N_facts=N_FACTS, batch_size=BATCH, seed=0,
                                reduce_to_AR=True)
    expected_L = 2 * N_FACTS + 1
    assert batch.L == expected_L
    assert batch.x_ids.shape == (BATCH, expected_L)
    assert batch.y_true_ids.shape == (BATCH, expected_L)


def test_labels_mark_queries_only():
    # labels are the recall targets at query positions; all other positions are ignored.
    # every labeled target must be a value that appeared in the sequence's kv pairs
    batch = generate_mqar_batch(V=V, L=L, N_facts=N_FACTS, batch_size=BATCH, seed=7)
    labeled = batch.y_true_ids >= 0
    assert labeled.any(), "expected at least one query target per batch"
    for row in range(BATCH):
        targets = batch.y_true_ids[row][labeled[row]]
        sequence = set(batch.x_ids[row].tolist())
        assert set(targets.tolist()).issubset(sequence)


def test_key_and_query_repeats_are_wired_through():
    # repeats change the generated layout; smoke-check the wiring, not the layout
    base = generate_mqar_batch(V=256, L=128, N_facts=8, batch_size=4, seed=0)
    repeated_keys = generate_mqar_batch(V=256, L=128, N_facts=8, batch_size=4, seed=0,
                                        num_key_repeats=2)
    repeated_queries = generate_mqar_batch(V=256, L=128, N_facts=8, batch_size=4, seed=0,
                                           num_query_repeats=2)
    assert base.x_ids.shape == repeated_keys.x_ids.shape == repeated_queries.x_ids.shape
    assert not torch.equal(base.x_ids, repeated_keys.x_ids)
    assert not torch.equal(base.x_ids, repeated_queries.x_ids)
    # repeated queries -> more labeled recall targets
    assert (repeated_queries.y_true_ids >= 0).sum() > (base.y_true_ids >= 0).sum()


def test_simultaneous_key_and_query_repeats_are_rejected():
    # the generator supports one repeated axis at a time (zoology constraint)
    with pytest.raises(AssertionError):
        generate_mqar_batch(V=256, L=128, N_facts=8, batch_size=4, seed=0,
                            num_key_repeats=2, num_query_repeats=2)
