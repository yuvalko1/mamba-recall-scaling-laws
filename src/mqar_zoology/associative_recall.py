import numpy as np
import torch

from mqar_zoology.config import DataSegmentConfig
from mqar_zoology.data_utils import DataSegment

IGNORED_TOKEN = -100


def max_num_repeating_keys_and_queries(input_seq_len: int, num_kv_pairs: int = 8) -> tuple[int, int]:
    """
    Returns (max_N_k, max_N_q): the maximum num_key_repeats and num_query_repeats
    supported for the given input_seq_len and num_kv_pairs.

    Both are derived from the same constraint:
        num_kv_pairs * 2 * (N + 1) <= input_seq_len
    which ensures room for N repetition blocks plus at least one complementary section.
    """
    context_size = num_kv_pairs * 2
    max_N_k = input_seq_len // context_size - 1
    max_N_q = input_seq_len // context_size - 1
    return max_N_k, max_N_q


class MQARConfig(DataSegmentConfig):
    name: str="multiquery_ar"
    power_a: float=0.01
    num_kv_pairs: int=8
    random_non_queries: bool=True
    include_slices: bool=True

    def build(self, seed: int) -> DataSegment:
        return multiquery_ar(**self.model_dump(), seed=seed)


def multiquery_ar_repeated_keys(
    vocab_size: int,        # V
    num_examples: int,
    input_seq_len: int,     # L
    seed: int,
    num_kv_pairs: int=8,    # N_facts
    random_non_queries: bool=True,
    num_key_repeats: int=2,
    include_slices: bool=True,
    **kwargs
) -> DataSegment:
    """
    Variant of multiquery_ar where each key appears num_key_repeats times in the context,
    each time paired with a different value. Queries always ask for the canonical (last) value.
    Uses a uniform gap distribution (power_a is ignored).
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * (num_key_repeats + 1) <= input_seq_len, \
        "not enough room for all KV chunks and at least one query slot per pair"
    assert num_key_repeats > 1

    np.random.seed(seed)

    context_size = num_kv_pairs * 2                          # one KV chunk
    total_context_size = context_size * num_key_repeats      # all KV chunks combined
    num_chunks = num_key_repeats - 1

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    # canonical KV block
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # extra KV chunks: same keys (permuted), different values, labels ignored
    extra_keys = np.hstack([np.apply_along_axis(np.random.permutation, 1, keys) for _ in range(num_chunks)])
    extra_values = np.hstack(
        [np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)
         for _ in range(num_chunks)])

    extra_kvs = np.zeros((num_examples, context_size * num_chunks), dtype=np.int64)
    extra_kvs[:, 0::2] = extra_keys
    extra_kvs[:, 1::2] = extra_values

    # query section fits within the remaining L - total_context_size tokens
    query_len = input_seq_len - total_context_size          # tokens available for queries
    space = query_len // 2                                   # number of (key, gap) slots
    p = np.ones(space) / space                               # uniform gap distribution

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    queries = np.zeros((num_examples, query_len + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)

    labels = np.full((num_examples, input_seq_len + 1), IGNORED_TOKEN, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + total_context_size + 1, values=values, axis=1)

    # assemble: [extra KV chunks | canonical KV | query section]  (total = L + 1 before slicing)
    # canonical KV is last, so queries always refer to the most recent value per key
    examples = np.concatenate([extra_kvs, kvs, queries], axis=1)
    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return DataSegment(inputs, labels, slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len})


def multiquery_ar_repeated_queries(
    vocab_size: int,        # V
    num_examples: int,
    input_seq_len: int,     # L
    seed: int,
    num_kv_pairs: int=8,    # N_facts
    random_non_queries: bool=True,
    num_query_repeats: int=2,
    include_slices: bool=True,
    **kwargs
) -> DataSegment:
    """
    Variant of multiquery_ar where the query section is repeated num_query_repeats times,
    each repetition with independently sampled gap positions.
    Uses a uniform gap distribution (power_a is ignored).
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * (num_query_repeats + 1) <= input_seq_len, \
        "not enough room for KV context and at least one query slot per pair per section"
    assert num_query_repeats > 1

    np.random.seed(seed)

    context_size = num_kv_pairs * 2
    query_len = (input_seq_len - context_size) // num_query_repeats  # tokens per query section
    space = query_len // 2                                            # query slots per section

    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # uniform gap distribution
    p = np.ones(space) / space

    def _generate_query_section():
        x = np.stack([np.arange(space, dtype=int)] * num_examples)
        gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
        _queries = np.zeros((num_examples, query_len), dtype=np.int64)
        _labels = np.full((num_examples, query_len), IGNORED_TOKEN, dtype=np.int64)
        np.put_along_axis(_queries, gaps * 2, values=keys, axis=1)
        np.put_along_axis(_labels, gaps * 2, values=values, axis=1)
        return _queries, _labels

    # generate all query sections (each with independent gap placements)
    all_queries, all_labels = zip(*[_generate_query_section() for _ in range(num_query_repeats)])

    inputs = torch.tensor(np.concatenate([kvs, *all_queries], axis=1))
    labels = torch.tensor(np.concatenate([
        np.full((num_examples, context_size), IGNORED_TOKEN, dtype=np.int64),
        *all_labels,
    ], axis=1))

    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return DataSegment(inputs, labels, slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len})


def multiquery_ar(
    vocab_size: int,        # V
    num_examples: int,
    input_seq_len: int,     # L
    seed: int,
    power_a: float=0.01,
    num_kv_pairs: int=8,    # N_facts
    random_non_queries: bool=True,
    num_key_repeats: int=1,
    num_query_repeats: int=1,
    include_slices: bool=True,
    **kwargs
) -> DataSegment:
    """
    Generates synthetic data for the multi-query associative recall task as described in
    Arora,Eyuboglu, et al. "Zoology: Measuring and improving recall in efficient language models.".

    Example:
        `multiquery_ar(vocab_size=12, num_kv_pairs=2, input_seq_len=16, random_non_queries=False)`
        will generate input and label sequences of the form:

                Key   Val  Key  Val            Query                         Query
        Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0
        Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

        The -100 labels are ignored by the loss function and metrics.

    We include one important note on the power law distribution. In real language data,
    the gap between repeated bigrams follows a power law. Intuitively, if the bigram
    "common buzzard" appears in text, the probability of the bigram appearing again
    drops the further away from the orginal mention we are. In our synthetic, we can
    control this with the power law parameters `train_power_a` and `test_power_a`.
    Setting these to 1.0 will result in a uniform distribution. You can visualize the
    distribution with the following code:
    ```
    space = 100
    power_a = 0.01
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()
    plt.plot(p)
    ```

    Args:
        vocab_size (int): The size of the vocabulary. As discussed in the Zoology
            paper, large vocabulary sizes (>1k) can be important for highlighting
            differences between model architectures. Defaults to 8_192.
        num_train_examples (int): The number of training examples to generate. Defaults
            to 100_000.
        num_test_examples (int): The number of test examples to generate. Defaults to
            3_000.
        input_seq_len (int): The length of the input sequence. Defaults to 64. In
            In Figure 2 of the Zoology paper, we vary the input sequence length from
            64 to 512 and the number of key-value pairs from 4 to 64.
        seed (int): The seed for the random number generator.
        num_kv_pairs (int): The number of key-value pairs.
        train_power_a (float, optional): The power for the power law distribution for
            training data. Defaults to 0.01.
        test_power_a (float, optional): The power for the power law distribution for
            test data. Defaults to 0.01.
        random_non_queries (bool, optional): If True, replace all the 0's (as in the
            example above) with random values in the input. Defaults to True.

    Returns:
        SyntheticData: A SyntheticData object containing the generated train and test
            inputs and labels.

    Raises:
        Warning: If potential data leakage is detected between the train and test sets.
    """
    assert not (num_key_repeats > 1 and num_query_repeats > 1), \
        "num_key_repeats and num_query_repeats cannot both be > 1"

    if num_key_repeats > 1:
        return multiquery_ar_repeated_keys(
            vocab_size=vocab_size, num_examples=num_examples, input_seq_len=input_seq_len,
            seed=seed, num_kv_pairs=num_kv_pairs, random_non_queries=random_non_queries,
            num_key_repeats=num_key_repeats, include_slices=include_slices,
        )

    if num_query_repeats > 1:
        return multiquery_ar_repeated_queries(
            vocab_size=vocab_size, num_examples=num_examples, input_seq_len=input_seq_len,
            seed=seed, num_kv_pairs=num_kv_pairs, random_non_queries=random_non_queries,
            num_query_repeats=num_query_repeats, include_slices=include_slices,
        )

    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 4 <= input_seq_len

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)

    labels = np.full((num_examples, input_seq_len + 1), IGNORED_TOKEN, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)

    examples = np.concatenate([kvs, queries], axis=1)
    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return DataSegment(
        inputs,
        labels,
        slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len}
    )