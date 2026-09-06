from dataclasses import dataclass

@dataclass
class MqarDimensions:

    # task
    V: int = None  # vocabulary size
    L: int = None  # context length
    N_facts: int = None  # number of facts
    N_k: int = None  # key repetitions (num_key_repeats; None -> from dataset kwargs)
    N_q: int = None  # query repetitions (num_query_repeats; None -> from dataset kwargs)

    # model
    D: int = None  # embedding dimensions
    N: int = None  # state dimensions
    M: int = None  # number of heads
    Lambda: int = None  # number of layers
