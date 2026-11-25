"""
Zero-shot associative recall (AR) on the pretrained Mamba2 family (natural-language models).

A pretrained `state-spaces/mamba2-*` checkpoint sees a list of random key-value facts and is asked,
one fact at a time, to complete the value of a key - no fine-tuning, greedy decoding.

Prompt (one per query; the context is shared by all queries of a seed):

    <prefix> k_1=v_1, k_2=v_2, ..., k_Nf=v_Nf, k_q=

with `prefix`, the fact/query templates and the separator taken from the config's `prompt` section;
keys are random lowercase strings of `facts.key_chars` letters (unique within a context), values
random strings of `facts.value_digits` digits. A query is correct iff the first run of digits the
model generates equals the value. Per seed (= one random context) `facts.queries_per_context` keys
are sampled and scored; accuracy is averaged over them. The saved grid has shape
(n_seeds, n_models, n_N_facts), i.e. (seed, y, x) like the training grids, so
`accuracy_grid__val.npy` plugs into the same figure-manifest conventions.

Reproducibility: every random choice (facts, query sample) derives from (runtime.seed, seed, N_f);
decoding is greedy; checkpoints and tokenizer are pinned to Hub revisions in the config, and the
resolved revisions plus library versions are archived next to the results (run_metadata.json).

Standalone entry point (the training grids keep src/main.py):

    python src/experiments/nl_zero_shot_mamba2.py -c config/appendix/nl_zero_shot_mamba2.json5

Needs `mamba-ssm`; the checkpoints are fetched from the Hub on the first run.
"""
import argparse
import json
import platform
import random
import re
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import json5
import numpy as np
import torch

if __name__ == '__main__':  # standalone run: make src/ importable like src/main.py does
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.common import results_dir, set_seed, test_results_dir

EXPERIMENT_TYPE = 'nl_zero_shot_mamba2'
SPLIT = 'val'  # single split: there is no training; the name keeps the bundle layout uniform


# ------------------------------------------------------------------------------------------ init

def initialize_nl_zero_shot_run(run_config: dict[str, Any], config_path: str | Path = None) -> str:
    """Mirror of experiments.grid_runs_utils.initialize_grid_run for this experiment type:
    names the run, creates results/<experiment_name>/<run_dir>/ and archives the config there as
    a byte-identical copy of the source file."""
    assert run_config['runtime'].get('experiment_type') == EXPERIMENT_TYPE
    run_config.setdefault('io', {})

    time_zone_name = run_config['io'].get('time_zone', None)
    time_zone = ZoneInfo(time_zone_name) if time_zone_name else None
    time_now = datetime.now(time_zone)
    title_timestamp = time_now.strftime("%d.%m.%Y %H-%M-%S")
    dir_timestamp = time_now.strftime("%d_%m_%Y__%H_%M_%S")

    experiment_name = run_config['runtime']['experiment_name']
    facts = run_config['facts']
    n_facts_axis = _parse_axis(facts['N_facts'])
    models = run_config['models']
    n_seeds = run_config['seeds']['n_seeds']

    descriptor = (f"Nf{int(n_facts_axis.min())}-{int(n_facts_axis.max())}_"
                  f"k{facts['key_chars']}_v{facts['value_digits']}_q{facts['queries_per_context']}_"
                  f"seeds{n_seeds}")
    run_name = (f"{experiment_name} | zero-shot AR, {len(models)} pretrained Mamba2 | "
                f"{descriptor.replace('_', ', ')} | {title_timestamp}")
    run_dir_name = f"hf_mamba2__{descriptor}__{dir_timestamp}"

    is_test = run_config['runtime'].get('is_test', False)
    results_dir_to_use = results_dir if not is_test else test_results_dir
    run_results_dir = results_dir_to_use / experiment_name / run_dir_name
    run_config['io']['run_results_dir'] = str(run_results_dir)
    run_results_dir.mkdir(parents=True, exist_ok=True)

    # byte-identical archive of the source config (runtime-filled fields stay in memory only)
    saved_config = run_results_dir / "run_config.json5"
    if config_path is not None:
        saved_config.write_bytes(Path(config_path).read_bytes())
    else:
        saved_config.write_text(json5.dumps(run_config, indent=2))

    print(f"\n\nstarting zero-shot AR run:\n{run_name}\n")
    print(f"results are saved to: {run_results_dir}\n")
    return run_name


def _parse_axis(spec) -> np.ndarray:
    """Axis spec as in the grid configs: a numpy expression string, or a list of ints."""
    axis = eval(spec) if isinstance(spec, str) else np.asarray(spec)
    axis = np.asarray(axis).astype(int)
    assert axis.ndim == 1 and axis.size >= 1 and (axis > 0).all(), spec
    return axis


# ----------------------------------------------------------------------------------------- facts

def make_facts(n_facts: int, rng: random.Random, key_chars: int, value_digits: int) -> list[tuple[str, str]]:
    assert n_facts <= 26 ** key_chars, f"{n_facts} unique keys of {key_chars} letters do not exist"
    keys: set[str] = set()
    while len(keys) < n_facts:
        keys.add(''.join(rng.choices(string.ascii_lowercase, k=key_chars)))
    keys_list = sorted(keys)
    rng.shuffle(keys_list)
    values = [''.join(rng.choices(string.digits, k=value_digits)) for _ in range(n_facts)]
    return list(zip(keys_list, values))


def build_prompts(facts, prompt_config, encode):
    """Token ids of the shared context and, per fact, of its query.

    context = enc(prefix) + [enc(fact_i) + enc(separator)] for every fact; the query of fact i is
    enc(query_format(k_i)), appended to the context. Facts and queries are tokenised piecewise so
    the query key tokenises exactly like the key inside its fact (both start with the template's
    leading space)."""
    fact_fmt, query_fmt = prompt_config['fact_format'], prompt_config['query_format']
    sep_ids = encode(prompt_config['separator'])
    context = list(encode(prompt_config['prefix']))
    queries = []
    for key, value in facts:
        context += encode(fact_fmt.format(key=key, value=value)) + sep_ids
        queries.append(encode(query_fmt.format(key=key)))
    return context, queries


def score_generation(text: str, value: str) -> bool:
    match = re.search(r'\d+', text)
    return bool(match) and match.group(0) == value


# ------------------------------------------------------------------------------------------ eval

def _context_seed(parent_seed: int, seed: int, n_facts: int) -> int:
    # distinct, order-independent stream per (seed, N_f); parent seed shifts everything
    return parent_seed * 1_000_003 + seed * 1_009 + n_facts


@torch.no_grad()
def evaluate_model(model, tokenizer, run_config, device: str, n_facts_axis: np.ndarray, seeds: range,
                   log=print) -> np.ndarray:
    """Accuracy per (seed, N_f) for one model: array of shape (n_seeds, n_N_facts)."""
    prompt_config = run_config['prompt']
    facts_config = run_config['facts']
    gen_config = run_config['generation']
    parent_seed = run_config['runtime']['seed']
    max_new = int(gen_config['max_new_tokens'])
    batch_size = int(gen_config['batch_size'])
    queries_per_context = facts_config.get('queries_per_context', None)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)

    accuracy = np.full((len(seeds), len(n_facts_axis)), np.nan)
    t0 = time.time()
    for j, n_facts in enumerate(n_facts_axis):
        for i, seed in enumerate(seeds):
            rng = random.Random(_context_seed(parent_seed, seed, int(n_facts)))
            facts = make_facts(int(n_facts), rng, facts_config['key_chars'], facts_config['value_digits'])
            context, queries = build_prompts(facts, prompt_config, encode)
            query_ids = list(range(len(facts)))
            if queries_per_context and queries_per_context < len(facts):
                query_ids = sorted(rng.sample(query_ids, queries_per_context))
            hits = 0
            # rectangular batches: group queries by token length
            by_len: dict[int, list[int]] = {}
            for q in query_ids:
                by_len.setdefault(len(queries[q]), []).append(q)
            for group in by_len.values():
                for start in range(0, len(group), batch_size):
                    chunk = group[start:start + batch_size]
                    ids = torch.tensor([context + queries[q] for q in chunk], device=device)
                    out = model.generate(ids, max_length=ids.shape[1] + max_new, top_k=1, cg=False,
                                         enable_timing=False, return_dict_in_generate=True, output_scores=False)
                    generated = out.sequences[:, ids.shape[1]:]
                    for row, q in enumerate(chunk):
                        hits += score_generation(tokenizer.decode(generated[row].tolist()), facts[q][1])
            accuracy[i, j] = hits / len(query_ids)
        log(f"N_f={int(n_facts):4d}  accuracy={accuracy[:, j].mean():.3f} +- {accuracy[:, j].std():.3f} "
            f"over {len(seeds)} seeds  ({time.time() - t0:.0f}s)")
    return accuracy


def load_model_and_tokenizer(model_config: dict[str, Any], tokenizer_config: dict[str, Any], device: str,
                             dtype: torch.dtype):
    """Pinned Hub revisions -> local snapshot dirs -> mamba_ssm model + transformers tokenizer."""
    import warnings
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    with warnings.catch_warnings():
        warnings.simplefilter('default')  # third-party loaders may raise benign UserWarnings
        model_dir = snapshot_download(model_config['hf_repo'], revision=model_config.get('revision'))
        tokenizer_dir = snapshot_download(tokenizer_config['hf_repo'], revision=tokenizer_config.get('revision'),
                                          allow_patterns=['*.json', '*.txt', '*.model', 'tokenizer*'])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        model = MambaLMHeadModel.from_pretrained(model_dir, device=device, dtype=dtype).eval()
    return model, tokenizer, Path(model_dir).name, Path(tokenizer_dir).name


def _run_one_model(model_index: int, run_config: dict[str, Any], device: str, out_file: str):
    """Worker: evaluate one model on one device; writes its (n_seeds, n_N_facts) array to out_file."""
    model_config = run_config['models'][model_index]
    if device.startswith('cuda'):
        torch.cuda.set_device(torch.device(device))  # mamba_ssm's Triton kernels launch on the current device
    set_seed(run_config['runtime']['seed'])
    dtype = getattr(torch, run_config['generation'].get('dtype', 'bfloat16'))
    n_facts_axis = _parse_axis(run_config['facts']['N_facts'])
    seeds = _seeds(run_config['seeds'])
    tag = f"[{model_config['name']} @ {device}]"
    t0 = time.time()
    model, tokenizer, model_rev, tokenizer_rev = load_model_and_tokenizer(
        model_config, run_config['tokenizer'], device, dtype)
    print(f"{tag} loaded in {time.time() - t0:.0f}s (snapshot {model_rev[:12]}, tokenizer {tokenizer_rev[:12]})", flush=True)
    accuracy = evaluate_model(model, tokenizer, run_config, device, n_facts_axis, seeds,
                              log=lambda s: print(f"{tag} {s}", flush=True))
    np.save(out_file, accuracy)
    meta = {'model': model_config, 'snapshot': model_rev, 'tokenizer_snapshot': tokenizer_rev,
            'device_name': torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else device,
            'seconds': round(time.time() - t0)}
    Path(out_file).with_suffix('.json').write_text(json.dumps(meta, indent=1))


def _seeds(seeds_config: dict[str, int]) -> range:
    n_seeds = seeds_config.get('n_seeds', 1)
    start_seed = seeds_config.get('start_seed', 0)
    return range(start_seed, start_seed + n_seeds)


def _devices(run_config: dict[str, Any]) -> list[str]:
    """Devices to spread the models over (one model per device, round-robin). As in the grid runner:
    parallel.devices_to_use lists them; null means every visible CUDA device (so CUDA_VISIBLE_DEVICES
    selects the set), or the CPU without CUDA. runtime.device, when set, pins everything to one device."""
    device = run_config['runtime'].get('device', None)
    if device:
        return [device]
    devices = (run_config.get('parallel') or {}).get('devices_to_use', None)
    if devices:
        return [f"cuda:{d}" if isinstance(d, int) else str(d) for d in devices]
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print("no GPU device available; using CPU instead")
    return ['cpu']


# ------------------------------------------------------------------------------------------- run

def execute_nl_zero_shot_run(run_name: str, run_config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every model of the config (one process per model, round-robin over the devices of
    parallel.devices_to_use, or sequentially on runtime.device) and save the (seed, model, N_f)
    accuracy grid, a per-cell CSV and the run metadata into the run's results dir."""
    import multiprocessing as mp

    run_results_dir = Path(run_config['io']['run_results_dir'])
    models = run_config['models']
    devices = _devices(run_config)
    n_facts_axis = _parse_axis(run_config['facts']['N_facts'])
    seeds = _seeds(run_config['seeds'])
    set_seed(run_config['runtime']['seed'])

    tmp_dir = run_results_dir / '_per_model'
    tmp_dir.mkdir(exist_ok=True)
    jobs = [(k, run_config, devices[k % len(devices)], str(tmp_dir / f"{k}__{m['name']}.npy"))
            for k, m in enumerate(models)]

    if len(devices) > 1:
        ctx = mp.get_context('spawn')
        # as many concurrent processes as devices; the (k % len(devices)) assignment keeps one model per device
        with ctx.Pool(processes=len(devices)) as pool:
            pool.starmap(_run_one_model, jobs)
    else:
        for job in jobs:
            _run_one_model(*job)

    accuracy_grid = np.stack([np.load(job[3]) for job in jobs], axis=1)  # (n_seeds, n_models, n_N_facts)
    np.save(run_results_dir / f"accuracy_grid__{SPLIT}.npy", accuracy_grid)

    with open(run_results_dir / f"accuracy_table__{SPLIT}.csv", 'w') as f:
        f.write("model,N_facts,accuracy_mean,accuracy_std,n_seeds\n")
        for k, m in enumerate(models):
            for j, n_facts in enumerate(n_facts_axis):
                cell = accuracy_grid[:, k, j]
                f.write(f"{m['name']},{int(n_facts)},{cell.mean():.4f},{cell.std():.4f},{len(seeds)}\n")

    per_model_meta = [json.loads(Path(job[3]).with_suffix('.json').read_text()) for job in jobs]
    metadata = {
        'run_name': run_name,
        'experiment_type': EXPERIMENT_TYPE,
        'N_facts': [int(x) for x in n_facts_axis],
        'seeds': list(seeds),
        'models': per_model_meta,
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'transformers': _version('transformers'), 'mamba_ssm': _version('mamba_ssm')},
        'example_prompt': _example_prompt(run_config),
    }
    (run_results_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=1))
    for job in jobs:  # the per-model partials are folded into the grid
        Path(job[3]).unlink(missing_ok=True)
        Path(job[3]).with_suffix('.json').unlink(missing_ok=True)
    tmp_dir.rmdir()

    print(f"\nfinished: {run_name}")
    for k, m in enumerate(models):
        print(f"  {m['name']:14s} " + ' '.join(f"{int(n)}:{accuracy_grid[:, k, j].mean():.2f}"
                                                for j, n in enumerate(n_facts_axis)))
    return {'accuracy_grid': accuracy_grid, 'run_results_dir': run_results_dir}


def _version(package: str) -> str | None:
    try:
        return __import__(package).__version__
    except Exception:
        return None


def _example_prompt(run_config: dict[str, Any]) -> str:
    """The exact text of the first query of seed 0 at the smallest N_f (for the record)."""
    facts_config = run_config['facts']
    n_facts = int(_parse_axis(facts_config['N_facts']).min())
    rng = random.Random(_context_seed(run_config['runtime']['seed'], _seeds(run_config['seeds']).start, n_facts))
    facts = make_facts(n_facts, rng, facts_config['key_chars'], facts_config['value_digits'])
    p = run_config['prompt']
    return (p['prefix'] + ''.join(p['fact_format'].format(key=k, value=v) + p['separator'] for k, v in facts)
            + p['query_format'].format(key=facts[0][0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument("-c", "--config", type=Path, required=True,
                        help="run config, e.g. config/appendix/nl_zero_shot_mamba2.json5")
    args = parser.parse_args()

    print(f"\nloading run config from: \n{args.config}")
    run_config = json5.loads(args.config.read_text())
    run_config['runtime']['experiment_name'] = args.config.stem
    run_name = initialize_nl_zero_shot_run(run_config=run_config, config_path=args.config)
    execute_nl_zero_shot_run(run_name, run_config=run_config)


if __name__ == "__main__":
    main()
