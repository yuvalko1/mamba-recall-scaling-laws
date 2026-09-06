import copy
import dataclasses
import io
import math
import multiprocessing as mp

import os
import shutil
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

from torch import nn

from mqar.generators import MqarBatch

try:
    from torch.amp import GradScaler
except ImportError:
    pass

import numpy as np
import torch
import wandb
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from mqar_zoology.associative_recall import IGNORED_TOKEN
from utils.common import set_seed, wandb_cache_dir, wandb_output_dir



def calculate_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    grad_sum_squared: torch.Tensor | None = None
    for param in model.parameters():
        if param.grad is not None:
            # detach gradient to avoid autograd tracking and flatten
            g = param.grad.detach().view(-1)
            s = torch.sum(g * g)
            if grad_sum_squared is None:
                # initialise on the same device as the gradient
                grad_sum_squared = s.clone()
            else:
                grad_sum_squared = grad_sum_squared + s
    if grad_sum_squared is None:
        # No gradients present; return a zero tensor on CPU
        return torch.tensor(0.0)
    return torch.sqrt(grad_sum_squared)


def _get_save_subdirs(save_dir: Path):

    train_logs_dir = save_dir / "train_logs"
    val_logs_dir = save_dir / "val_logs"
    test_logs_dir = save_dir / "test_logs"

    best_models_dir = save_dir / "best_models"
    model_checkpoints_dir = save_dir / "model_checkpoints"

    return train_logs_dir, val_logs_dir, test_logs_dir, best_models_dir, model_checkpoints_dir


def clean_up(start_datetime_str, verbose=False):
    if verbose:
        print('\nrunning clean up\n')
    tmp = f'./tmp/{start_datetime_str}'
    if os.path.exists(tmp):
        shutil.rmtree(tmp)


def get_available_context(device_name: str, use_amp: bool = False):
    """
    Returns (autocast_ctx, device_ctx).
    If use_amp is False, autocast_ctx is a nullcontext so everything runs in fp32.
    """
    device = torch.device(device_name)
    use_cuda = torch.cuda.is_available() and device.type == 'cuda'
    if not use_cuda:
        return nullcontext(), nullcontext()

    if not use_amp:
        # disable mixed precision: run everything in float32
        return nullcontext(), torch.cuda.device(device)

    # otherwise choose bf16 or fp16 based on SM capability
    major, minor = torch.cuda.get_device_capability(device)
    use_bf16 = (major >= 8)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    return torch.amp.autocast(device_type='cuda', dtype=dtype), torch.cuda.device(device)


def _copy_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def atomic_write_bytes(data: bytes, final_path: str):
    final_path = os.path.abspath(final_path)
    d = os.path.dirname(final_path)
    os.makedirs(d, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=d, prefix='._', suffix='.tmp')
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final_path)


def _save_model_checkpoint(model_state, models_dir, name, verbose=False):
    if verbose:
        print(f"saving {name} model checkpoint...")
    os.makedirs(models_dir, exist_ok=True)
    buf = io.BytesIO()
    torch.save(model_state, buf)
    atomic_write_bytes(buf.getvalue(), os.path.join(models_dir, f'{name}.pt'))
    if verbose:
        print("saved")



def send_logs(logs: dict[str, Any], logs_queue: mp.Queue, task_id: str, split: str):

    # prepare logs
    accuracy = logs['accuracy']
    log_item = task_id, split, accuracy


    # write to queue
    logs_queue.put(log_item)


def _safe_path_name(name: str, default='default') -> str:
    s = ''.join(c if c.isalnum() else '_' for c in name)   # non-alnum -> _
    s = '_'.join(part for part in s.split('_') if part)    # collapse repeats
    s = s.strip('_') or default
    return s


def run_train_loop(
        model: nn.Module,
        dataloaders: dict[str, DataLoader],
        run_config: dict[str, Any],
        debug_text: str = None,
        logs_queue: mp.Queue = None,
        task_id: str = None,
        progress_queue: mp.Queue = None,
) -> dict[str, Any]:

    runtime_config = run_config['runtime']
    train_config = run_config['training']
    wandb_config = run_config['wandb']

    save_dir = Path(run_config["io"]["run_results_dir"])
    safe_run_name = _safe_path_name(wandb_config['run_name'])

    set_seed(runtime_config['seed'])

    model.to(runtime_config['device'])

    start_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    label_smoothing = train_config.get('label_smoothing', 0)
    loss_fn = CrossEntropyLoss(ignore_index=IGNORED_TOKEN, label_smoothing=label_smoothing)

    # dirs
    train_logs_dir, val_logs_dir, test_logs_dir, best_models_dir, model_checkpoints_dir = _get_save_subdirs(save_dir)

    # ---

    model.eval()

    desc_text = (f"{debug_text} | " if debug_text is not None else "")

    initial_val_log = evaluate_split(
        model, dataloaders['val'], run_config,
        loss_fn=loss_fn, step=0, desc_text=desc_text,
    )
    val_accuracy = initial_val_log['accuracy']

    if (not runtime_config['should_train']) or (val_accuracy > train_config['threshold_accuracy']):
        send_logs(logs=initial_val_log, logs_queue=logs_queue, task_id=task_id, split='val')
        # save model checkpoint
        if train_config['save_model_at_evaluation']:
            state = _copy_state_dict(model)
            _save_model_checkpoint(state, model_checkpoints_dir, name=safe_run_name)
        time.sleep(5)  # safety (make sure logs are sent and model is saved)
        return initial_val_log

    # prepare for training

    # build optimizer
    optimizer_config = train_config['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config['learning_rate'],
        betas=optimizer_config['betas'],
        weight_decay=optimizer_config['weight_decay'],
    )

    # build scheduler from utils.config if provided
    if (scheduler_config := train_config.get('scheduler')) is not None:
        scheduler = _construct_scheduler_from_config(optimizer, scheduler_config)
    else:
        scheduler = None

    # scaler (for AMP)
    if train_config.get('use_amp', False):
        scaler = GradScaler()
    else:
        scaler = None

    # wandb
    if wandb_config['activate']:

        cache_dir = str(wandb_config.get('cache_dir', wandb_cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['WANDB_CACHE_DIR'] = cache_dir

        if not wandb_config['verbose']:
            os.environ["WANDB_SILENT"] = "true"  # suppresses wandb console output
            os.environ["WANDB_CONSOLE"] = "off"  # don't wrap/redirect console

        # if a run is already active in this process, close it first
        if wandb.run is not None:
            wandb.finish()

        output_dir = str(wandb_config.get('output_dir', wandb_output_dir))
        os.makedirs(output_dir, exist_ok=True)

        # init; retry since the wandb service can be busy when many workers start at once
        num_init_attempts = 5
        for attempt in range(num_init_attempts):
            try:
                wandb_run = wandb.init(
                    project=wandb_config['project_name'],
                    name=wandb_config['run_name'],
                    dir=output_dir,
                    config=run_config,
                    reinit='finish_previous',
                )
                break
            except Exception:
                if attempt == num_init_attempts - 1:
                    raise
                time.sleep(5 * (attempt + 1))

    else:
        wandb_run = None

    # train model
    best_model, results = _train_model(
        model, dataloaders,
        optimizer, scheduler, scaler,
        loss_fn, run_config,
        wandb_run=wandb_run,
        desc_text=desc_text,
        logs_queue=logs_queue,
        task_id=task_id,
        progress_queue=progress_queue,
    )

    wandb.finish()
    clean_up(start_str)

    return results


def wandb_log(logs: dict, step: int, name: str = None, commit: bool = True):

    if name is not None:
        named_logs = {f"{k} ({name})": v for k, v in logs.items()}
    else:
        named_logs = logs

    wandb.log(named_logs, step=step, commit=commit)


def _train_model(
        model, dataloaders,
        optimizer, scheduler, scaler,
        loss_fn,
        run_config,
        wandb_run=None,
        desc_text=None,
        logs_queue=None,
        task_id=None,
        progress_queue=None,
):
    runtime_config = run_config['runtime']
    train_config = run_config['training']
    wandb_config = run_config['wandb']

    device = runtime_config['device']

    save_dir = Path(run_config["io"]["run_results_dir"])
    logs_save_steps = int(train_config.get("save_logs_any_num_steps", None))
    use_amp = train_config.get('use_amp', False)
    wandb_train_log_steps = int(wandb_config.get('log_train_any_num_steps', 1))
    assert wandb_train_log_steps >= 1

    # gradient accumulation setup
    accum_steps = int(train_config.get('grad_accum_steps', 1))
    assert accum_steps >= 1

    # prepare logs dir
    safe_run_name = _safe_path_name(wandb_config['run_name'])

    # ...
    model.to(device)
    model.train()

    # init
    step_ = 0
    best_accuracy = 0
    best_state = _copy_state_dict(model)
    start_time = time.perf_counter()

    # tqdm
    use_tqdm = run_config['training']['view_train_tqdm']
    if use_tqdm:
        train_bar = tqdm(
            dataloaders['train'],
            desc=f'{desc_text}Train (batches)',
            unit='batch',
            leave=True,
            dynamic_ncols=True,
        )
        train_iterator = train_bar
    else:
        train_bar = None
        train_iterator = dataloaders['train']
    num_training_steps = len(train_iterator)

    # batch-progress reporting to the parent grid scheduler (group-level tqdm view);
    # throttled to ~1% granularity so queue traffic stays negligible
    progress_every = max(1, num_training_steps // 100)

    def _send_progress(batches_done: int):
        if progress_queue is None or task_id is None:
            return
        try:
            progress_queue.put_nowait((task_id, batches_done, num_training_steps))
        except Exception:
            pass  # progress display is best-effort; never disturb training


    # dirs
    (train_logs_dir, val_logs_dir, test_logs_dir,
     best_models_dir, model_checkpoints_dir) = _get_save_subdirs(save_dir)

    should_break = False

    # training loop: steps
    for step, batch in enumerate(train_iterator):

        cur_step_logs = {}

        # train
        train_logs, _ = train_or_evaluate_batch_stats(
            model, batch, device, train_config,
            ignored_token=IGNORED_TOKEN, train=True,
            loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
            use_amp=use_amp, scaler=scaler,
        )

        # save logs
        if (logs_save_steps is not None) and (step % logs_save_steps == 0):
            send_logs(logs=train_logs, logs_queue=logs_queue, task_id=task_id, split='train')

        # evaluate
        if (step > 0) and (step % train_config['evaluate_any_num_steps'] == 0) or (step == num_training_steps-1):

            best_accuracy, best_state, should_break, cur_step_logs, val_logs = _perform_evaluation_step(
                model, dataloaders, run_config, loss_fn, step, desc_text,
                best_accuracy, best_state, cur_step_logs, save_dir,
            )

            # save logs
            send_logs(logs=val_logs, logs_queue=logs_queue, task_id=task_id, split='val')

        # update logs and log to wandb
        if wandb_config['activate'] and (step % wandb_train_log_steps == 0):
            cur_step_logs.update({'train': train_logs})
            if step != len(train_iterator)-1:
                wandb_log(cur_step_logs, step=step)

        # update tqdm
        if train_bar is not None:
            train_bar.set_postfix({
                'loss': f"{train_logs['loss']:.3e}",
                'accuracy': f"{train_logs['accuracy']:.3f}",
                'grad_norm': f"{train_logs['grad_norm']:.2e}",
            })

        # report batch progress to the parent grid scheduler
        if step % progress_every == 0:
            _send_progress(step + 1)

        if should_break:
            break

    # end run

    _send_progress(num_training_steps)  # early break included: the task's training is over

    # close tqdm
    if train_bar is not None:
        train_bar.close()

    # finish wandb run
    if wandb_run is not None:
        wandb_run.summary["status"] = "finished"
        wandb_run.finish()

    # save model state (best and last)
    if train_config.get('save_model_checkpoints', True):
        final_state = _copy_state_dict(model)
        _save_model_checkpoint(best_state, best_models_dir, safe_run_name)
        _save_model_checkpoint(final_state, model_checkpoints_dir, safe_run_name)

    # evaluate best model on test set
    model.load_state_dict(best_state)
    model.eval()
    test_logs = evaluate_split(
        model, dataloaders['test'], run_config,
        loss_fn=loss_fn, step=step_, desc_text=desc_text,
    )
    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    send_logs(logs=test_logs, logs_queue=logs_queue, task_id=task_id, split='test')
    time.sleep(5)  # safety (make sure logs are sent)

    final_results = copy.copy(test_logs)

    best_model = model

    wandb.finish()

    return best_model, final_results


def _perform_evaluation_step(
        model, dataloaders, run_config, loss_fn, step, desc_text,
        best_accuracy, best_state, cur_step_logs, save_dir,
):
    should_break = False

    # configs
    train_config = run_config['training']
    wandb_config = run_config['wandb']
    accuracy_threshold = train_config['threshold_accuracy']
    safe_run_name = _safe_path_name(wandb_config['run_name'])

    # dirs
    _, _, _, best_models_dir, model_checkpoints_dir = _get_save_subdirs(save_dir)

    # evaluate
    model.eval()
    val_logs = evaluate_split(
        model, dataloaders['val'], run_config,
        loss_fn=loss_fn, step=step, desc_text=desc_text,
    )
    val_accuracy = val_logs['accuracy']
    model.train()

    # log
    if wandb_config['activate']:

        if wandb_config['log_val']:
            cur_step_logs.update({'val': val_logs})

    # save best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_state = _copy_state_dict(model)
        if train_config.get('save_model_checkpoints', True):
            _save_model_checkpoint(best_state, best_models_dir, name=safe_run_name)

    # save model checkpoint
    if train_config['save_model_at_evaluation']:
        state = _copy_state_dict(model)
        _save_model_checkpoint(state, model_checkpoints_dir, name=safe_run_name)

    # stop training if exceeded threshold
    if val_accuracy > accuracy_threshold:
        model.train()
        should_break = True

    return best_accuracy, best_state, should_break, cur_step_logs, val_logs


def _construct_scheduler_from_config(optimizer, scheduler_config):
    """
    LR *multiplier* schedule (ratios only):
      - Warmup:  0 → 1 over `num_warmup_steps`
      - Flat:    1 over `num_steps_at_max`
      - Decay:   1 → decay_factor over `num_decay_steps`, shaped by `decay_type`
                 ("linear": linear or exp interpolation, see `interpolation`;
                  "cosine": cosine interpolation)
      - Flat:    decay_factor thereafter

    Applies the same ratio to all param groups (multiplies each group's base lr).

    Expected keys in `scheduler_config`:
      - decay_type ("linear" or "cosine", default "linear")
      - num_warmup_steps (int >= 0, default 0)
      - num_steps_at_max (int >= 0, default 0)
      - num_decay_steps  (int >= 0, default 0)
      - decay_factor (float in [0,1], default 1.0; the LR floor, as a fraction of max)
      - interpolation ("linear" or "exp", default "linear"; decay_type "linear" only)
    """
    if not scheduler_config:
        return None

    decay_type = str(scheduler_config.get("decay_type", "linear"))
    assert decay_type in ['linear', 'cosine']

    n_warm  = int(scheduler_config.get("num_warmup_steps", 0) or 0)
    n_at_max = int(scheduler_config.get("num_steps_at_max", 0) or 0)
    n_decay = int(scheduler_config.get("num_decay_steps", 0) or 0)
    decay_factor = float(scheduler_config.get("decay_factor", 1.0))
    interpolation = str(scheduler_config.get("interpolation", "linear"))

    if (n_warm < 0) or (n_decay < 0) or (n_at_max < 0):
        raise ValueError("num_warmup_steps, num_decay_steps and num_steps_at_max must be >= 0.")
    if not (0.0 <= decay_factor <= 1.0):
        raise ValueError("decay_factor must be in [0, 1].")
    assert interpolation in ['linear', 'exp']

    # Nothing to schedule (always 1.0)
    if n_warm == 0 and n_decay == 0 and abs(decay_factor - 1.0) < 1e-12:
        return None

    def lr_lambda(step: int) -> float:

        # Phase 1: warmup 0 → 1
        if n_warm > 0 and step < n_warm:
            return step / float(max(1, n_warm))

        # Phase 2: flat at 1
        if n_at_max > 0 and step < n_warm + n_at_max:
            return 1

        # Phase 3: decay 1 → decay_factor
        if n_decay > 0 and step < n_warm + n_at_max + n_decay:
            t = step - (n_warm + n_at_max)
            frac = t / float(max(1, n_decay))  # in [0,1]
            if decay_type == "cosine":
                return decay_factor + (1.0 - decay_factor) * 0.5 * (1.0 + math.cos(math.pi * frac))
            elif interpolation == 'linear':
                return (1.0 - frac) * 1.0 + frac * decay_factor
            elif interpolation == "exp":
                # exponential: 1.0 at frac=0, decay_factor at frac=1
                return decay_factor ** frac

        # Phase 4: flat at decay_factor
        return decay_factor

    return LambdaLR(optimizer, lr_lambda)


@dataclasses.dataclass
class Accumulator:
    correct: int = 0
    total: int = 0


def evaluate_split(
        model, dataloader, config,
        loss_fn=None, ignored_token=IGNORED_TOKEN, step=0,
        verbose=False,
        desc_text="",
):
    use_tqdm = config['training']['view_evaluation_tqdm']

    if verbose:
        print(f"\nEvaluating over '{config['dataset']}'...")

    device = config['runtime']['device']

    model.to(device)
    model.eval()

    overall_accumulator = Accumulator(total=0, correct=0)
    loss_sum = 0
    total_n_query_tokens = 0

    if use_tqdm:
        val_bar = tqdm(
            dataloader,
            desc=f'{desc_text}Step {step} | Evaluation (batches)',
            unit='batch',
            leave=False,
        )
        val_iterator = val_bar
    else:
        val_iterator = dataloader
        val_bar = None

    for i, batch in enumerate(val_iterator):

        batch_logs, batch_accumulator = train_or_evaluate_batch_stats(
            model, batch, device, config,
            ignored_token=ignored_token, train=False, loss_fn=loss_fn,
        )
        n_query_tokens = batch_accumulator.total
        n_correct_preds = batch_accumulator.correct

        # update accumulators
        total_n_query_tokens += n_query_tokens
        overall_accumulator.total += n_query_tokens
        overall_accumulator.correct += n_correct_preds

        # update sums
        if loss_fn is not None:
            batch_mean_loss = batch_logs['loss']
            loss_sum += batch_mean_loss * n_query_tokens

    # mean accumulators
    overall_accuracy = float(overall_accumulator.correct / overall_accumulator.total)
    overall_error_rate = 1 - overall_accuracy

    loss_mean = loss_sum / total_n_query_tokens

    if val_bar is not None:
        val_bar.set_postfix(accuracy=f"{overall_accuracy:.3f}, loss={loss_mean:.3e}")
        val_bar.close()

    if verbose:
        print(
            f"Evaluation completed! "
            f"\n{overall_accuracy = }\n",
        )

    split_logs = {
        'accuracy': overall_accuracy,
        'error_rate': overall_error_rate,
        'loss': loss_mean,
    }

    return split_logs


def train_or_evaluate_batch_stats(
        model, batch: MqarBatch, device, train_config, ignored_token=IGNORED_TOKEN,
        train=False, loss_fn=None,
        optimizer=None, scheduler=None, scaler=None,
        use_amp=False,
):
    batch_logs = {}

    x_ids = batch.x_ids.to(device)
    y_true_ids = batch.y_true_ids.to(device)

    # gradient accumulation setup
    accum_steps = int(train_config.get('grad_accum_steps', 1) or 1)
    ga_counter = int(train_config.get('_ga_counter', 0))

    # set train/eval mode + zero grads
    if train:
        model.train()
        # zero grads only at the start of an accumulation cycle
        if ga_counter % accum_steps == 0:
            optimizer.zero_grad()
    else:
        model.eval()

    # get the right autocast + device contexts (only active on CUDA)
    autocast_ctx, device_ctx = get_available_context(device, use_amp=use_amp)

    with torch.set_grad_enabled(train):
        # both move-to-device & mixed-precision
        with device_ctx, autocast_ctx:
            y_pred_logits = model(input_ids=x_ids).logits  # [B, L, V]
            y_pred_logits = y_pred_logits.transpose(2, 1)  # [B, V, L]

    # Compute cross-entropy in float32 for stability
    if loss_fn is not None:
        raw_loss = loss_fn(y_pred_logits.float(), y_true_ids)
        loss = raw_loss
        if train and accum_steps > 1:
            # average the loss so accumulated grads match full-batch grads
            loss = loss / accum_steps

    # training-only: backward + step + scheduler
    if train:

        assert loss_fn is not None

        clip_grad_max_norm = train_config['clip_grad_max_norm']
        do_step = ((ga_counter + 1) % accum_steps == 0)

        if use_amp:
            assert scaler is not None, "Pass in a GradScaler when training"
            scaler.scale(loss).backward()
            if do_step:
                if train_config['clip_grad']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
                scaler.step(optimizer)  # this internally calls optimizer.step()
                scaler.update()

        else:
            loss.backward()
            if do_step:
                if train_config['clip_grad']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
                optimizer.step()  # step the optimizer manually when not using amp

        # advance the learning rate schedule *after* the optimizer has stepped
        if do_step and scheduler is not None:
            scheduler.step()

        # update accumulation counter (wraps to 0 at boundary)
        train_config['_ga_counter'] = (ga_counter + 1) % accum_steps


    # collect predictions
    y_pred_ids = y_pred_logits.argmax(dim=1)
    y_correct = y_pred_ids.eq(y_true_ids)
    if ignored_token is not None:
        y_correct = y_correct.masked_select(y_true_ids.ne(ignored_token)).detach().cpu()

    accuracy = y_correct.float().mean()

    batch_accumulator = Accumulator(
        correct=y_correct.sum().numpy(),
        total=y_correct.numel(),
    )

    # to numpy
    def _serializable(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return x

    # logs
    batch_logs['accuracy'] = _serializable(accuracy)
    if loss_fn is not None:
        batch_logs['loss'] = _serializable(raw_loss)
    if train:
        batch_logs['learning_rate'] = _serializable(optimizer.param_groups[0]['lr'])
        batch_logs['grad_norm'] = _serializable(calculate_grad_norm(model))

    return batch_logs, batch_accumulator
