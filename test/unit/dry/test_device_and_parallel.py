import experiments.grid_runs_utils as gru


class _FakeCuda:
    """Deterministic stand-in for torch.cuda, so results don't depend on the host."""
    def __init__(self, device_count):
        self._device_count = device_count

    def is_available(self):
        return self._device_count > 0

    def device_count(self):
        return self._device_count


def _patch_cuda(monkeypatch, device_count):
    fake = _FakeCuda(device_count)
    monkeypatch.setattr(gru.torch, 'cuda', fake)
    return fake


def test_select_device_normalization_with_two_gpus(monkeypatch):
    _patch_cuda(monkeypatch, device_count=2)
    assert gru._select_device(None) == 'cuda:0'
    assert gru._select_device(1) == 'cuda:1'
    assert gru._select_device('1') == 'cuda:1'
    assert gru._select_device('cuda:1') == 'cuda:1'
    assert gru._select_device('CPU') == 'cpu'
    assert gru._select_device('cuda:7') == 'cpu'   # out of range -> cpu
    assert gru._select_device('nonsense') == 'cpu'
    assert gru._select_device('cuda:x') == 'cpu'


def test_select_device_without_cuda(monkeypatch):
    _patch_cuda(monkeypatch, device_count=0)
    assert gru._select_device(None) == 'cpu'
    assert gru._select_device(0) == 'cpu'
    assert gru._select_device('cuda:0') == 'cpu'


def test_parallel_settings_auto_devices_and_dedupe(monkeypatch):
    _patch_cuda(monkeypatch, device_count=2)
    run_config = {
        'wandb': {'activate': False},
        'parallel': {'devices_to_use': None, 'num_processes_per_device': 4,
                     'num_cpu_threads_per_process': 1},
    }
    devices, procs_per_device, threads = gru.get_parallel_settings(run_config)
    assert devices == ['cuda:0', 'cuda:1']
    assert procs_per_device == 4 and threads == 1

    run_config['parallel']['devices_to_use'] = ['cuda:1', 1, '1']
    devices, _, _ = gru.get_parallel_settings(run_config)
    assert devices == ['cuda:1']  # deduped


def test_parallel_settings_sequential_fallback(monkeypatch):
    _patch_cuda(monkeypatch, device_count=0)
    run_config = {'wandb': {'activate': False}, 'runtime': {'device': None}}
    devices, procs_per_device, threads = gru.get_parallel_settings(run_config)
    assert devices == ['cpu'] and procs_per_device == 1 and threads == 1


def test_wandb_api_key_detection(monkeypatch, tmp_path):
    monkeypatch.setenv('WANDB_API_KEY', 'k')
    assert gru._wandb_api_key_available()
    monkeypatch.delenv('WANDB_API_KEY')
    # point NETRC at an empty file -> no credentials
    netrc_path = tmp_path / 'netrc'
    netrc_path.write_text('')
    monkeypatch.setenv('NETRC', str(netrc_path))
    assert not gru._wandb_api_key_available()
    netrc_path.write_text('machine api.wandb.ai\nlogin user\npassword secret\n')
    assert gru._wandb_api_key_available()
