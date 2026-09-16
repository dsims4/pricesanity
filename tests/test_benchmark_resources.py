import torch

from pricesanity.benchmark.resources import controlled_thread_budget, hardware_fingerprint


def test_thread_budget_is_scoped_and_hardware_metadata_records_it() -> None:
    original = torch.get_num_threads()
    with controlled_thread_budget(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == original

    fingerprint = hardware_fingerprint(device="cpu", cpu_worker_count=1)
    assert fingerprint["cpu_worker_count"] == 1
    assert fingerprint["data_loader_worker_count"] == 0
    assert fingerprint["device"] == "cpu"
    assert fingerprint["logical_cpu_count"]
    assert fingerprint["backend"] == "CPU"
