import logging
import os
import torch

from typing import Type, Set
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, Future

from template_experiment.experiments import runnable_experiments

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
logger = logging.getLogger(__name__)


def run_experiment(exp_class: Type, gpu: int) -> None:
    os.environ[CUDA_VISIBLE_DEVICES] = str(gpu)
    exp_class().run()


def main():
    assert isinstance(runnable_experiments, list)
    cuda_devices = os.environ.get(CUDA_VISIBLE_DEVICES, None)
    if cuda_devices is None:
        raise Exception(f"{CUDA_VISIBLE_DEVICES} not set")

    n_gpus = torch.cuda.device_count()
    cuda_ids = set(map(int, cuda_devices.split(",")))
    assert len(cuda_ids) == n_gpus

    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futs: Set[Future] = set()
        for exp in runnable_experiments:
            if not cuda_ids:
                done_futs, futs = wait(futs, return_when=FIRST_COMPLETED)
                cuda_ids |= set(map(lambda df: df.cuda_id, done_futs))

            a_cuda_id = cuda_ids.pop()
            a_fut = executor.submit(run_experiment(exp, a_cuda_id))
            a_fut.cuda_id = a_cuda_id
            futs.add(a_fut)


if __name__ == "__main__":
    main()
