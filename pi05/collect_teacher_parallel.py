"""Run collect_teacher.py on disjoint train-seed chunks, then expose one convertible directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from pipeline import ROOT, fingerprint, validate_episode, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--start-index', type=int, required=True)
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--stream', type=Path)
    a = p.parse_args()
    if a.count <= 0 or not 1 <= a.workers <= 16 or a.count % a.workers:
        p.error('count must be a positive multiple of workers (1-16)')
    per, stride = a.count // a.workers, 1000
    if 2 * per > stride or a.start_index + a.workers * stride > 90000:
        p.error('Worker seed ranges overlap or exceed the train partition')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ | {'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

    def worker(i: int) -> None:
        cmd = [sys.executable, str(ROOT / 'collect_teacher.py'), '--mode', 'collect', '--noise',
               '--output', str(out / 'chunks' / f'worker_{i:02d}'), '--count', str(per),
               '--max-attempts', str(2 * per), '--start-index', str(a.start_index + i * stride),
               '--gate', str(a.gate.resolve())]
        if i == 0 and a.stream:
            cmd += ['--stream', str(a.stream.resolve())]
        with (out / f'worker_{i:02d}.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    with ThreadPoolExecutor(a.workers) as pool:
        jobs = {pool.submit(worker, i): i for i in range(a.workers)}
        for job in as_completed(jobs):
            job.result()
            print(f'completed worker {jobs[job]}', flush=True)
    sources = sorted((out / 'chunks').glob('worker_*/episode_*'))
    seen, sigmas, attempts = set(), {}, 0
    for i, path in enumerate(sources):
        meta = validate_episode(path)
        if meta['seed'] in seen or meta['fingerprint'] != fingerprint() or meta['split'] != 'train' or not meta['is_success']:
            raise RuntimeError(f'Invalid or duplicated demonstration: {path}')
        seen.add(meta['seed'])
        sigmas[str(meta['noise_sigma_rad'])] = sigmas.get(str(meta['noise_sigma_rad']), 0) + 1
        (out / f'episode_{i:06d}').symlink_to(path.relative_to(out), target_is_directory=True)
    for f in (out / 'chunks').glob('worker_*/attempts.jsonl'):
        attempts += len(f.read_text().splitlines())
    if len(sources) != a.count:
        raise RuntimeError(f'Expected {a.count} episodes, got {len(sources)}')
    write_json(out / 'summary.json', dict(episodes=len(sources), accepted=len(sources), attempts=attempts,
               requested=a.count, split='train', fingerprint=fingerprint(), workers=a.workers,
               unique_seeds=len(seen), noise_sigma_counts=sigmas, teacher_gate=str(a.gate.resolve()),
               teacher='pi05/teacher.py', complete=True))
    print(json.dumps(dict(complete=True, episodes=len(sources), attempts=attempts, sigmas=sigmas)), flush=True)


if __name__ == '__main__':
    main()
