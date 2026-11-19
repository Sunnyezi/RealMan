"""Persistent pick-and-place pipeline: convert, check, warm-start probe, VNC-visible training, validation.

Runs in tmux.  Training uses pi05/train_visible.sh with PI05_TASK=place, so the VNC viewer on :15 shows
labelled place demonstrations during supervised updates.  After training, the final checkpoint is evaluated
on the validation seeds with plain chunk execution and with temporal ensembling (pure policy, no takeover).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from pipeline import ROOT, write_json
from place_task import place_fingerprint

SIM_PYTHON = str(Path.home() / '.conda/envs/realman-mujoco/bin/python')


def worker(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    run, repo, raw = spec_path.parent, spec['repo_id'], Path(spec['raw'])
    openpi = str(ROOT / 'vendor/openpi/.venv/bin/python')
    env = os.environ | dict(OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1', PI05_TASK='place')
    status = dict(spec, started_utc=datetime.now(timezone.utc).isoformat())

    def stage(name: str, **extra) -> None:
        status.update(stage=name, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(run / 'job_status.json', status)

    def command(args: list[str], log: str, environment: dict | None = None) -> None:
        with (run / log).open('w') as out:
            subprocess.run(args, cwd=ROOT.parent, env=environment or env, stdout=out, stderr=subprocess.STDOUT, check=True)

    def wait_gpu(fraction: float) -> None:
        need = fraction * 49140 + 1024
        while int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                          text=True).split()[0]) < need:
            time.sleep(30)
    try:
        stage('checking_data')
        summary = json.loads((raw / 'summary.json').read_text())
        if not (summary.get('complete') and summary['fingerprint'] == place_fingerprint() and summary['split'] == 'train'
                and summary['task'] == 'place' and summary['unique_seeds'] == summary['accepted'] == spec['episodes']):
            raise RuntimeError('Incomplete or mismatched place collection')
        if not spec.get('skip_conversion'):
            stage('converting_dataset')
            command([openpi, str(ROOT / 'prepare_dataset.py'), '--raw', str(raw), '--repo-id', repo], 'conversion.log',
                    env | {'JAX_PLATFORMS': 'cpu'})
            norm = ROOT / 'outputs/openpi_assets/pi05_realman_absolute12' / repo / 'norm_stats.json'
            write_json(run / 'normalization.json', dict(path=str(norm), sha256=hashlib.sha256(norm.read_bytes()).hexdigest(),
                       source='computed from this place dataset (new task, new action distribution)'))
            command([openpi, str(ROOT / 'openpi_run.py'), 'check-data', '--repo-id', repo, '--batch-size', '4'],
                    'batch_check.log', env | {'JAX_PLATFORMS': 'cpu'})
        train_env = env | dict(XLA_PYTHON_CLIENT_MEM_FRACTION=str(spec['mem_fraction']),
                               PI05_INIT_CHECKPOINT=spec['init_checkpoint'], PI05_PEAK_LR=str(spec['peak_lr']))
        if not spec.get('skip_training'):
            stage('waiting_for_gpu_memory')
            wait_gpu(spec['mem_fraction'])
            stage('warm_start_probe')
            command(['bash', str(ROOT / 'train_visible.sh'), repo, spec['experiment'] + '_probe', str(raw), '1', '4'],
                    'probe_job.log', train_env)
            stage('training')
            command(['bash', str(ROOT / 'train_visible.sh'), repo, spec['experiment'], str(raw), str(spec['steps']), '4'],
                    'training_job.log', train_env)
        checkpoint = ROOT / 'checkpoints/pi05_realman_absolute12' / spec['experiment'] / str(spec['steps'] - 1)
        results = {}
        for name, extra in (('ens001', ['--ensemble-decay', '0.01']), ('plain', [])):
            stage('evaluating_' + name, evaluations=results)
            wait_gpu(.30)
            out = ROOT / 'runs' / (spec['experiment'] + '_eval') / f"{checkpoint.name}_{name}_v{spec['val_start']}"
            command([SIM_PYTHON, str(ROOT / 'evaluate_checkpoint.py'), '--task', 'place', '--execution', 'policy',
                     '--repo-id', repo, '--checkpoint', str(checkpoint), '--output', str(out),
                     '--split', 'validation', '--start-index', str(spec['val_start']), '--count', '100',
                     '--port', '8096', *extra], f'eval_{name}.log', env)
            s = json.loads((out / 'results/summary.json').read_text())
            results[name] = dict(output=str(out), successes=s['successes'], episodes=s['episodes'],
                                 success_95ci=s['success_95ci'], failures=s.get('failures'))
        stage('evaluated', evaluations=results)
    except BaseException as exc:
        status.update(failed_stage=status.get('stage'), error=repr(exc))
        stage('failed')
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--experiment', default='place_20261119')
    p.add_argument('--raw', type=Path, default=ROOT / 'data/place_20261119')
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--peak-lr', type=float, default=2e-5)
    p.add_argument('--mem-fraction', type=float, default=.55)
    p.add_argument('--val-start', type=int, default=40000, help='validation index; seeds 100000+index')
    p.add_argument('--skip-conversion', action='store_true')
    p.add_argument('--skip-training', action='store_true')
    a = p.parse_args()
    if a.worker:
        worker(a.worker)
        return
    if not re.fullmatch('[A-Za-z0-9_-]+', a.experiment) or a.steps <= 0:
        p.error('Invalid experiment/steps')
    if not a.skip_training and (ROOT / 'checkpoints/pi05_realman_absolute12' / a.experiment).exists():
        p.error('Refusing to overwrite checkpoint')
    run = ROOT / 'runs' / a.experiment
    run.mkdir(parents=True, exist_ok=True)
    spec = dict(experiment=a.experiment, task='place', steps=a.steps, repo_id='local/rm65_rh56_pi05_' + a.experiment,
                raw=str(a.raw.resolve()), episodes=a.episodes, batch_size=4, peak_lr=a.peak_lr,
                mem_fraction=a.mem_fraction, val_start=a.val_start,
                init_checkpoint=str(ROOT / 'checkpoints/pi05_realman_absolute12/baseline_20261117/29999'),
                skip_conversion=a.skip_conversion, skip_training=a.skip_training,
                teacher='pi05/place_teacher.py via pi05/collect_place.py',
                code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip())
    write_json(run / 'job_spec.json', spec)
    cmd = shlex.join([sys.executable, str(Path(__file__).resolve()), '--worker', str(run / 'job_spec.json')])
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'pi05-' + a.experiment, cmd], check=True)
    print(json.dumps(spec), flush=True)


if __name__ == '__main__':
    main()
