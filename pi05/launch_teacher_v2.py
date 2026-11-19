"""Persistent conversion and warm-start fine-tuning on teacher_v2 demonstrations (VNC-visible training)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from pipeline import ROOT, fingerprint, write_json

SOURCE_NORM_REPO = 'local/rm65_rh56_pi05_train1000'


def worker(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    run, repo, raw = spec_path.parent, spec['repo_id'], Path(spec['raw'])
    openpi = str(ROOT / 'vendor/openpi/.venv/bin/python')
    env = os.environ | dict(OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    status = dict(spec, started_utc=datetime.now(timezone.utc).isoformat())

    def stage(name: str) -> None:
        status.update(stage=name, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(run / 'job_status.json', status)

    def command(args: list[str], log: str, environment: dict | None = None) -> None:
        with (run / log).open('w') as out:
            subprocess.run(args, cwd=ROOT.parent, env=environment or env, stdout=out, stderr=subprocess.STDOUT, check=True)
    try:
        stage('checking_data')
        summary = json.loads((raw / 'summary.json').read_text())
        if not (summary.get('complete') and summary['fingerprint'] == fingerprint() and summary['split'] == 'train'
                and summary['unique_seeds'] == summary['episodes'] == spec['episodes']):
            raise RuntimeError('Incomplete or mismatched teacher_v2 collection')
        if not spec.get('skip_conversion'):
            stage('converting_dataset')
            command([openpi, str(ROOT / 'prepare_dataset.py'), '--raw', str(raw), '--repo-id', repo], 'conversion.log',
                    env | {'JAX_PLATFORMS': 'cpu'})
            assets = ROOT / 'outputs/openpi_assets/pi05_realman_absolute12'
            source, destination = assets / SOURCE_NORM_REPO / 'norm_stats.json', assets / repo / 'norm_stats.json'
            shutil.copy2(destination, run / 'new_data_norm_stats_unused.json')
            shutil.copy2(source, destination)
            write_json(run / 'normalization.json', dict(source=str(source), destination=str(destination),
                       sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                       reason='Preserve normalization of the warm-start checkpoint'))
            command([openpi, str(ROOT / 'openpi_run.py'), 'check-data', '--repo-id', repo, '--batch-size', '4'],
                    'batch_check.log', env | {'JAX_PLATFORMS': 'cpu'})
        train_env = env | dict(XLA_PYTHON_CLIENT_MEM_FRACTION=str(spec['mem_fraction']),
                               PI05_INIT_CHECKPOINT=spec['init_checkpoint'], PI05_PEAK_LR=str(spec['peak_lr']))
        stage('waiting_for_gpu_memory')
        need = spec['mem_fraction'] * 49140 + 1024
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).split()[0])
            if free >= need:
                break
            time.sleep(30)
        stage('warm_start_probe')
        command(['bash', str(ROOT / 'train_visible.sh'), repo, spec['experiment'] + '_probe', str(raw), '1', '4'],
                'probe_job.log', train_env)
        stage('training')
        command(['bash', str(ROOT / 'train_visible.sh'), repo, spec['experiment'], str(raw), str(spec['steps']), '4'],
                'training_job.log', train_env)
        stage('trained')
    except BaseException as exc:
        status.update(failed_stage=status.get('stage'), error=repr(exc))
        stage('failed')
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--experiment', default='teacher_v2_20261119')
    p.add_argument('--raw', type=Path, default=ROOT / 'data/teacher_v2_20261119')
    p.add_argument('--episodes', type=int, default=1200)
    p.add_argument('--steps', type=int, default=16000)
    p.add_argument('--peak-lr', type=float, default=1.5e-5)
    p.add_argument('--mem-fraction', type=float, default=.55)
    p.add_argument('--skip-conversion', action='store_true')
    a = p.parse_args()
    if a.worker:
        worker(a.worker)
        return
    if not re.fullmatch('[A-Za-z0-9_-]+', a.experiment) or a.steps <= 0:
        p.error('Invalid experiment/steps')
    if (ROOT / 'checkpoints/pi05_realman_absolute12' / a.experiment).exists():
        p.error('Refusing to overwrite checkpoint')
    run = ROOT / 'runs' / a.experiment
    run.mkdir(parents=True, exist_ok=True)
    spec = dict(experiment=a.experiment, steps=a.steps, repo_id='local/rm65_rh56_pi05_' + a.experiment,
                raw=str(a.raw.resolve()), episodes=a.episodes, batch_size=4, peak_lr=a.peak_lr, mem_fraction=a.mem_fraction,
                init_checkpoint=str(ROOT / 'checkpoints/pi05_realman_absolute12/baseline_20261117/29999'),
                normalization=SOURCE_NORM_REPO, skip_conversion=a.skip_conversion,
                teacher='pi05/teacher.py via pi05/collect_teacher.py',
                code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip())
    write_json(run / 'job_spec.json', spec)
    cmd = shlex.join([sys.executable, str(Path(__file__).resolve()), '--worker', str(run / 'job_spec.json')])
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'pi05-' + a.experiment, cmd], check=True)
    print(json.dumps(spec), flush=True)


if __name__ == '__main__':
    main()
