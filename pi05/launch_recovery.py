"""Persistent correction collection, warm-start training and full-policy acceptance."""
import argparse
from collections import Counter
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
import urllib.request

import numpy as np

from pipeline import BASES, ROOT, fingerprint, validate_episode, write_json


def worker(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    run = spec_path.parent
    repo = spec['repo_id']
    checkpoint = Path(spec['init_checkpoint'])
    target_closure = spec.get('target_closure', False)
    rollout_repo = spec.get('rollout_repo_id', spec['source_repo_id'])
    rollout_checkpoint = spec.get('rollout_checkpoint', str(checkpoint))
    workers = 4
    per_worker = spec['corrections'] // workers
    eval_index = spec['validation_seed_range'][0] - BASES['validation']
    raw = ROOT/'data'/spec['experiment']
    corrections = ROOT/'data'/(spec['experiment']+'_corrections')
    sim = sys.executable
    openpi = str(ROOT/'vendor/openpi/.venv/bin/python')
    env = os.environ | dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                            MUJOCO_GL='egl', PYTHONUNBUFFERED='1', XLA_PYTHON_CLIENT_MEM_FRACTION='.30',
                            PYTHONPATH=str(ROOT/'vendor/openpi/packages/openpi-client/src')+':'+str(ROOT.parent/'work/pi05_setup/sim_client'))
    processes = []
    status = dict(spec, started_utc=datetime.now(timezone.utc).isoformat())
    def stage(name: str) -> None:
        status.update(stage=name, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(run/'job_status.json', status)
    def command(args: list[str], log: str, environment: dict | None = None) -> None:
        with (run/log).open('w') as out:
            subprocess.run(args, cwd=ROOT.parent, env=environment or env, stdout=out, stderr=subprocess.STDOUT, check=True)
    def stop() -> None:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        processes.clear()
    try:
        stage('collecting_corrections')
        corrections.mkdir(parents=True, exist_ok=False)
        with (run/'collection_server.log').open('w') as server_log, (run/'collection_viewer.log').open('w') as viewer_log:
            server = subprocess.Popen([openpi, str(ROOT/'openpi_run.py'), 'serve', '--repo-id', rollout_repo,
                    '--checkpoint', rollout_checkpoint, '--port', '8096'], cwd=ROOT.parent, env=env,
                    stdout=server_log, stderr=subprocess.STDOUT)
            processes.append(server)
            viewer = subprocess.Popen(['bash', str(ROOT.parent/'grasp/grasp.sh'), 'live', '--envs', '1',
                    '--stream', str(run/'collection_state.npz')], cwd=ROOT.parent,
                    stdout=viewer_log, stderr=subprocess.STDOUT)
            processes.append(viewer)
            for _ in range(240):
                if server.poll() is not None or viewer.poll() is not None:
                    raise RuntimeError('Collection server/viewer stopped')
                try:
                    urllib.request.urlopen('http://127.0.0.1:8096/healthz', timeout=1).close()
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('Collection server startup')
            logs = []
            collectors = []
            try:
                if target_closure:
                    stage('target_closure_pilot')
                    pilot = corrections/'pilot'
                    command([sim, str(ROOT/'collect_recovery.py'), '--target-closure',
                             '--count', '8', '--max-attempts', '24', '--start-index', '64400',
                             '--output', str(pilot), '--stream', str(run/'collection_state.npz')], 'pilot.log')
                    command([sim, str(ROOT/'check_recovery.py'), '--raw', str(pilot), '--samples', '4',
                             '--output', str(run/'pilot_replay.json')], 'pilot_replay.log')
                    stage('collecting_corrections')
                for idx in range(workers):
                    log = (run/f'collector_{idx}.log').open('w')
                    logs.append(log)
                    args = [sim, str(ROOT/'collect_recovery.py'), '--output', str(corrections/f'worker_{idx}'),
                            '--count', str(per_worker), '--max-attempts', str(per_worker*(3 if target_closure else 2)),
                            '--start-index', str((65400 if target_closure else 50000)+idx*1000)]
                    if target_closure:
                        args.append('--target-closure')
                    if idx == 0:
                        args += ['--stream', str(run/'collection_state.npz')]
                    process = subprocess.Popen(args, cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
                    collectors.append(process)
                    processes.append(process)
                while any(p.poll() is None for p in collectors):
                    if any(p.poll() not in (None, 0) for p in collectors):
                        raise RuntimeError('Correction collector failed; inspect collector logs')
                    if viewer.poll() is not None or server.poll() is not None:
                        raise RuntimeError('Collection viewer/server stopped')
                    time.sleep(5)
                if any(p.returncode for p in collectors):
                    raise RuntimeError('Correction collector failed')
            finally:
                stop()
                for log in logs:
                    log.close()
        stage('verifying_and_mixing_data')
        raw.mkdir(parents=True, exist_ok=False)
        recovered = sorted(corrections.glob('worker_*/episode_*'))
        if len(recovered) != spec['corrections']:
            raise RuntimeError(f"Expected {spec['corrections']} correction episodes")
        originals = sorted((ROOT/'data/train1000').glob('episode_*'))
        original_count = spec['original_successes']
        if len(originals) < original_count:
            raise RuntimeError('Incomplete original training set')
        selected = [originals[int(i)] for i in np.linspace(0, len(originals)-1, original_count, dtype=int)]
        seen = set()
        interventions = Counter()
        phases = Counter()
        for i,path in enumerate(recovered+selected):
            meta = validate_episode(path)
            if meta['split']!='train' or not meta['is_success'] or meta['fingerprint']!=fingerprint() or meta['seed'] in seen:
                raise RuntimeError('Invalid mixed training episode')
            if target_closure and i < len(recovered):
                interventions[meta['intervention']] += 1
                phases.update(meta['phase_frames'])
            seen.add(meta['seed'])
            (raw/f'episode_{i:06d}').symlink_to(path.resolve(), target_is_directory=True)
        if target_closure and (not all(phases[k] > 0 for k in ('align', 'secure', 'lift'))
                               or sum(interventions.values()) != len(recovered) or None in interventions):
            raise RuntimeError('Missing grasp phase or intervention labels')
        write_json(raw/'summary.json', dict(episodes=len(seen), corrections=len(recovered),
                   original_successes=len(selected), interventions=dict(interventions),
                   correction_phase_frames=dict(phases), split='train', fingerprint=fingerprint(),
                   unique_seeds=len(seen)))
        # Audit each correction worker and replay representative suffixes from actual saved states.
        for idx in range(workers):
            command([sim, str(ROOT/'check_recovery.py'), '--raw', str(corrections/f'worker_{idx}'),
                     '--samples', '3', '--output', str(run/f'replay_audit_{idx}.json')], f'replay_audit_{idx}.log')
        stage('converting_dataset')
        command([openpi, str(ROOT/'prepare_dataset.py'), '--raw', str(raw), '--repo-id', repo], 'conversion.log', env | {'JAX_PLATFORMS':'cpu'})
        assets = ROOT/'outputs/openpi_assets/pi05_realman_absolute12'
        source = assets/spec['source_repo_id']/'norm_stats.json'
        destination = assets/repo/'norm_stats.json'
        shutil.copy2(destination, run/'new_data_norm_stats_unused.json')
        shutil.copy2(source, destination)
        write_json(run/'normalization.json', dict(source=str(source), destination=str(destination),
                   sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), reason='Preserve normalization of warm-start weights'))
        command([openpi, str(ROOT/'openpi_run.py'), 'check-data', '--repo-id', repo, '--batch-size','4'], 'batch_check.log', env | {'JAX_PLATFORMS':'cpu'})
        train_env = env | dict(XLA_PYTHON_CLIENT_MEM_FRACTION='.55', PI05_INIT_CHECKPOINT=str(checkpoint), PI05_PEAK_LR=str(spec['peak_lr']))
        stage('warm_start_probe')
        command(['bash', str(ROOT/'train_visible.sh'), repo, spec['experiment']+'_probe', str(raw), '1', '4'], 'probe_job.log', train_env)
        stage('training')
        command(['bash', str(ROOT/'train_visible.sh'), repo, spec['experiment'], str(raw), str(spec['steps']), '4'], 'training_job.log', train_env)
        stage('validating_new_policy')
        final = ROOT/'checkpoints/pi05_realman_absolute12'/spec['experiment']/str(spec['steps']-1)
        command([sim, str(ROOT/'evaluate_checkpoint.py'), '--repo-id', repo, '--checkpoint', str(final),
                 '--count','100','--start-index',str(eval_index),'--execution','policy','--output',str(run/'validation')], 'validation_job.log')
        status['validation'] = json.loads((run/'validation/results/summary.json').read_text())
        stage('trained_and_validated')
    except BaseException as exc:
        status.update(failed_stage=status.get('stage'), error=repr(exc))
        stage('failed')
        raise
    finally:
        stop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--experiment', default='recovery_20261118')
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--target-closure', action='store_true')
    a = p.parse_args()
    if a.worker:
        worker(a.worker)
        return
    if not re.fullmatch('[A-Za-z0-9_-]+',a.experiment) or a.steps<=0:
        p.error('Invalid experiment/steps')
    gate = json.loads((ROOT/'data/recovery_pilot_20261118/summary.json').read_text())
    if not gate['complete'] or gate['accepted']<10 or gate['fingerprint']!=fingerprint():
        p.error('Need completed matching recovery pilot')
    run = ROOT/'runs'/a.experiment
    if (ROOT/'checkpoints/pi05_realman_absolute12'/a.experiment).exists():
        p.error('Refusing to overwrite checkpoint')
    spec = dict(experiment=a.experiment, steps=a.steps, repo_id='local/rm65_rh56_pi05_'+a.experiment,
                source_repo_id='local/rm65_rh56_pi05_train1000',
                init_checkpoint=str(ROOT/'checkpoints/pi05_realman_absolute12/baseline_20261117/29999'),
                batch_size=4, corrections=200, original_successes=200, peak_lr=1e-5,
                evaluation='new policy full task only, no baseline rerun', validation_seed_range=[110000,110099])
    if a.target_closure:
        spec.update(target_closure=True, corrections=100, original_successes=1000,
                    rollout_repo_id='local/rm65_rh56_pi05_recovery_20261118',
                    rollout_checkpoint=str(ROOT/'checkpoints/pi05_realman_absolute12/recovery_20261118/11999'),
                    validation_seed_range=[120000, 120099])
    spec['code_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip()
    if not (Path(spec.get('rollout_checkpoint', spec['init_checkpoint']))/'params').is_dir():
        p.error('Missing rollout checkpoint')
    run.mkdir(parents=True, exist_ok=False)
    write_json(run/'job_spec.json', spec)
    command = shlex.join([sys.executable,str(Path(__file__).resolve()),'--worker',str(run/'job_spec.json')])
    subprocess.run(['tmux','new-session','-d','-s','pi05-'+a.experiment,command],check=True)
    print(json.dumps(spec),flush=True)


if __name__ == '__main__':
    main()
