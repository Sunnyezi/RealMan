"""Collect successful image/action correction suffixes after learned-policy prefixes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pipeline import BASES, GRASP, PROMPT, RGBEnv, fingerprint, write_json
from recovery import RecoveryExpert, save_initial


def pregrasp_trigger(palm: np.ndarray, grasp: np.ndarray, index_target: float,
                     near_wait: int, steps: int, deadline: int) -> tuple[str | None, int]:
    """Intervene on a missed closure before the hand finishes descending."""
    lateral = np.linalg.norm(palm[:2]-grasp[:2])
    vertical = palm[2]-grasp[2]
    near = lateral <= .022 and -.023 <= vertical <= .018
    approaching = lateral <= .028 and -.023 <= vertical <= .040
    if index_target >= .12 and not near:
        return 'closure_before_alignment', 0
    near_wait = near_wait+1 if approaching and index_target < .12 else 0
    if near_wait >= 3 or (near and index_target < .12):
        return 'near_without_closure', near_wait
    if steps >= deadline and not near:
        return 'approach_not_aligned', near_wait
    return None, near_wait


def main() -> None:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=200)
    p.add_argument('--max-attempts', type=int, default=400)
    p.add_argument('--start-index', type=int, default=50000)
    p.add_argument('--port', type=int, default=8096)
    p.add_argument('--stream', type=Path)
    p.add_argument('--target-closure', action='store_true')
    args = p.parse_args()
    if args.count <= 0 or args.max_attempts < args.count or args.start_index < 0 or args.start_index+args.max_attempts > 90000:
        p.error('Invalid training seed/count range')
    args.output.mkdir(parents=True, exist_ok=False)
    policy = WebsocketClientPolicy(port=args.port)
    if policy.get_server_metadata().get('action_dim') != 12:
        raise ValueError('Need action_dim=12')
    env = RGBEnv(args.stream)
    regions = json.loads((GRASP/'sensor_map.json').read_text())
    thumb = np.array([i for i,r in enumerate(regions) if r['region'].startswith('right_thumb_')])
    index = np.array([i for i,r in enumerate(regions) if r['region'].startswith('right_index_')])
    accepted = 0
    rows = []
    source_hash = hashlib.sha256(Path(__file__).read_bytes()+b''.join(Path(__file__).with_name(n).read_bytes() for n in ('recovery.py','stabilizer.py','evaluate_stabilized.py'))).hexdigest()
    try:
        with (args.output/'attempts.jsonl').open('w') as log:
            for attempt in range(args.max_attempts):
                seed = BASES['train']+args.start_index+attempt
                env.reset(seed)
                prefix = []
                frame_limit = ((68, 74, 80)[attempt%3] if args.target_closure
                               else (30,60,90,120,140)[attempt%5])
                e = env.base
                visual_grasp = e.vision.last + np.asarray(e.cfg['grasp_xyz'])-np.asarray(e.cfg['object_xyz'])
                trigger = None
                near_wait = 0
                had_touch = False
                started = time.monotonic()
                for step in range(frame_limit):
                    if step%2 == 0:
                        chunk = np.asarray(policy.infer(env.observation())['actions'])
                        if chunk.ndim != 2 or chunk.shape[1] != 12 or len(chunk)<2 or not np.isfinite(chunk).all():
                            raise ValueError('Invalid policy actions')
                    action = chunk[step%2].copy()
                    prefix.append(action)
                    _, info = env.step(action)
                    if args.target_closure:
                        had_touch |= info['hand_contacts'] > 0
                        if info['physics_failure'] or info['dropped'] or info['is_success'] or info['hand_contacts'] >= 2:
                            break
                        trigger, near_wait = pregrasp_trigger(
                            e.d.xpos[e.bid], visual_grasp, e.d.ctrl[env.ids[8]],
                            near_wait, step+1, frame_limit)
                        if info['table_collision'] and trigger is None:
                            trigger = 'ground_contact_before_closure'
                        if trigger is not None:
                            break
                    elif info['is_success'] or info['physics_failure']:
                        break
                prefix_metrics = dict(info)
                snapshot = args.output/'pending_initial.npz'
                save_initial(env, snapshot)
                e.steps, e.hold, e.max_height = 0, 0., 0.
                frames = []
                kind = None
                phase_frames = Counter()
                error = None
                try:
                    if args.target_closure and trigger is None:
                        raise ValueError('No pre-grasp intervention detected')
                    # Preserve the approach when open and clear; otherwise the expert retreats and refreshes RGB-D.
                    can_continue = bool(args.target_closure and not had_touch and not info['table_collision']
                                        and e.d.ctrl[env.ids[8]] < .12
                                        and e.d.xpos[e.bid, 2] >= visual_grasp[2]-.023)
                    expert = RecoveryExpert(env, thumb, index, direct_align=can_continue)
                    kind = expert.kind
                    for _ in range(300):
                        obs = env.observation()
                        phase = expert.phase if expert.phase == 'release' else expert.control.phase
                        action = expert.update().astype(np.float32)
                        phase_frames[phase] += 1
                        frames.append({'image': obs['observation/image'], 'wrist_image': obs['observation/wrist_image'],
                                       'state': obs['observation/state'], 'actions': action, 'timestamp': e.steps*.05})
                        _, info = env.step(action)
                        env.publish('correction expert / '+expert.phase+' / '+expert.control.phase)
                        if info['is_success'] or info['physics_failure'] or (args.target_closure and info['dropped']):
                            break
                except ValueError as exc:
                    error = str(exc)
                    info = dict(info, is_success=False)
                row = dict(seed=seed, split='train', prompt=PROMPT, fingerprint=fingerprint(),
                           source='policy_visited_state_expert_correction', correction_kind=kind,
                           prefix_steps=len(prefix), prefix_metrics=prefix_metrics, recovery_frames=len(frames),
                           intervention=trigger, phase_frames=dict(phase_frames), source_sha256=source_hash,
                           wall_seconds=time.monotonic()-started, error=error, **info)
                row['is_success'] = bool(row['is_success'] and len(frames)>=10 and error is None)
                if args.target_closure:
                    row['is_success'] &= bool(trigger and phase_frames['align'] and phase_frames['secure']
                                              and phase_frames['lift'] and (phase_frames['lift']+phase_frames['hold'] >= 40))
                rows.append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
                if row['is_success']:
                    dest = args.output/f'episode_{accepted:06d}'
                    dest.mkdir()
                    snapshot.replace(dest/'initial_state.npz')
                    np.savez_compressed(dest/'prefix.npz', actions=np.asarray(prefix))
                    np.savez_compressed(dest/'trajectory.npz', **{k:np.stack([f[k] for f in frames]) for k in frames[0]})
                    write_json(dest/'metadata.json', row)
                    accepted += 1
                else:
                    snapshot.unlink(missing_ok=True)
                print(json.dumps(dict(attempt=attempt, seed=seed, accepted=accepted, success=row['is_success'], kind=kind, intervention=trigger, error=error)), flush=True)
                write_json(args.output/'summary.json', dict(episodes=len(rows), accepted=accepted, requested=args.count,
                    successes=accepted, success_rate=accepted/len(rows), split='train', fingerprint=fingerprint(),
                    source_sha256=source_hash, complete=accepted==args.count))
                if accepted == args.count:
                    break
    finally:
        env.close()
    if accepted != args.count:
        raise RuntimeError(f'Collected {accepted}/{args.count} corrections')


if __name__ == '__main__':
    main()
