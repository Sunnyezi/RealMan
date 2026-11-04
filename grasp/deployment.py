"""Analyze recorded hardware signals, replay a policy offline, or score trials."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from benchmark import wilson
from protocol import fingerprint

# In order, matching GraspEnv._obs. A hardware adapter must supply calibrated values.
FIELDS = (
    ('robot_qpos', 19), ('robot_qvel', 19), ('palm_xyz_m', 3),
    ('palm_rotation_row_major', 9), ('palm_world_twist_angular_linear', 6),
    ('object_minus_palm_m', 3), ('vision_valid', 1), ('vision_age_s', 1),
    ('previous_action', 9), ('target_xyz_m', 3), ('hand_target_rad', 6),
    ('touch_force_N', 17), ('touch_contact', 17), ('grasp_error_m', 3),
    ('lift_error_m', 3), ('trajectory_valid', 1),
)
OBSERVATION_SIZE = sum(size for _, size in FIELDS)
ACTION_SIZE = 9
CONTROL_DT_S = .05
STREAMS = ('joints', 'rgbd', 'touch')
HAND_ACTUATOR_QPOS = [7, 8, 11, 13, 15, 17]


def vector(value, size, field):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{field} must have {size} finite values')
    return result


def observation(fields):
    parts = []
    for key, size in FIELDS:
        part = vector(fields[key], size, key)
        if key in {'vision_valid', 'touch_contact', 'trajectory_valid'} and not np.all(np.isin(part, [0, 1])):
            raise ValueError(f'{key} must contain binary values')
        if key == 'touch_force_N':
            if np.any(part < 0):
                raise ValueError('Touch force must be nonnegative after calibration')
            part = part / 10.
        parts.append(part)
    result = np.concatenate(parts)
    if np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError('Observation exceeds float32 range')
    return result.astype(np.float32)


def safety_reasons(sample, proposal, limits):
    """Check a proposal already vetted by a workcell collision/clearance checker."""
    reasons = []
    now = sample['timestamp_ns']
    if sample['estop'] or sample['controller_fault']:
        reasons.append('controller_stop')
    if not sample['observation']['vision_valid'][0] or not sample['observation']['trajectory_valid'][0]:
        reasons.append('invalid_perception_or_path')
    for stream in STREAMS:
        age = (now - sample['streams'][stream]['timestamp_ns']) / 1e9
        if not 0 <= age <= limits['max_age_s'][stream]:
            reasons.append(f'{stream}_stale_or_unsynchronized')
    if proposal.get('collision_free') is not True:
        reasons.append('collision_check_failed')
    clearance = float(proposal['min_ground_clearance_m'])
    if not math.isfinite(clearance) or clearance < limits['min_ground_clearance_m']:
        reasons.append('ground_clearance')
    q = vector(proposal['arm_q_rad'], 6, 'arm_q_rad')
    q_measured = vector(sample['observation']['robot_qpos'], 19, 'robot_qpos')
    hand = vector(proposal['hand_q_rad'], 6, 'hand_q_rad')
    xyz = vector(proposal['target_xyz_m'], 3, 'target_xyz_m')
    dt = float(proposal['duration_s'])
    for name, values in [('arm', q), ('hand', hand), ('xyz', xyz)]:
        if np.any(values < limits[f'{name}_min']) or np.any(values > limits[f'{name}_max']):
            reasons.append(f'{name}_limits')
    if not math.isfinite(dt) or dt <= 0 or np.any(np.abs(q-q_measured[:6]) > np.asarray(limits['arm_max_speed_rad_s']) * dt):
        reasons.append('arm_speed')
    hand_prev = q_measured[HAND_ACTUATOR_QPOS]
    if not math.isfinite(dt) or dt <= 0 or np.any(np.abs(hand-hand_prev) > np.asarray(limits['hand_max_speed_rad_s']) * dt):
        reasons.append('hand_speed')
    return reasons


def load_limits(path):
    limits = json.loads(path.read_text())
    for group, size in (('arm', 6), ('hand', 6), ('xyz', 3)):
        low = vector(limits[f'{group}_min'], size, f'{group}_min')
        high = vector(limits[f'{group}_max'], size, f'{group}_max')
        if np.any(low >= high):
            raise ValueError(f'Invalid {group} limits')
    for group in ('arm', 'hand'):
        if np.any(vector(limits[f'{group}_max_speed_rad_s'], 6, f'{group}_max_speed_rad_s') <= 0):
            raise ValueError(f'Invalid {group} speed limits')
    positive = [limits['min_ground_clearance_m'], *(limits['max_age_s'][name] for name in STREAMS)]
    if not np.all(np.isfinite(positive)) or any(value <= 0 for value in positive):
        raise ValueError('Clearance and freshness limits must be positive and finite')
    return limits


def read_rows(path):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError('Recording is empty')
    return rows


def validate_recording(rows):
    times = [row['timestamp_ns'] for row in rows]
    if any(type(t) is not int or t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Recording timestamps must be increasing integer nanoseconds')
    previous = {}
    for row in rows:
        observation(row['observation'])
        for name in STREAMS:
            record = row['streams'][name]
            timestamp, sequence = record['timestamp_ns'], record['sequence']
            if (type(timestamp) is not int or not 0 <= timestamp <= row['timestamp_ns']
                    or type(sequence) is not int or sequence < 0):
                raise ValueError(f'Invalid clock or sequence for {name}')
            if name in previous:
                old_time, old_seq = previous[name]
                if sequence < old_seq or timestamp < old_time or (sequence == old_seq and timestamp != old_time):
                    raise ValueError(f'Nonmonotonic frame metadata for {name}')
            previous[name] = timestamp, sequence


def distribution(values):
    if not len(values):
        return None
    return dict(zip(('p05', 'p50', 'p95'), np.quantile(values, [.05, .5, .95], axis=0).tolist()))


def calibration(rows):
    validate_recording(rows)
    ages, sequences = {}, {}
    for name in STREAMS:
        samples = [row['streams'][name] for row in rows]
        ages[name] = distribution([(row['timestamp_ns'] - sample['timestamp_ns']) / 1e9
                                   for row, sample in zip(rows, samples)])
        unique = sorted({sample['sequence'] for sample in samples})
        sequences[name] = dict(skipped_sequence_ids=unique[-1] - unique[0] + 1 - len(unique),
                               recorded_frames=len(unique))
    tracking = []
    touch_zero = []
    latency = []
    for row in rows:
        fields = row['observation']
        if 'commanded_arm_q_rad' in row:
            target = vector(row['commanded_arm_q_rad'], 6, 'commanded_arm_q_rad')
            tracking.append(np.asarray(fields['robot_qpos'][:6]) - target)
        if row.get('no_contact') is True:
            touch_zero.append(fields['touch_force_N'])
        if 'command_sent_ns' in row and 'command_ack_ns' in row:
            sent, ack = row['command_sent_ns'], row['command_ack_ns']
            if type(sent) is not int or type(ack) is not int or not 0 <= sent <= ack <= row['timestamp_ns']:
                raise ValueError('Invalid command timestamp pair')
            latency.append((ack - sent) / 1e9)
    return dict(rows=len(rows), observation_age_s=ages, stream_sequence_coverage=sequences,
                arm_tracking_error_rad=distribution(tracking),
                touch_no_contact_force_N=distribution(touch_zero),
                command_ack_latency_s=distribution(latency),
                control_period_s=distribution(np.diff([row['timestamp_ns'] for row in rows]) / 1e9))


def replay(rows, checkpoint, limits):
    validate_recording(rows)
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    manifest = json.loads((checkpoint.parent / 'run.json').read_text())
    if manifest['fingerprint'] != fingerprint():
        raise ValueError('Checkpoint does not match the current floor scene')
    torch.set_num_threads(1)
    class ReplaySpace(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), np.float32)
        action_space = gym.spaces.Box(-1., 1., (ACTION_SIZE,), np.float32)

    base = DummyVecEnv([ReplaySpace])
    normalizer = None
    try:
        normalizer = VecNormalize.load(str(checkpoint.with_name(checkpoint.stem + '_vecnormalize.pkl')), base)
        normalizer.training = False
        policy = PPO.load(str(checkpoint), device='cpu')
        results = []
        for row in rows:
            obs = observation(row['observation'])
            start = time.perf_counter()
            action, _ = policy.predict(normalizer.normalize_obs(obs[None, :]), deterministic=True)
            elapsed = time.perf_counter() - start
            action = vector(action[0], ACTION_SIZE, 'action')
            fields = row['observation']
            desired_xyz = np.asarray(fields['target_xyz_m']) + .005*np.clip(action[:3], -1., 1.)
            desired_hand = np.asarray(fields['hand_target_rad']) + .06*np.clip(action[3:], -1., 1.)
            recorded = row.get('recorded_proposal')
            results.append(dict(timestamp_ns=row['timestamp_ns'], action=action.tolist(), inference_s=elapsed,
                                desired_xyz_m=desired_xyz.tolist(), desired_hand_rad=desired_hand.tolist(),
                                recorded_proposal_rejections=safety_reasons(row, recorded, limits)
                                if recorded is not None else None))
        return dict(checkpoint=str(checkpoint), rows=len(results), inference_s=distribution([r['inference_s'] for r in results]),
                    inference_over_50_ms=sum(r['inference_s'] > CONTROL_DT_S for r in results), results=results)
    finally:
        if normalizer is None:
            base.close()
        else:
            normalizer.close()


def scenarios(center, radius, lighting, seed):
    center = vector(center, 2, 'center_xy_m')
    if not math.isfinite(radius) or radius <= 0 or not lighting or len(set(lighting)) != len(lighting):
        raise ValueError('Positive radius and distinct lighting labels are required')
    rng = np.random.default_rng(seed)
    rows = []
    for ix, x in enumerate(np.linspace(-radius, radius, 5)):
        for iy, y in enumerate(np.linspace(-radius, radius, 5)):
            for iyaw, yaw in enumerate((-33.75, -11.25, 11.25, 33.75)):
                index = len(rows)
                rows.append(dict(scenario_id=index, xy_m=(center + [x, y]).tolist(),
                                 yaw_deg=yaw, lighting=lighting[(ix+iy+iyaw) % len(lighting)]))
    order = rng.permutation(len(rows))
    return dict(seed=seed, scenarios=[rows[index] for index in order])


def acceptance(rows, target=90, scenario_ids=None):
    """Score pre-declared scenarios; count drops and protective stops as failures."""
    groups = {}
    for row in rows:
        candidate = row['candidate']
        scenario = row['scenario_id']
        bucket = groups.setdefault(candidate, {})
        if scenario in bucket:
            raise ValueError(f'Duplicate scenario {scenario} for {candidate}')
        lift, hold, cycle = (float(row[name]) for name in ('held_height_m', 'hold_s', 'cycle_s'))
        if not np.all(np.isfinite([lift, hold, cycle])) or min(lift, hold, cycle) < 0:
            raise ValueError('Trial measurements must be nonnegative and finite')
        if any(type(row[key]) is not bool for key in ('drop', 'protective_stop')):
            raise ValueError('drop and protective_stop must be booleans')
        success = lift >= .10 and hold >= 2 and not row['drop'] and not row['protective_stop']
        bucket[scenario] = dict(success=success, cycle_s=cycle, drop=row['drop'], stop=row['protective_stop'])
    expected = scenario_ids
    report = {}
    for candidate, trials in groups.items():
        if len(trials) != 100 or (expected is not None and set(trials) != expected):
            raise ValueError('Every candidate requires the same 100 distinct scenarios')
        expected = set(trials)
        successes = sum(row['success'] for row in trials.values())
        report[candidate] = dict(successes=successes, success_rate=successes/100, ci95=wilson(successes,100),
                                 passed_success_target=successes >= target,
                                 cycle_s=distribution([row['cycle_s'] for row in trials.values()]),
                                 drops=sum(row['drop'] for row in trials.values()),
                                 protective_stops=sum(row['stop'] for row in trials.values()))
    return dict(trials_per_candidate=100, target_successes=target, candidates=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['calibrate', 'replay', 'scenarios', 'accept'])
    parser.add_argument('input', type=Path, nargs='?')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--limits', type=Path)
    parser.add_argument('--scenarios', type=Path)
    parser.add_argument('--center-xy', type=float, nargs=2)
    parser.add_argument('--radius', type=float, default=.05)
    parser.add_argument('--lighting', nargs='+')
    parser.add_argument('--seed', type=int, default=91721)
    args = parser.parse_args()
    if args.mode == 'replay' and (args.checkpoint is None or args.limits is None):
        parser.error('replay requires --checkpoint and --limits')
    if args.mode == 'scenarios':
        if args.center_xy is None or args.lighting is None:
            parser.error('scenarios requires --center-xy and --lighting')
        args.output.write_text(json.dumps(scenarios(args.center_xy, args.radius, args.lighting, args.seed), indent=2))
        return
    if args.input is None:
        parser.error('input is required')
    if args.input.resolve() == args.output.resolve():
        parser.error('output must differ from input')
    rows = read_rows(args.input)
    if args.mode == 'calibrate':
        report = calibration(rows)
    elif args.mode == 'accept':
        if args.scenarios is None:
            parser.error('accept requires --scenarios')
        manifest = json.loads(args.scenarios.read_text())['scenarios']
        scenario_ids = {row['scenario_id'] for row in manifest}
        if len(manifest) != 100 or len(scenario_ids) != 100:
            raise ValueError('The manifest must contain 100 unique scenarios')
        report = acceptance(rows, scenario_ids=scenario_ids)
    else:
        report = replay(rows, args.checkpoint, load_limits(args.limits))
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
