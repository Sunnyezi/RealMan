"""Fit the known cube's top face and check a yaw-adjusted candidate in simulation."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from build_scene import R, solve
from env import GraspEnv
from protocol import VALIDATION_SEEDS, stress_yaw

ANGLES = np.deg2rad(np.arange(-45., 45., .5))
ROTATIONS = Rotation.from_euler('z', ANGLES[:, None]).as_matrix()


def fit_cube_top(env):
    view = env.vision
    rgb = view.rgb.astype(float)
    mask = ((rgb[:, :, 2] > 35) & (rgb[:, :, 2] > 3*rgb[:, :, 0])
            & (rgb[:, :, 2] > 3*rgb[:, :, 1]) & np.isfinite(view.depth)
            & (view.depth > .01) & (view.depth < 2.))
    v, u = np.where(mask)
    if len(u) < 12:
        raise ValueError('Insufficient depth pixels for cube pose')
    depth = view.depth[v, u]
    focal = 64 / np.tan(np.deg2rad(env.m.cam_fovy[view.cid])/2)
    points = np.column_stack([(u-63.5)*depth/focal, -(v-63.5)*depth/focal, -depth])
    points = points @ env.d.cam_xmat[view.cid].reshape(3, 3).T + env.d.cam_xpos[view.cid]
    face = points[points[:, 2] >= np.quantile(points[:, 2], .90)-.0015]
    if len(face) < 12:
        raise ValueError('Insufficient visible pixels on the top face')
    rotated = face[:, :2] @ ROTATIONS[:, :2, :2]
    low, high = rotated.min(axis=1), rotated.max(axis=1)
    index = np.argmin(np.prod(high-low, axis=1))
    if np.any((high-low)[index] < .02):
        raise ValueError('Partial top face cannot determine cube pose')
    center_xy = ((low[index]+high[index])/2) @ ROTATIONS[index, :2, :2].T
    half_height = env.m.geom_size[env.og, 2]
    return np.r_[center_xy, np.median(face[:, 2])-half_height], ROTATIONS[index], ANGLES[index]


def check_candidate(env, center, yaw_rotation):
    rotation = yaw_rotation @ R
    grasp_offset = np.asarray(env.cfg['grasp_xyz'])-env.cfg['object_xyz']
    goal = center + yaw_rotation @ grasp_offset
    previous = np.asarray(env.cfg['home_xyz'])
    q = np.asarray(env.cfg['home_q'])
    interpolate = Slerp([0, 1], Rotation.from_matrix(np.stack((R, rotation))))
    endpoints = (goal+[0, 0, .08], goal, goal+[0, 0, env.cfg['lift_distance']])
    for segment, endpoint in enumerate(endpoints):
        for fraction in np.linspace(.2, 1., 5):
            xyz = previous + (endpoint-previous)*fraction
            target = interpolate(fraction).as_matrix() if not segment else rotation
            q = solve(env.m, env.ikd, xyz, target, q)
            for contact in env.ikd.contact:
                kinds = {int(env.m.geom_contype[contact.geom1]), int(env.m.geom_contype[contact.geom2])}
                if kinds == {4, 8} and contact.dist < -.0005:
                    raise ValueError('Arm floor penetration')
        previous = endpoint


def probe():
    env = GraspEnv(randomize=True)
    rows = []
    try:
        for seed in VALIDATION_SEEDS:
            yaw = stress_yaw(seed)
            row = dict(seed=seed, yaw_deg=float(np.rad2deg(yaw)), candidate_reachable=False)
            try:
                env.reset(seed=seed, options={'object_yaw_rad': yaw})
                center, rotation, estimate = fit_cube_top(env)
                error = abs((estimate-yaw+np.pi/4) % (np.pi/2)-np.pi/4)
                row.update(yaw_error_deg=float(np.rad2deg(error)),
                           center_xy_error_mm=float(np.linalg.norm(center[:2]-env.d.xpos[env.ob, :2])*1000))
                check_candidate(env, center, rotation)
                row['candidate_reachable'] = True
            except (ValueError, RuntimeError) as error:
                row['failure'] = str(error)
            rows.append(row)
    finally:
        env.close()
    errors = [row['yaw_error_deg'] for row in rows if 'yaw_error_deg' in row]
    return dict(scenes=len(rows), estimates=len(errors), reachable=sum(row['candidate_reachable'] for row in rows),
                median_yaw_error_deg=float(np.median(errors)) if errors else None,
                p95_yaw_error_deg=float(np.quantile(errors, .95)) if errors else None, results=rows)
