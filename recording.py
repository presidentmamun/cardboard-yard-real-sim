"""Save / load a drive: one CSV row per step (time, 4 actions [+ brake pedal, parking brake with physics v5],
12 proprio) plus <name>_start.npz with the episode's start (full sim state + random generator state + physics
profile), so replay.py can run the same drive again."""

import csv
import json
import math
import os

import numpy as np

ACTION = ['drive', 'steering', 'boom_lever', 'bucket_lever']
ACTION_BRAKE = ['brake_pedal', 'park_brake']  # physics v5 only: pedal 0..1, parking brake 0 / 1
PROPRIO = ['pos_x_m', 'pos_y_m', 'yaw_cos', 'yaw_sin', 'vel_x_mps', 'vel_y_mps', 'steer_angle_rad', 'boom_angle_rad',
           'bucket_angle_rad', 'yaw_rate_radps', 'lift_cyl_pressure_mpa', 'tilt_cyl_pressure_mpa']


def start_of(env, seed=None):
    """Call right after env.reset(): everything needed to start this episode again."""
    d = env.data  # same layout as env._full_state(), but float64 (float32 would make the replay drift)
    state = np.concatenate([d.qpos, d.qvel, [env._boom_target, env._bucket_target, env._wheel_angle, env._v_cmd,
                                             env._steer_cmd], env.workers.reshape(-1), env.obstacles.reshape(-1),
                            env.pile_center, env.pile_half]).astype(np.float64)
    return {'state': state,
            'rng': json.dumps(env.np_random.bit_generator.state),
            'seed': -1 if seed is None else int(seed),
            'physics': str(getattr(env, 'physics', ''))}


def start_path(csv_path):
    return os.path.splitext(csv_path)[0] + '_start.npz'


class Recorder:
    def __init__(self):
        self.rows, self.start, self.n_act = [], None, len(ACTION)

    def begin(self, env, seed=None):
        self.rows, self.start = [], start_of(env, seed)

    def add(self, env, action):
        """Call before env.step(action): the proprio the action was chosen on, like the dataset. Actions keep full
        float32 precision (9 digits): rounded actions make the replay drift (contacts amplify tiny differences)."""
        p = env._proprio()
        t = len(self.rows)
        action = np.asarray(action, np.float32).reshape(-1)
        self.n_act = len(action)  # 4, or 6 with the brake columns
        self.rows.append([f'{t * env.dt:.2f}', t] + [f'{v:.9g}' for v in action] + [f'{v:.5f}' for v in p]
                         + [f'{math.degrees(math.atan2(p[3], p[2])):.2f}'])

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            n_pro = len(self.rows[0]) - 3 - self.n_act
            w.writerow(['time_s', 'step'] + (ACTION + ACTION_BRAKE)[:self.n_act] + PROPRIO[:n_pro] + ['yaw_deg'])
            w.writerows(self.rows)
        np.savez(start_path(path), **self.start)
        return path


def load(path):
    """-> actions (T, 4) or (T, 6) with the brake columns, proprio (T, P), start dict or None (no _start.npz
    next to the CSV)."""
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames
    acols = ACTION + [c for c in ACTION_BRAKE if c in cols]
    act = np.array([[float(x[c]) for c in acols] for x in rows], np.float32)
    n = len(act)
    while n and not np.isfinite(act[n - 1]).all():  # dataset episodes end with an observation-only row (nan action)
        n -= 1
    rows, act = rows[:n], act[:n]
    pcols = [c for c in PROPRIO if c in cols]
    pro = np.array([[float(x[c]) for c in pcols] for x in rows], np.float32)
    sp = start_path(path)
    start = None
    if os.path.exists(sp):
        z = np.load(sp)
        start = {'state': z['state'], 'rng': str(z['rng']) if 'rng' in z else '', 'seed': int(z['seed']) if 'seed' in z else -1,
                 'physics': str(z['physics']) if 'physics' in z else ''}
    return act, pro, start


def restart(env, start, seed=0):
    """Reset env to a saved start; returns 'seed' (exact replay) or 'state' (close, drifts a little).
    A drive saved with its seed is reset from that seed: bit-exact. The saved state alone does not hold
    MuJoCo's solver warm start etc., so a replay from it drifts by some 10 cm (used when the seed does not match)."""
    if start is None:
        env.reset(seed=seed)
        return 'seed'
    if start['seed'] >= 0:
        env.reset(seed=start['seed'])
        if np.allclose(start_of(env)['state'], start['state'], atol=1e-9, rtol=0):
            return 'seed'
    env.reset(options={'state': start['state']})
    if start['rng']:  # the people pick their next waypoints from this generator during the episode
        env.np_random.bit_generator.state = json.loads(start['rng'])
    return 'state'
