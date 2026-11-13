"""Checks of physics v5 (brakes). Run:  .venv\\Scripts\\python.exe test_v5.py

  1. v5 with a 4-number action steps exactly like v4 (bit-equal, expert drive)
  2. stopping from full speed: releasing the lever / full brake pedal / half pedal / parking brake
  3. parking brake holds the machine against full drive
  4. a drive with brake columns is saved, loaded and replayed exactly

    test_v5.py --dump out.npz --physics v4 --steps 300   # only write an expert trajectory (to compare two
                                                         # copies of the simulator, e.g. the original v4 zip)
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def expert_run(physics, steps, seed=0):
    from cardboard_yard_real import CardboardYardRealEnv
    from cardboard_yard_real.expert_v1 import LoaderExpertPolicyV1

    env = CardboardYardRealEnv(physics=physics)
    policy = LoaderExpertPolicyV1(seed=seed)
    env.reset(seed=seed)
    pro, act = [], []
    for _ in range(steps):
        a = np.asarray(policy.act_single(env), np.float32)
        act.append(a)
        env.step(a)
        pro.append(env._proprio().astype(np.float64))
    out = dict(proprio=np.array(pro), action=np.array(act), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy())
    env.close()
    return out


def stop_test(env, seed, action_after, label, speed_up=45, watch=60):
    """Full speed ahead for speed_up steps, then action_after: -> (speed before, steps to stop, distance).
    label None: quiet run that returns whether the way was free (nothing touched, nobody near)."""
    env.reset(seed=seed)
    touched = False
    for _ in range(speed_up):
        info = env.step([1, 0, 0, 0])[4]
        touched |= bool(info['hit_static'] or info['hit_obstacle'] or info['hit_worker'] or info['pieces_in_bucket'])
    v0 = env.forward_speed()
    x0, y0, _ = env.loader_pose()
    n = None
    for i in range(watch):
        info = env.step(action_after)[4]
        touched |= bool(info['hit_static'] or info['hit_obstacle'] or info['hit_worker'] or info['pieces_in_bucket'])
        if n is None and abs(env.forward_speed()) < 0.02:
            n = i + 1
            x1, y1, _ = env.loader_pose()
    dist = float(np.hypot(x1 - x0, y1 - y0)) if n else float('nan')
    if label is None:
        return (not touched) and v0 > 2.95 and n is not None
    print(f'  {label:<24} from {v0:.2f} m/s: stops in {(n or 0) * env.dt:.1f} s, {dist:.2f} m')
    return v0, n, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default=None)
    ap.add_argument('--physics', default='v4')
    ap.add_argument('--steps', type=int, default=300)
    args = ap.parse_args()
    if args.dump:
        np.savez(args.dump, **expert_run(args.physics, args.steps))
        print('saved', args.dump)
        return

    from cardboard_yard_real import CardboardYardRealEnv
    from recording import Recorder, load, restart

    ok = True

    print('1. v5 without brake input == v4')
    a, b = expert_run('v4', args.steps), expert_run('v5', args.steps)
    same = all(np.array_equal(a[k], b[k]) for k in a)
    print('  bit-equal:', same)
    ok &= same

    print('2. stopping from full speed (v5)')
    env = CardboardYardRealEnv(physics='v5')
    # a seed whose start has 10 m of free floor ahead (the slowest stop needs 6.75 + 4.5 m)
    seed = next(s for s in range(200) if stop_test(env, s, [0, 0, 0, 0], None))
    print('  seed', seed)
    _, n_rel, d_rel = stop_test(env, seed, [0, 0, 0, 0], 'lever released')
    _, n_half, d_half = stop_test(env, seed, [0, 0, 0, 0, 0.5, 0], 'brake pedal half')
    _, n_full, d_full = stop_test(env, seed, [0, 0, 0, 0, 1, 0], 'brake pedal full')
    _, n_gas, d_gas = stop_test(env, seed, [1, 0, 0, 0, 1, 0], 'full brake + full drive')
    _, n_park, d_park = stop_test(env, seed, [0, 0, 0, 0, 0, 1], 'parking brake')
    good = bool(n_rel and n_half and n_full and n_park and n_gas and d_full < d_half < d_rel and abs(d_gas - d_full) < 0.05)
    print('  full < half < released, brake cuts the drive:', good)
    ok &= good

    print('3. parking brake holds against full drive')
    env.reset(seed=seed)
    x0, y0, _ = env.loader_pose()
    for _ in range(40):
        env.step([1, 0, 0, 0, 0, 1])
    x1, y1, _ = env.loader_pose()
    moved = float(np.hypot(x1 - x0, y1 - y0))
    for _ in range(20):
        env.step([1, 0, 0, 0, 0, 0])
    released = abs(env.forward_speed())
    good = moved < 0.01 and released > 1.0
    print(f'  moved {moved:.4f} m with the brake on; {released:.2f} m/s 2 s after release:', good)
    ok &= good

    print('4. save / load / replay a drive with brake columns')
    rng = np.random.default_rng(1)
    env.reset(seed=3)
    rec = Recorder()
    rec.begin(env, 3)
    for t in range(150):
        act = np.array([rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), 0, 0,
                        rng.uniform(0, 1) if 40 <= t % 60 < 55 else 0, 1.0 if 120 <= t < 135 else 0.0], np.float32)
        rec.add(env, act)
        env.step(act)
    end = env._proprio().copy()
    with tempfile.TemporaryDirectory() as tmp:
        path = rec.save(os.path.join(tmp, 'd.csv'))
        act2, pro2, start = load(path)
    env2 = CardboardYardRealEnv(physics=start['physics'])
    how = restart(env2, start)
    for row in act2:
        env2.step(row)
    off = float(np.abs(env2._proprio() - end).max())
    good = act2.shape == (150, 6) and start['physics'] == 'v5' and how == 'seed' and off == 0.0
    print(f'  actions {act2.shape}, physics {start["physics"]}, start from {how}, replay off by {off}:', good)
    ok &= good

    print('ALL OK' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
