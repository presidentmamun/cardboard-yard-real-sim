"""Watch the expert Komatsu WA100-8 loader work in CardboardYardReal (physics v4, the setting of the v4 dataset).

    python demo.py --viewer                       # live 3D window
    python demo.py --video yard.mp4 --steps 700   # 2x2 camera video, no window
    python demo.py --csv ep.csv --steps 3000      # per-step time, 4 actions, 12 proprio (+ ep_start.npz for replay.py)

The video is a 2x2 grid: chase (follows the loader) | top (the view the world model sees)
on top, cab (front, onboard) | rear (back, onboard) below.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# the cardboard_yard and cardboard_yard_real packages sit next to this script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardboard_yard_real import CardboardYardRealEnv
from cardboard_yard_real.expert_v1 import LoaderExpertPolicyV1
from recording import Recorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=700)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--physics', default='v4')
    ap.add_argument('--video', default=None, help='write an mp4 here')
    ap.add_argument('--csv', default=None, help='write time, action and proprio per step here')
    ap.add_argument('--viewer', action='store_true', help='open the MuJoCo viewer')
    ap.add_argument('--p-weak', type=float, default=0.0, help='chance of a noisy episode')
    ap.add_argument('--noise-scale', type=float, default=0.0)
    args = ap.parse_args()

    env = CardboardYardRealEnv(physics=args.physics)
    policy = LoaderExpertPolicyV1(p_weak=args.p_weak, noise_scale=args.noise_scale, seed=args.seed)
    env.reset(seed=args.seed)
    rec = Recorder()
    rec.begin(env, args.seed)

    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    frames = []
    for t in range(args.steps):
        a = np.asarray(policy.act_single(env), np.float32)
        rec.add(env, a)
        _, _, terminated, _, info = env.step(a)
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(env.dt)
        if args.video:
            v = {c: env.render(camera=c, width=640, height=360) for c in ('chase', 'top', 'cab', 'rear')}
            frames.append(np.concatenate([np.concatenate([v['chase'], v['top']], axis=1),
                                          np.concatenate([v['cab'], v['rear']], axis=1)], axis=0))
        if t % 50 == 0 or terminated:
            print(f'step {t:4d}  in bucket={info["pieces_in_bucket"]:2d}  in hopper={info["pieces_in_hopper"]:2d}')
        if terminated:
            print('episode finished:', 'hit a person' if info['hit_worker'] else 'hopper target reached')
            break

    if args.csv:
        print('csv saved:', rec.save(args.csv), len(rec.rows), 'rows (+ _start.npz; watch again: python replay.py', args.csv + ')')
    if args.video:
        import imageio

        imageio.mimsave(args.video, frames, fps=int(round(1 / env.dt)), macro_block_size=2)
        print('video saved:', args.video)
    env.close()


if __name__ == '__main__':
    main()
