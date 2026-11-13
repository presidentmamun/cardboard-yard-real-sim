"""Watch the expert Komatsu WA100-8 loader work in CardboardYardReal (physics v4, the setting of the v4 dataset).

    python demo.py --viewer                       # live 3D window
    python demo.py --video yard.mp4 --steps 700   # 2x2 camera video, no window
    python demo.py --csv ep.csv --steps 3000      # per-step time, 4 actions, 12 proprio (+ ep_start.npz for replay.py)
    python demo.py --expert v2 --viewer           # the braking expert on physics v5 (brake pedal in the action)

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
from cardboard_yard_real.expert_v2 import LoaderExpertPolicyV2
from recording import Recorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=700)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--physics', default=None, help='default: v4 with expert v1, v5 with expert v2')
    ap.add_argument('--expert', default='v1', choices=['v1', 'v2'],
                    help="v1 = the v4 dataset's expert (no brake); v2 = v1 + brake pedal (needs physics v5)")
    ap.add_argument('--video', default=None, help='write an mp4 here')
    ap.add_argument('--csv', default=None, help='write time, action and proprio per step here')
    ap.add_argument('--viewer', action='store_true', help='open the MuJoCo viewer')
    ap.add_argument('--p-weak', type=float, default=0.0, help='chance of a noisy episode')
    ap.add_argument('--noise-scale', type=float, default=0.0)
    args = ap.parse_args()

    physics = args.physics or ('v5' if args.expert == 'v2' else 'v4')
    env = CardboardYardRealEnv(physics=physics)
    if args.expert == 'v2' and not env.has_brake:
        print(f'note: physics {physics} has no brake: expert v2 drives like v1 here')
    cls = LoaderExpertPolicyV2 if args.expert == 'v2' else LoaderExpertPolicyV1
    policy = cls(p_weak=args.p_weak, noise_scale=args.noise_scale, seed=args.seed)
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
            print(f'step {t:4d}  in bucket={info["pieces_in_bucket"]:2d}  in hopper={info["pieces_in_hopper"]:2d}'
                  + (f'  brake={a[4]:.2f}' if len(a) > 4 else ''))
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
