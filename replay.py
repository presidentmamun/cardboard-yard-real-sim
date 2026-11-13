"""Run a saved drive again from its CSV: a manual drive (drives/*.csv), a demo.py --csv run, or a dataset episode.

    python replay.py drives/drive_20261030_101500_seed0.csv          # window; C camera, Space pause, Esc quit
    python replay.py ep.csv --video ep.mp4                           # 2x2 camera video, no window
    python replay.py ep.csv --check                                  # no window: only how close the replay is

The start comes from <csv name>_start.npz next to the CSV (written by manual.py and demo.py). Without it the
episode starts from --seed, which is the same yard only if the drive was made with that seed.
At the end it prints how far the replayed loader is from the CSV (position, boom / bucket angle).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardboard_yard_real import CardboardYardRealEnv
from recording import load, restart

CAMERAS = ['chase', 'top', 'cab', 'rear']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--seed', type=int, default=0, help='only used when there is no _start.npz')
    ap.add_argument('--physics', default=None,
                    help='physics profile; default: the one saved with the drive (_start.npz), else v4')
    ap.add_argument('--camera', default='chase', choices=CAMERAS)
    ap.add_argument('--video', default=None, help='write a 2x2 camera mp4 here instead of a window')
    ap.add_argument('--check', action='store_true', help='no window, no video: only compare with the CSV')
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    args = ap.parse_args()

    act, pro, start = load(args.csv)
    print(f'{len(act)} steps; start: ' + ('saved start state' if start is not None else f'seed {args.seed} (no _start.npz)')
          + ('' if start is None or start['rng'] else ', no random-generator state: walking people may differ'))
    physics = args.physics or (start or {}).get('physics') or 'v4'
    if act.shape[1] > 4 and physics in ('v0', 'v1', 'v2', 'v3', 'v4'):
        print(f'warning: the CSV has brake columns but physics {physics} has no brake: they are ignored')
    print('physics', physics)
    env = CardboardYardRealEnv(physics=physics)
    how = restart(env, start, args.seed)
    print('started from ' + ('the seed (exact replay)' if how == 'seed' else 'the saved state (close, small drift)'))

    screen = None
    if not args.video and not args.check:
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((args.width, args.height))
        font = pygame.font.SysFont('consolas', 18)
    cam, paused, frames = CAMERAS.index(args.camera), False, []
    err_pos, err_arm = [], []
    t = 0
    while t < len(act):
        if screen is not None:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    t = len(act)
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_c:
                    cam = (cam + 1) % len(CAMERAS)
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    paused = not paused
            if paused or t >= len(act):
                time.sleep(0.05)
                continue
        t0 = time.time()
        p = env._proprio()
        n = min(len(p), pro.shape[1])
        d = p[:n] - pro[t, :n]
        err_pos.append(float(np.hypot(d[0], d[1])))
        err_arm.append(float(max(abs(d[7]), abs(d[8]))))
        _, _, terminated, _, info = env.step(act[t])
        t += 1
        if screen is not None:
            img = env.render(camera=CAMERAS[cam], width=args.width, height=args.height)
            screen.blit(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))
            hud = (f'REPLAY {CAMERAS[cam]} | step {t}/{len(act)} | action {np.round(act[t - 1, :4], 2)} | '
                   f'off by {err_pos[-1]:.2f} m | hopper {info["pieces_in_hopper"]}')
            if act.shape[1] > 4 and act[t - 1, 4] > 0:
                hud += f' | BRAKE {act[t - 1, 4]:.1f}'
            if act.shape[1] > 5 and act[t - 1, 5] > 0.5:
                hud += ' | PARK'
            screen.blit(font.render(hud, True, (0, 255, 255), (0, 0, 0)), (8, 8))
            pygame.display.flip()
            time.sleep(max(0.0, env.dt - (time.time() - t0)))
        elif args.video:
            v = {c: env.render(camera=c, width=640, height=360) for c in CAMERAS}
            frames.append(np.concatenate([np.concatenate([v['chase'], v['top']], axis=1),
                                          np.concatenate([v['cab'], v['rear']], axis=1)], axis=0))
        if t % 500 == 0:
            print(f'step {t:5d}  position off by {err_pos[-1]:.3f} m  (max so far {max(err_pos):.3f} m)')
        if terminated and t < len(act):
            print(f'replay episode ended at step {t} (the CSV has {len(act)} steps)')
            break

    if not err_pos:
        return
    print(f'replayed {t} steps: position off by max {max(err_pos):.3f} m, mean {np.mean(err_pos):.3f} m, '
          f'end {err_pos[-1]:.3f} m; boom/bucket angle off by max {max(err_arm):.3f} rad')
    if args.video:
        import imageio

        imageio.mimsave(args.video, frames, fps=int(round(1 / env.dt)), macro_block_size=2)
        print('video saved:', args.video)
    env.close()


if __name__ == '__main__':
    main()
