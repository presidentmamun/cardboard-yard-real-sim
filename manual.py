"""Drive the wheel loader in CardboardYardReal (physics v4, Komatsu WA100-8) yourself with the keyboard.

    python manual.py              # chase camera
    python manual.py --camera top --seed 3
    python manual.py --no-save    # do not save the drives
    python manual.py --joy-map 1,0,3,2 --joy-invert 1,0,1,0   # joystick axes for drive, steer, boom, bucket

Joystick / gamepad (plugged in before start): analog sticks give values between -1 and 1. Default (Xbox-style pad):
              left stick up/down = drive, left stick left/right = steer,
              right stick up/down = boom, right stick left/right = bucket.
              Button 0 (A) = save now, button 7 (Start) = new episode. Keyboard keys work at the same time.

Keys (hold):  W/S forward/back   A/D steer left/right
              I/K boom up/down   J/L bucket curl (lip up) / dump (lip down)
Other:        C next camera   P save the drive now   R reset (new episode)   Esc quit

Each drive is saved on R / Esc / episode end to drives/drive_<date>_<time>_seed<N>.csv (+ _start.npz);
watch it again with: python replay.py drives/<that file>.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pygame

# the cardboard_yard and cardboard_yard_real packages sit next to this script
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cardboard_yard_real import CardboardYardRealEnv
from recording import Recorder

CAMERAS = ['chase', 'top', 'cab', 'rear']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--physics', default='v4')
    ap.add_argument('--camera', default='chase', choices=CAMERAS)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--out-dir', default=str(HERE / 'drives'), help='where the drives are saved')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--joy-map', default='1,0,3,2', help='joystick axis numbers for drive, steer, boom, bucket')
    ap.add_argument('--joy-invert', default='1,1,1,1', help='1 = flip that axis (sticks give -1 when pushed up)')
    ap.add_argument('--joy-dead', type=float, default=0.1, help='dead zone: smaller stick values count as 0')
    args = ap.parse_args()

    env = CardboardYardRealEnv(physics=args.physics)
    seed = args.seed
    rec = Recorder()
    env.reset(seed=seed)
    rec.begin(env, seed)
    saved = False
    cam = CAMERAS.index(args.camera)

    def save():
        nonlocal saved
        if args.no_save or saved or not rec.rows:
            return
        name = time.strftime('drive_%Y%m%d_%H%M%S') + f'_seed{seed}.csv'
        print('drive saved:', rec.save(os.path.join(args.out_dir, name)), f'({len(rec.rows)} steps)')
        saved = True

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    joy = None
    if pygame.joystick.get_count():
        joy = pygame.joystick.Joystick(0)
        joy.init()
        print(f'joystick: {joy.get_name()} ({joy.get_numaxes()} axes, {joy.get_numbuttons()} buttons)')
    joy_axes = [int(v) for v in args.joy_map.split(',')]
    joy_sign = [-1.0 if v.strip() == '1' else 1.0 for v in args.joy_invert.split(',')]

    def joystick():
        if joy is None:
            return np.zeros(4, np.float32)
        out = []
        for ax, sg in zip(joy_axes, joy_sign):
            v = joy.get_axis(ax) if ax < joy.get_numaxes() else 0.0
            out.append(0.0 if abs(v) < args.joy_dead else sg * v)
        return np.clip(np.array(out, np.float32), -1, 1)

    font = pygame.font.SysFont('consolas', 18)
    info = {'pieces_in_bucket': 0, 'pieces_in_hopper': 0}
    done = False
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                save()
                env.close()
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_c:
                cam = (cam + 1) % len(CAMERAS)
            if (ev.type == pygame.KEYDOWN and ev.key == pygame.K_p) or (ev.type == pygame.JOYBUTTONDOWN and ev.button == 0):
                save()
                saved = False  # keep driving; a later save writes a new, longer file
            if (ev.type == pygame.KEYDOWN and ev.key == pygame.K_r) or (ev.type == pygame.JOYBUTTONDOWN and ev.button == 7):
                save()
                seed += 1
                env.reset(seed=seed)
                rec.begin(env, seed)
                saved, done = False, False
        k = pygame.key.get_pressed()
        # action: [speed, steer, boom (+ raises), bucket (+ lip up)], each in [-1, 1]
        a = np.array([
            k[pygame.K_w] - k[pygame.K_s],
            k[pygame.K_a] - k[pygame.K_d],
            k[pygame.K_i] - k[pygame.K_k],
            k[pygame.K_j] - k[pygame.K_l],
        ], np.float32)
        j = joystick()
        a = np.where(a != 0, a, j)  # a held key wins over the stick on that lever
        t0 = time.time()
        if not done:
            rec.add(env, a)
            _, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                done = True
                print(f'episode end at step {len(rec.rows)}: hopper {info["pieces_in_hopper"]} -> press R for a new one')
                save()
        img = env.render(camera=CAMERAS[cam], width=args.width, height=args.height)
        screen.blit(pygame.surfarray.make_surface(img.swapaxes(0, 1)), (0, 0))
        hud = (f'{CAMERAS[cam]} | seed {seed} step {len(rec.rows)} | action {np.round(a, 2)} | '
               f'bucket {info["pieces_in_bucket"]}  hopper {info["pieces_in_hopper"]}')
        for flag in ('hit_worker', 'hit_obstacle', 'hit_static'):
            if info.get(flag):
                hud += f' | {flag.upper()}'
        if done:
            hud += ' | END (R = new episode)'
        screen.blit(font.render(hud, True, (255, 255, 0), (0, 0, 0)), (8, 8))
        pygame.display.flip()
        time.sleep(max(0.0, env.dt - (time.time() - t0)))


if __name__ == '__main__':
    main()
