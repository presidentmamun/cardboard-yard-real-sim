"""Drive the wheel loader in CardboardYardReal (Komatsu WA100-8) yourself with the keyboard or a gamepad.
Physics v5 (default) = the v4 physics + brake pedal and parking brake.

    python manual.py                       # chase camera, lever drive
    python manual.py --drive-mode pedal    # accelerator pedal + F / N / R selector, like the real machine
    python manual.py --camera top --seed 3
    python manual.py --physics v4          # exactly the v4 dataset's simulator (no brake)
    python manual.py --no-save             # do not save the drives
    python manual.py --joy-map 1,0,3,2 --joy-invert 1,0,1,0   # joystick axes for drive, steer, boom, bucket

Keys (hold):  A/D steer left/right   I/K boom up/down   J/L bucket curl (lip up) / dump (lip down)
  lever mode: W/S drive forward/back        Space brake pedal
  pedal mode: W accelerator                 S or Space brake pedal     E / Q direction up / down (R - N - F)
Other:        B parking brake on/off   M lever <-> pedal mode   H status panel on/off
              C next camera   P save the drive now   R reset (new episode)   Esc quit

Joystick / gamepad (plugged in before start): analog sticks give values between -1 and 1. Default (Xbox-style pad):
              left stick up/down = drive (lever mode), left stick left/right = steer,
              right stick up/down = boom, right stick left/right = bucket,
              left trigger = brake pedal, right trigger = accelerator (pedal mode),
              LB / RB = direction down / up (pedal mode), button 1 (B) = parking brake,
              button 0 (A) = save now, button 7 (Start) = new episode. Keyboard keys work at the same time.

Each drive is saved on R / Esc / episode end to drives/drive_<date>_<time>_seed<N>.csv (+ _start.npz);
watch it again with: python replay.py drives/<that file>.csv
"""

import argparse
import math
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
from controls import GEAR_NAMES, PedalDrive, Trigger
from recording import Recorder

CAMERAS = ['chase', 'top', 'cab', 'rear']
YELLOW, WHITE, GREY, RED, GREEN, BLACK = (255, 255, 0), (235, 235, 235), (90, 90, 90), (230, 50, 40), (60, 200, 90), (0, 0, 0)


def draw_bar(screen, font, x, y, label, value, color, w=150, h=12):
    """A labelled bar for a value in [-1, 1] (drive: centre = 0) or [0, 1] (pedals)."""
    screen.blit(font.render(label, True, WHITE), (x, y - 3))
    bx = x + 78
    pygame.draw.rect(screen, GREY, (bx, y, w, h), 1)
    if label == 'drive':
        mid = bx + w // 2
        pygame.draw.line(screen, GREY, (mid, y), (mid, y + h))
        n = int(abs(value) * (w // 2 - 1))
        if n:
            pygame.draw.rect(screen, color, (mid if value > 0 else mid - n, y + 1, n, h - 2))
    else:
        n = int(max(0.0, min(1.0, value)) * (w - 2))
        if n:
            pygame.draw.rect(screen, color, (bx + 1, y + 1, n, h - 2))


def draw_panel(screen, font, env, a, mode, pedal, brake, park):
    """Status panel, bottom left: speed, direction, pedals, parking brake, arm angles, cylinder pressures."""
    p = env._proprio()
    v = env.forward_speed()
    rows = 7 if env.has_brake else 6
    w, h = 250, 22 * rows + 14
    x0, y0 = 8, screen.get_height() - h - 8
    box = pygame.Surface((w, h), pygame.SRCALPHA)
    box.fill((0, 0, 0, 150))
    screen.blit(box, (x0, y0))
    x, y = x0 + 8, y0 + 8
    way = 'forward' if v > 0.05 else 'reverse' if v < -0.05 else 'stopped'
    screen.blit(font.render(f'{abs(v) * 3.6:5.1f} km/h  {way}', True, YELLOW), (x, y))
    y += 22
    if mode == 'pedal':
        screen.blit(font.render(f'direction  [{GEAR_NAMES[pedal.gear]}]   (Q / E)', True, WHITE), (x, y))
        y += 22
        draw_bar(screen, font, x, y + 3, 'accel', pedal.pedal, GREEN)
    else:
        screen.blit(font.render('lever drive   (M: pedal)', True, WHITE), (x, y))
        y += 22
        draw_bar(screen, font, x, y + 3, 'drive', float(a[0]), GREEN)
    y += 22
    if env.has_brake:
        draw_bar(screen, font, x, y + 3, 'brake', brake, RED)
        y += 22
        screen.blit(font.render('PARKING BRAKE ON (B)' if park else 'parking brake off', True, RED if park else GREY), (x, y))
        y += 22
    else:
        screen.blit(font.render(f'no brake ({env.physics})', True, GREY), (x, y))
        y += 22
    screen.blit(font.render(f'boom {math.degrees(p[7]):6.1f}  bucket {math.degrees(p[8]):6.1f} deg', True, WHITE), (x, y))
    y += 22
    if len(p) >= 12:
        screen.blit(font.render(f'lift {p[10]:5.1f}  tilt {p[11]:5.1f} MPa', True, WHITE), (x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--physics', default='v5', help="v5 = v4 + brakes (default); v4 = the v4 dataset's simulator")
    ap.add_argument('--drive-mode', default='lever', choices=['lever', 'pedal'],
                    help='lever: one forward / back lever (v4 way); pedal: accelerator + F / N / R selector')
    ap.add_argument('--camera', default='chase', choices=CAMERAS)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--out-dir', default=str(HERE / 'drives'), help='where the drives are saved')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--no-panel', action='store_true', help='start without the status panel (H shows it)')
    ap.add_argument('--joy-map', default='1,0,3,2', help='joystick axis numbers for drive, steer, boom, bucket')
    ap.add_argument('--joy-invert', default='1,1,1,1', help='1 = flip that axis (sticks give -1 when pushed up)')
    ap.add_argument('--joy-dead', type=float, default=0.1, help='dead zone: smaller stick values count as 0')
    ap.add_argument('--joy-brake', type=int, default=4, help='trigger axis of the brake pedal (-1 = none)')
    ap.add_argument('--joy-accel', type=int, default=5, help='trigger axis of the accelerator, pedal mode (-1 = none)')
    args = ap.parse_args()

    env = CardboardYardRealEnv(physics=args.physics)
    seed = args.seed
    rec = Recorder()
    env.reset(seed=seed)
    rec.begin(env, seed)
    saved = False
    cam = CAMERAS.index(args.camera)
    mode, pedal, park, panel = args.drive_mode, PedalDrive(), False, not args.no_panel

    def save():
        nonlocal saved
        if args.no_save or saved or not rec.rows:
            return
        name = time.strftime('drive_%Y%m%d_%H%M%S') + f'_seed{seed}.csv'
        print('drive saved:', rec.save(os.path.join(args.out_dir, name)), f'({len(rec.rows)} steps)')
        saved = True

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption(f'CardboardYardReal {args.physics} - manual drive')
    joy = None
    if pygame.joystick.get_count():
        joy = pygame.joystick.Joystick(0)
        joy.init()
        print(f'joystick: {joy.get_name()} ({joy.get_numaxes()} axes, {joy.get_numbuttons()} buttons)')
    joy_axes = [int(v) for v in args.joy_map.split(',')]
    joy_sign = [-1.0 if v.strip() == '1' else 1.0 for v in args.joy_invert.split(',')]
    trig_brake, trig_accel = Trigger(joy, args.joy_brake), Trigger(joy, args.joy_accel)

    def joystick():
        if joy is None:
            return np.zeros(4, np.float32)
        out = []
        for ax, sg in zip(joy_axes, joy_sign):
            v = joy.get_axis(ax) if ax < joy.get_numaxes() else 0.0
            out.append(0.0 if abs(v) < args.joy_dead else sg * v)
        return np.clip(np.array(out, np.float32), -1, 1)

    font = pygame.font.SysFont('consolas', 18)
    small = pygame.font.SysFont('consolas', 15)
    info = {'pieces_in_bucket': 0, 'pieces_in_hopper': 0}
    done = False
    while True:
        for ev in pygame.event.get():
            key = ev.key if ev.type == pygame.KEYDOWN else None
            btn = ev.button if ev.type == pygame.JOYBUTTONDOWN else None
            if ev.type == pygame.QUIT or key == pygame.K_ESCAPE:
                save()
                env.close()
                pygame.quit()
                return
            if key == pygame.K_c:
                cam = (cam + 1) % len(CAMERAS)
            if key == pygame.K_h:
                panel = not panel
            if key == pygame.K_m:  # switch drive mode: start the other mode from rest (neutral, pedal up)
                mode = 'pedal' if mode == 'lever' else 'lever'
                pedal.reset()
            if key == pygame.K_b or btn == 1:
                park = not park
            if mode == 'pedal':
                if key == pygame.K_e or btn == 5:
                    pedal.shift(+1)
                if key == pygame.K_q or btn == 4:
                    pedal.shift(-1)
            if key == pygame.K_p or btn == 0:
                save()
                saved = False  # keep driving; a later save writes a new, longer file
            if key == pygame.K_r or btn == 7:
                save()
                seed += 1
                env.reset(seed=seed)
                rec.begin(env, seed)
                saved, done, park = False, False, False
                pedal.reset()
        k = pygame.key.get_pressed()
        j = joystick()
        # action: [speed, steer, boom (+ raises), bucket (+ lip up)], each in [-1, 1]
        a = np.array([
            k[pygame.K_w] - k[pygame.K_s],
            k[pygame.K_a] - k[pygame.K_d],
            k[pygame.K_i] - k[pygame.K_k],
            k[pygame.K_j] - k[pygame.K_l],
        ], np.float32)
        a = np.where(a != 0, a, j)  # a held key wins over the stick on that lever
        brake = max(float(k[pygame.K_SPACE]), trig_brake.value())
        if mode == 'pedal':  # W / right trigger = accelerator, S = brake too; the selector gives the direction
            brake = max(brake, float(k[pygame.K_s]))
            a[0] = pedal.update(bool(k[pygame.K_w]), trig_accel.value(), env.dt)
        if env.has_brake:  # v5: + [brake pedal 0..1, parking brake 0 / 1]
            a = np.concatenate([a, [brake, float(park)]]).astype(np.float32)
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
        hud = (f'{CAMERAS[cam]} | seed {seed} step {len(rec.rows)} | action {np.round(a[:4], 2)} | '
               f'bucket {info["pieces_in_bucket"]}  hopper {info["pieces_in_hopper"]}')
        for flag in ('hit_worker', 'hit_obstacle', 'hit_static'):
            if info.get(flag):
                hud += f' | {flag.upper()}'
        if done:
            hud += ' | END (R = new episode)'
        screen.blit(font.render(hud, True, YELLOW, BLACK), (8, 8))
        if panel:
            draw_panel(screen, small, env, a, mode, pedal, brake, park)
        # why the machine does not move, when the operator asks it to
        want = bool(k[pygame.K_w]) or abs(float(j[0])) > 0 or trig_accel.value() > 0
        if env.has_brake and park and want:
            msg = 'PARKING BRAKE ON - press B to release'
        elif env.off_yard():
            msg = 'YARD LIMIT - drive back towards the yard'
        elif mode == 'pedal' and pedal.gear == 0 and want:
            msg = 'NEUTRAL - press E (forward) or Q (reverse)'
        else:
            msg = ''
        if msg:
            t = font.render(msg, True, WHITE, RED)
            screen.blit(t, ((args.width - t.get_width()) // 2, args.height - 40))
        pygame.display.flip()
        time.sleep(max(0.0, env.dt - (time.time() - t0)))


if __name__ == '__main__':
    main()
