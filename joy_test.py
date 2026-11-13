"""Show the joystick's axis and button numbers live, to set manual.py --joy-map / --joy-invert.

    python joy_test.py     (move each stick / press each button and watch which number changes; Esc quits)
"""
import pygame

pygame.init()
screen = pygame.display.set_mode((640, 360))
font = pygame.font.SysFont('consolas', 20)
if not pygame.joystick.get_count():
    print('no joystick found: plug it in, then start again')
    raise SystemExit
joy = pygame.joystick.Joystick(0)
joy.init()
clock = pygame.time.Clock()
while True:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            raise SystemExit
    screen.fill((0, 0, 0))
    lines = [joy.get_name()]
    lines += [f'axis {i}: {joy.get_axis(i):+.2f}' for i in range(joy.get_numaxes())]
    lines += ['buttons down: ' + ' '.join(str(i) for i in range(joy.get_numbuttons()) if joy.get_button(i))]
    for k, t in enumerate(lines):
        screen.blit(font.render(t, True, (255, 255, 0)), (12, 12 + 26 * k))
    pygame.display.flip()
    clock.tick(30)
