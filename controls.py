"""Operator controls for manual.py: turn pedals / selector into the simulator's drive lever value.

Two drive modes:
  lever  one drive lever, forward (+) / back (-): the v4 way (W / S, left stick).
  pedal  like the real HST machine: a direction selector F / N / R plus an accelerator pedal (0..1).
         drive = direction x accelerator. Releasing the accelerator slows the machine down (HST), the brake
         pedal stops it faster.

Only the resulting action is saved in a drive's CSV, so both modes replay the same way.
"""

GEAR_NAMES = {-1: 'R', 0: 'N', 1: 'F'}


class PedalDrive:
    """Direction selector + accelerator pedal. A key is on / off, so a held key moves the pedal down over
    `rise` seconds and lets it up over `fall` seconds; an analog pedal (gamepad trigger) is used as it is."""

    def __init__(self, rise=0.8, fall=0.25):
        self.rise, self.fall = rise, fall
        self.gear = 0  # -1 reverse, 0 neutral, +1 forward
        self.pedal = 0.0

    def reset(self):
        self.gear, self.pedal = 0, 0.0

    def shift(self, step):
        """step +1: R -> N -> F, step -1: F -> N -> R."""
        self.gear = max(-1, min(1, self.gear + step))

    def update(self, key_down, analog, dt):
        """key_down: accelerator key held. analog: trigger value 0..1 (0 = not used). -> drive lever in [-1, 1]."""
        if analog > 0.0:
            self.pedal = min(1.0, analog)
        elif key_down:
            self.pedal = min(1.0, self.pedal + dt / self.rise)
        else:
            self.pedal = max(0.0, self.pedal - dt / self.fall)
        return self.gear * self.pedal


class Trigger:
    """Gamepad trigger axis -> 0..1. Triggers rest at -1 and read +1 fully pressed, but report 0 until they are
    first moved: the axis counts only after it was seen at rest (below -0.5), so an untouched trigger is 0."""

    def __init__(self, joy, axis):
        self.joy, self.axis, self.seen = joy, axis, False

    def value(self):
        if self.joy is None or self.axis < 0 or self.axis >= self.joy.get_numaxes():
            return 0.0
        raw = self.joy.get_axis(self.axis)
        if raw < -0.5:
            self.seen = True
        if not self.seen:
            return 0.0
        v = (raw + 1.0) / 2.0
        return 0.0 if v < 0.05 else min(1.0, v)
