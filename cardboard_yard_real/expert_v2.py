"""Expert v2 for CardboardYardReal physics v5: the v1 driver (expert_v1.py, the v4 dataset's expert) plus the
brake pedal. Planning, steering and arm work are v1's; v2 only adds action[4] (brake pedal) and action[5]
(parking brake, always 0) on top of v1's 4 numbers:

- safety stop for a person (v1 sets the drive to 0 and waits for the 1 m/s^2 HST slow-down: 4.7 m from full
  speed): full brake while still rolling (1.4 m), then a light brake held while it waits for the person
- static safety stop (the bucket would reach a column / wall / the hopper in the next 1.2 s): full brake
- ordinary stops and direction changes (v1 asks for drive 0 or the other direction while still rolling, e.g. at
  the end of a path, in front of the pile or the hopper): a soft brake, harder the faster the machine rolls,
  softer with a loaded bucket so the cardboard stays in
- no brake while v1 drives on (same direction as the machine rolls): slowing down for a goal stays with the lever

Needs physics v5 (v4 has no brake: there the two extra numbers are ignored and v2 drives exactly like v1).
"""

import numpy as np

from .expert_v1 import ControllerV1, LoaderExpertPolicyV1


class ControllerV2(ControllerV1):
    BRAKE_EMERGENCY = 1.0  # person / static safety stop
    BRAKE_HOLD = 0.3  # standing still, waiting for a person to pass
    BRAKE_SOFT = (0.25, 0.6)  # ordinary stop: pedal at 0 m/s .. at BRAKE_SOFT_V and faster
    BRAKE_SOFT_V = 2.0  # m/s
    BRAKE_LOADED = 0.6  # ordinary stops with a loaded bucket: this share of the pedal
    ROLL_MIN = 0.08  # m/s: slower than this counts as standing still

    def reset(self):
        super().reset()
        self.brake_stats = {'emergency': 0, 'hold': 0, 'soft': 0}

    def act(self, env):
        n_people, n_static = self.stats['people_stop'], self.stats['static_stop']
        a = super().act(env)
        people_stop = self.stats['people_stop'] > n_people
        static_stop = self.stats['static_stop'] > n_static
        v = env.forward_speed()
        rolling = abs(v) > self.ROLL_MIN
        brake = 0.0
        if (people_stop or static_stop) and rolling:
            brake = self.BRAKE_EMERGENCY
            self.brake_stats['emergency'] += 1
        elif people_stop:
            brake = self.BRAKE_HOLD
            self.brake_stats['hold'] += 1
        elif rolling and (abs(a[0]) < 0.02 or a[0] * v < 0):
            lo, hi = self.BRAKE_SOFT
            brake = lo + (hi - lo) * min(1.0, abs(v) / self.BRAKE_SOFT_V)
            if self.phase in self.LOADED:
                brake *= self.BRAKE_LOADED
            self.brake_stats['soft'] += 1
        return np.concatenate([a, [brake, 0.0]]).astype(np.float32)


class LoaderExpertPolicyV2(LoaderExpertPolicyV1):
    controller_cls = ControllerV2
