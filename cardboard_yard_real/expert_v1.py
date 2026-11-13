"""Expert v1 for CardboardYardReal: the v0 driver with smoother driving and the dumping seen in the site video
(E:/office-projects/video_behavior_notes.md). The v0 expert (data of swm/CardboardYard-v0) is unchanged.

- gentler steering: lower pursuit gain, gas and articulation rate-limited in every phase (v0: only when loaded)
- stops pushing a little before the wall, so the bucket lip does not scrape it
- raises the boom fully earlier on the way to the hopper, stops a little before it (tires off the hopper wall)
- dumping: tips and holds longer, then rocks the bucket in small, quicker moves
- pile stops the push with a small load: lifts the boom a little and racks the bucket back while pushing on
- re-plans after 1.0 m off the path (WA100 plans with 85% of full lock); no gas while curling at the push limit
- plans around walking people within 8 m too (v0 planned through them and then waited for the safety stop)
- blocked by a person on every re-plan: makes room after a while (v0 kept re-planning the same path)
- (BACKOUT_TURN, V-pattern back-out: off, spilled loads and swung the rear into columns)
- at a drive goal accepts 40 deg heading error (the next phase lines up) instead of re-planning for the last degrees
- static safety stop: stops and re-plans when the next 1.2 s would bring the bucket to a column / wall / the
  hopper (driving between pile and hopper)
- backs off and re-plans when it touches a column / wall / the hopper on the way
- plans with the bucket at carry height (a raised bucket was planned over the hopper wall, then lowered)
- does not swing the articulation standing still when the bucket would sweep into a column / wall (swings
  on the move instead, as planned)
- walking people right beside the loader are left out of the plan (the safety stop has them)
- the safety stop predicts the walking people 2 s ahead (v0: where they are now) and slows down near them;
  the plan keeps their current places (their predicted tracks walled the lanes off: plans along the columns,
  column hits in 7 of 48 seeds, a deadlock in seed 10)
"""

from cardboard_yard.expert_policy import LoaderExpertPolicy, _Controller


class ControllerV1(_Controller):
    PURSUIT_GAIN = 1.4
    SLEW_ALWAYS = True
    THR_SLEW_EMPTY = 0.06  # empty bucket: quicker on the gas than loaded (THR_SLEW 0.025)
    STEER_SLEW = 0.06
    PUSH_STOP = 0.3
    RAISE_FROM = 7.0
    STOP_EARLY = 0.2
    DUMP_HOLD, DUMP_ROCK_PERIOD, DUMP_ROCK = 18, 6, 0.15
    DUMP_LEAVE_MIN = 10
    REV_COST = 3.0  # long transport (real layout): turn around and drive forward instead of reversing 15 m
    PLAN_EXPAND = 30000
    CROWD = (0.15, 0.3)  # WA100: the pile stopped the push with half a load (seed 4: 13 pieces)
    CURL_WALL_STOP = True
    ESCAPE_REVERSE = True  # bucket facing a person standing still: every forward escape was stopped too (seed 2)
    REPLAN_DEV = 1.0  # re-plan sooner when a turn runs wide (v0: 1.8 m, the bucket reached a column first)
    PLAN_NEAR_PEOPLE = 8.0  # the safety stop keeps 2.2 m from everyone, so the plan must too (near ones)
    BLOCK_KEEP = True  # a person beside the planned path blocked the loader for good (re-plan, same path, ...)
    BACKOUT_TURN = False  # tried (seed 100: 15 -> 19) but spilled the load and swung the rear wheels into a column (103: 23 -> 0, 1048)
    STATIC_SAFE_PHASES = ('go_pile', 'align_pile', 'go_hopper', 'back_out', 'leave_hopper')  # the turning back-out sweeps
    # the bucket too; align_pile drives through the column row; leave_hopper reverses at 0.6 (seed 211 of the fresh
    # seeds: counterweight on a pallet behind)
    PILE_ROW_GAP = 4.0  # WA100 layout: go_pile goal (y 2.1) stood in the column row (y 2.7): seeds 110 / 1048 turned
    # 500-900 steps between the columns; the goal now is 4 m south of the row and align_pile drives straight through
    BACKOUT_ROW_GAP = 3.5  # back out through the column row (bucket ~2.9 m ahead of the articulation clears it), then
    # turn to the hopper in the open (seeds 116 / 147: U-turned east above the row, 500 steps at col 17.75)
    HOPPER_ROW_FENCE = True  # after that back-out the plans still went north through the row to turn (101 / 116 / 147)
    CRUISE_V = 0.75  # ~2.25 m/s on open stretches with nobody within 10 m (trips ~1000 steps: 2 per episode at best)
    BACKOUT_V = 0.42  # reverse off the pile (10 m since BACKOUT_ROW_GAP: 145 steps at 0.3)
    LANE_COL_CLEAR = 0.3  # lane x keeps the bucket edge 0.3 m off those columns (pile lanes reached 8.4, column face 9.4)
    GOAL_YAW_TOL = 0.7  # seed 1048: 500 steps re-planning at the hopper goal for the last 10 degrees
    STATIC_SAFE = True  # seed 103: tracking with the articulation off the plan swung the bucket into a column
    CONTACT_ESCAPE = True  # seed 27: bucket on a column for 1900 steps (crept, so never 'stuck')
    PLAN_LOW_BOOM = True  # seed 45: planned with the bucket raised after a start / dump, lowered onto the hopper wall
    SWING_CHECK = True  # seed 23 (people test): swung the articulation at a cusp, bucket into a column
    PLAN_SKIP_CLOSE = 2.8  # seed 110: a person beside the loader put every plan's start in collision (escape loops)
    PEOPLE_AHEAD = 2.0  # walking people (v4: 1.0-1.5 m/s) reached the stop zone before the loader could stop


class LoaderExpertPolicyV1(LoaderExpertPolicy):
    controller_cls = ControllerV1
