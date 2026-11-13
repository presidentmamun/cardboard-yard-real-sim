"""Komatsu WA100-8 dimensions. Body: measured from the manufacturer's side photo (WA100-8_03_1440x810.jpg, user's
Downloads); wheels, widths and the whole linkage: Komatsu's dimension drawing WA100-8.DXF (wa100_model.py pins).

Scale: overall length 6.245 m and height 3.035 m (used-machine listing, biglemon.kenkey.jp/en/item/109907) both give
193.5 px/m on that photo. Frame as in the simulator: x forward (bucket side), z up, origin on the ground midway
between the axles (articulation). Photo pixel (u, v) -> x = (768.75 - u) / 193.5, z = (714 - v) / 193.5.
The photo shows the boom slightly raised and the bucket racked back, not on the ground.
"""

PX_PER_M = 193.5
U0, V0 = 768.75, 714.0  # articulation (between the hubs at u 517.5 / 1020) and ground line


def px(u, v):
    return ((U0 - u) / PX_PER_M, (V0 - v) / PX_PER_M)


# --- wheels (17.5-25 class tyres; photo: diameter 253 px, rim 130 px)
WHEELBASE = 2.60
TYRE_R = 0.635  # drawn 0.65 unloaded; the axle is 0.61 m high (loaded)
HUB_Z = 0.61
RIM_R = 0.336
TYRE_W = 0.47  # drawing: 2250 over the tyres - 1780 tread
TREAD = 1.78  # drawing

# --- cab (rear frame)
CAB_ROOF_Z = 3.03
CAB_ROOF_X = (0.083, -1.22)  # drawing
CAB_WIND_BOTTOM = (0.433, 2.0)  # the windshield leans forward: bottom edge ahead of the top (drawing)
CAB_FLOOR_Z = 1.36
CAB_DOOR_X = (-0.16, -0.86)
CAB_W = 1.32  # drawing front view: cab walls +-0.65..0.68

# --- engine hood, rear
HOOD_TOP_Z = 2.28  # drawing
HOOD_X = (-1.25, -2.5)  # flat top (drawing)
HOOD_REAR_X = -2.88  # nearly vertical rear face (big grille); drawing (the side photo's perspective shortened the rear)
HOOD_BOTTOM_Z = 1.2
HOOD_W = 1.43  # drawing top view: +-0.716
BUMPER_HW = 0.69  # drawing top view
BUMPER_X = -2.98  # drawing: rear overhang 1680 from the rear axle
BUMPER_Z = (0.82, 1.17)
EXHAUST = (-1.98, 2.78)  # top of the curved stack (drawing)
PRECLEANER = (-1.47, 2.53)  # drawing
DECAL_Z = (1.88, 2.08)
DECAL_X = (-1.6, -2.2)

# --- front: boom and tilt linkage (photo pose)
BOOM_PIVOT = (0.59, 1.70)
BUCKET_HINGE = (2.27, 0.86)  # estimate (hidden behind the bucket ear)
BOOM_LEN = 1.88
CRANK_PIVOT = (2.11, 1.37)
CRANK_TOP = (2.16, 1.91)
TILT_BASE = (0.92, 1.82)
WORK_LAMP = ((0.78, 0.98), (1.72, 1.98))  # drawing  # x range, z range of the lamp box on the tower
BUCKET_W = 2.34  # drawing
BUCKET_SCALE = 1.0  # size of the bucket shape about its hinge (the photo machine may carry a bigger bucket)
