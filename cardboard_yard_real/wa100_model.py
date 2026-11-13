"""Komatsu WA100-8 as in the manufacturer's photos: true dimensions (wa100_dims.py), clean paint. Showroom model
(scripts/examples/wa100_showroom.py), not yet tied to the simulator physics (its boom, wheels and pivots differ).

All points are in the photo pose of WA100-8_03 (boom slightly raised, bucket racked back), frame of wa100_dims:
x forward, z up, origin on the ground between the axles. Each moving body's parts are given relative to its pivot,
so joint angle 0 = the photo pose.
"""

import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import meshes as MS
from . import wa100_dims as D
from .loader_model import ASSETS, cyl_between, loft, tire_l3, wheel_disc

YELLOW = (247, 190, 0)  # Komatsu yellow in the photos

MATERIALS = {
    # name: (texture file or None, rgba, specular, shininess)
    'wa_paint': (None, f'{YELLOW[0] / 255:.3f} {YELLOW[1] / 255:.3f} {YELLOW[2] / 255:.3f} 1', 0.5, 0.7),
    'wa_black': (None, '0.07 0.07 0.075 1', 0.25, 0.3),
    'wa_dark': (None, '0.17 0.18 0.19 1', 0.25, 0.3),  # dark grey front plate, hood top cover
    'wa_glass': (None, '0.3 0.37 0.42 0.55', 0.9, 0.95),  # tinted; the photos show the interior in the side view
    # cylinder rods: polished steel. A light base colour (0.82) rendered plain white under the hall light; a darker
    # base with a full, tight highlight reads as shiny metal (MuJoCo has no reflections on curved geoms)
    'wa_chrome': (None, '0.42 0.44 0.47 1', 1.0, 1.0),
    'wa_lamp': (None, '0.97 0.97 0.95 1', 0.8, 0.9),
    'wa_tail': (None, '0.8 0.06 0.04 1', 0.6, 0.7),
    'wa_amber': (None, '1.0 0.5 0.05 1', 0.6, 0.7),
    'wa_rubber': (None, '0.1 0.1 0.105 1', 0.3, 0.25),
    'wa_steel': (None, '0.55 0.55 0.53 1', 0.4, 0.4),
    'wa_komatsu': ('wa_komatsu.png', '1 1 1 1', 0.45, 0.6),
    'wa_plate': ('wa_plate.png', '1 1 1 1', 0.3, 0.3),
    'wa_door': ('wa_door.png', '1 1 1 1', 0.3, 0.3),
}


def make_textures():
    """Decal textures: KOMATSU blue on yellow (lift arm), the hood plate (3 blue stripes + WA100 on dark grey),
    KOMATSU white on black (cab door panel)."""
    def font(size):
        for f in ('C:/Windows/Fonts/ariblk.ttf', 'C:/Windows/Fonts/arialbd.ttf'):
            if os.path.exists(f):
                return ImageFont.truetype(f, size)
        return ImageFont.load_default()

    def centred(d, box, text, f, fill):
        b = d.textbbox((0, 0), text, font=f)
        x0, y0, x1, y1 = box
        d.text(((x0 + x1 - (b[2] - b[0])) / 2 - b[0], (y0 + y1 - (b[3] - b[1])) / 2 - b[1]), text, font=f, fill=fill)

    out = {}
    # KOMATSU: the logo outline of Komatsu's drawing (letter shapes, the tick on the T), not a font
    logo = Image.open(os.path.join(ASSETS, 'wa_logo_mask.png')).convert('L')  # black letters on white

    def logo_tex(bg, fg, w=1100, h=220, pad=0.04):
        im = Image.new('RGB', (w, h), bg)
        lw = int(w * (1 - 2 * pad))
        lh = int(lw * logo.height / logo.width)
        m = logo.resize((lw, lh), Image.LANCZOS).point(lambda v: 255 - v)  # letters = 255
        im.paste(Image.new('RGB', (lw, lh), fg), ((w - lw) // 2, (h - lh) // 2), m)
        return im

    out['wa_komatsu.png'] = logo_tex(YELLOW, (20, 40, 130))
    im = Image.new('RGB', (1024, 256), YELLOW)  # stripes on the paint, then the dark plate with the model name
    d = ImageDraw.Draw(im)
    for k in range(3):
        d.rectangle((10, 30 + 72 * k, 250, 80 + 72 * k), fill=(20, 40, 130))
    d.rectangle((280, 20, 1014, 236), fill=(45, 46, 48))
    f = font(170)
    b = d.textbbox((0, 0), 'WA', font=f)
    d.text((330 - b[0], 128 - (b[3] + b[1]) / 2), 'WA', font=f, fill=YELLOW)
    d.text((330 - b[0] + (b[2] - b[0]) + 10, 128 - (b[3] + b[1]) / 2), '100', font=f, fill=(245, 245, 245))
    out['wa_plate.png'] = im
    out['wa_door.png'] = logo_tex((18, 18, 20), (215, 215, 215), pad=0.08)
    paths = {}
    for name, im in out.items():
        p = os.path.join(ASSETS, name)
        im.save(p)
        paths[name] = p
    return paths


def _shoulder_lugs(R, hw, bars=22, depth=0.13, lug=0.05):
    """Lug blocks on both sidewalls continuing each tread bar down the tyre side (the toothed outline of the L3
    tyre in the side photos)."""
    m = MS.Mesh()
    pitch = 2 * math.pi / bars
    for k in range(bars):
        for side in (-1, 1):
            a0 = k * pitch + (pitch / 2 if side > 0 else 0.0) + 0.2
            blk = MS.box(0.05, lug / 2, depth / 2).moved(0, side * (hw + lug / 2 - 0.02), R - lug - depth / 2 + 0.02)  # proud of the sidewall
            m.add(blk.moved(0, 0, 0, rot=MS.rot_y(-a0)))
    return m


def _side(poly, y0, y1):
    return MS.extrude(poly, y0, y1)


def _decal_rot(center, half_w, half_h, normal, angle):
    """MS.decal rotated about the y axis by angle (rad, + turns +x towards -z) around its centre."""
    c = np.asarray(center, float)
    m = MS.decal((0, 0, 0), half_w, half_h, normal)
    return m.moved(0, 0, 0, rot=MS.rot_y(angle)).moved(*c)


# --------------------------------------------------------------------- linkage (DXF, lowered pose = joint angles 0)
# pins from the Komatsu WA100-8 dimension drawing (scripts/examples/wa100_dxf_export.py), bucket on the ground
PIVOT = np.array([0.637, 1.804])  # boom pivot on the front frame
HINGE = np.array([2.375, 0.114])  # bucket hinge at the boom tips
LEVER_P = np.array([2.093, 1.031])  # tilt lever pivot (bracket on top of the boom's front cross member)
LEVER_T = np.array([2.096, 1.621])  # tilt cylinder rod end on the lever (top arm)
LEVER_C = np.array([1.780, 0.410])  # lever bottom pin -> link (the bottom arm points down and back)
LINK_D = np.array([2.382, 0.459])  # link pin on the bucket
TILT_A = np.array([0.933, 1.772])  # tilt cylinder base on the front frame
LIFT_A = np.array([0.552, 1.347])  # lift cylinder base on the front frame
LIFT_B = np.array([1.471, 0.853])  # lift cylinder rod end under the arms
RAISED = {'hinge': (2.383, 3.485), 'lever_p': (1.457, 3.234), 'lift_b': (1.615, 2.606)}  # drawing, boom fully raised
DECAL_ARM = (1.254, 1.398, 0.0207, 0.443)  # KOMATSU on the lift arms: centre x, z (lowered pose), extra angle, half length
LIFT_Y, ARM_Y = 0.36, 0.53  # lift cylinders / lift arm centre planes (|y|; front view of the drawing)
_PARTS = json.load(open(os.path.join(ASSETS, 'wa100_dxf_parts.json')))  # lever / link outlines (lowered pose)


def build():
    """{body: [(name, material, Mesh), ...]} with bodies rear, front, boom (origin PIVOT), bucket (origin HINGE),
    wheel (hub at the origin, axis y)."""
    P = {'rear': [], 'front': [], 'boom': [], 'bucket': [], 'wheel': []}
    add = lambda body, name, mat, m: P[body].append((name, mat, m))

    # ===== rear frame =====
    add('rear', 'chassis', 'wa_black', loft([(-2.62, 0.4, 0.66, 1.22, 0.05), (-0.22, 0.4, 0.66, 1.22, 0.05)]))
    # engine hood: tall box, flat top, rounded vertical rear corners, the rear face almost vertical
    hx0, hx1 = D.HOOD_X
    HW = D.HOOD_W / 2
    # (seen from above the rear corners are rounded with ~0.25 m radius: the half width shrinks towards the rear)
    add('rear', 'hood', 'wa_paint', loft([
        (D.HOOD_REAR_X - 0.005, 0.56, 1.24, 2.1, 0.2), (-2.86, 0.62, 1.215, 2.16, 0.18), (-2.82, 0.66, 1.2, 2.2, 0.15),
        (-2.75, 0.69, 1.2, 2.23, 0.11), (-2.62, 0.71, 1.2, 2.265, 0.08), (hx1, HW, 1.2, D.HOOD_TOP_Z, 0.05),
        (hx0, HW, 1.2, D.HOOD_TOP_Z, 0.05)], n=8))
    add('rear', 'hood_top', 'wa_dark', loft([(-1.98, HW - 0.07, D.HOOD_TOP_Z - 0.02, D.HOOD_TOP_Z + 0.015, 0.01),
                                             (hx0 - 0.02, HW - 0.07, D.HOOD_TOP_Z - 0.02, D.HOOD_TOP_Z + 0.015, 0.01)]))
    louv = MS.Mesh()
    for k in range(6):  # top cover slots, side louver blocks
        louv.add(MS.box(0.06, 0.4, 0.004).moved(-1.32 - 0.11 * k, 0, D.HOOD_TOP_Z + 0.018))
    for s in (-1, 1):
        for x0, x1, z0, z1 in ((-1.955, -2.135, 1.36, 1.86), (-2.26, -2.34, 1.6, 1.86)):
            for k in range(int((z1 - z0) / 0.045)):
                louv.add(MS.box(abs(x1 - x0) / 2, 0.006, 0.012).moved((x0 + x1) / 2, s * (HW + 0.003), z0 + 0.03 + 0.045 * k))
        for k in range(5):
            louv.add(MS.box(0.04, 0.006, 0.006).moved(-1.37 - 0.1 * k, s * (HW + 0.003), 2.14))
    for s in (-1, 1):  # panel seams: the side door gap, the diagonal panel edge
        louv.add(MS.box(0.004, 0.006, 0.47).moved(-2.43, s * (HW + 0.003), 1.72))
        louv.add(cyl_between((-1.38, s * (HW + 0.003), 2.0), (-1.62, s * (HW + 0.003), 1.26), 0.004))
    add('rear', 'louvers', 'wa_black', louv)
    # rear: big black grille, round lamps above it, bumper / counterweight with tail lamps
    gr = MS.Mesh()  # nearly the whole rear face: black surround, 10 bars, 3 posts
    gx = D.HOOD_REAR_X - 0.01
    for k in range(10):
        gr.add(MS.box(0.008, 0.42, 0.02).moved(gx + 0.002, 0, 1.36 + 0.08 * k))
    for y in (-0.42, -0.14, 0.14, 0.42):
        gr.add(MS.box(0.01, 0.025, 0.41).moved(gx + 0.002, y, 1.72))
    add('rear', 'grille', 'wa_black', gr)
    add('rear', 'radiator', 'wa_dark', MS.box(0.004, 0.43, 0.41).moved(gx + 0.008, 0, 1.72))  # behind the bars
    lamps = MS.Mesh()
    for s in (-1, 1):
        lamps.add(cyl_between((gx - 0.02, s * 0.36, 2.06), (gx + 0.05, s * 0.36, 2.06), 0.055))
    add('rear', 'rear_lamp_rings', 'wa_black', lamps)
    lens = MS.Mesh()
    for s in (-1, 1):
        lens.add(cyl_between((gx - 0.03, s * 0.36, 2.06), (gx - 0.02, s * 0.36, 2.06), 0.042))
    add('rear', 'rear_lamps', 'wa_lamp', lens)
    # bumper / counterweight, and the rear frame's underside rising from behind the rear tyres (drawing)
    add('rear', 'bumper', 'wa_black', loft([(D.BUMPER_X, D.BUMPER_HW, D.BUMPER_Z[0], D.BUMPER_Z[1], 0.04), (-2.73, D.BUMPER_HW, 0.72, D.BUMPER_Z[1], 0.04),
                                            (-2.4, 0.6, 0.61, D.BUMPER_Z[1], 0.04), (-2.0, 0.42, 0.47, D.BUMPER_Z[1], 0.04)]))
    tl = MS.Mesh()
    for s in (-1, 1):
        tl.add(MS.box(0.012, 0.045, 0.07).moved(D.BUMPER_X - 0.008, s * (D.BUMPER_HW - 0.06), 1.06))
    add('rear', 'tail_lamps', 'wa_tail', tl)
    ex = MS.Mesh()  # exhaust stack (curves to the rear at the top) and the pre-cleaner
    ex_y = 0.28
    ex_x, ex_z = D.EXHAUST
    pts = [(ex_x + 0.11, D.HOOD_TOP_Z), (ex_x + 0.11, ex_z - 0.13), (ex_x + 0.095, ex_z - 0.055), (ex_x + 0.055, ex_z - 0.01), (ex_x, ex_z)]
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        ex.add(cyl_between((x0, ex_y, z0), (x1, ex_y, z1), 0.075))
    ex.add(cyl_between((ex_x + 0.11, ex_y, D.HOOD_TOP_Z), (ex_x + 0.11, ex_y, D.HOOD_TOP_Z + 0.06), 0.12))  # base flange
    px_, pz_ = D.PRECLEANER  # drawing: stem r 0.07, cap r 0.16 at 2.47..2.55
    ex.add(cyl_between((px_, -0.22, D.HOOD_TOP_Z), (px_, -0.22, pz_ - 0.07), 0.07))
    ex.add(cyl_between((px_, -0.22, pz_ - 0.08), (px_, -0.22, pz_ - 0.02), 0.16))
    ex.add(cyl_between((px_, -0.22, pz_ - 0.02), (px_, -0.22, pz_ + 0.01), 0.11))
    add('rear', 'stacks', 'wa_black', ex)
    mir = MS.Mesh()  # mirrors on long stalks at the rear of the hood, and on the cab front pillars
    for s in (-1, 1):
        mir.add(cyl_between((-2.32, s * 0.6, D.HOOD_TOP_Z), (-2.72, s * 0.88, 2.58), 0.014))  # hood mirror (drawing top view: -2.75, +-0.9)
        mir.add(cyl_between((-2.77, s * 0.88, 2.59), (-2.77, s * 0.93, 2.59), 0.095))  # round mirror facing sideways (drawing: 0.21 x 0.17 seen from the side)
        mir.add(cyl_between((0.02, s * 0.64, 2.82), (-0.04, s * 0.92, 2.72), 0.016))  # cab mirror arm from the front pillar
        mir.add(cyl_between((-0.04, s * 0.92, 2.72), (-0.05, s * 0.96, 2.66), 0.016))
        mir.add(MS.box(0.025, 0.09, 0.17).moved(-0.06, s * 1.0, 2.62))  # tall rectangular mirror (drawing: 0.34 m)
    add('rear', 'mirrors', 'wa_black', mir)
    fen = MS.Mesh()  # rear fenders: flat black plates over the rear tyres
    for s in (-1, 1):
        fen.add(MS.box(0.55, 0.26, 0.02).moved(-1.3, s * 0.88, 1.43))
        fen.add(MS.box(0.02, 0.26, 0.12).moved(-0.77, s * 0.88, 1.33))
    add('rear', 'rear_fenders', 'wa_black', fen)
    # cab: yellow lower body, large glass, black pillars, yellow roof with an overhang
    cx0, cx1 = D.CAB_ROOF_X
    wb = D.CAB_WIND_BOTTOM
    cw = D.CAB_W / 2
    # lower cab: yellow sides and door, the front below the windshield is glass down to 1.72 m, black below it
    # (a shell, not a solid block: the interior shows through the glass)
    lower = MS.Mesh()
    for s in (-1, 1):
        lower.add(_side([(wb[0] - 0.04, D.CAB_FLOOR_Z), (wb[0] - 0.06, wb[1]), (cx1, wb[1]), (cx1, D.CAB_FLOOR_Z)], s * cw - 0.02, s * cw + 0.0) if s > 0
                  else _side([(wb[0] - 0.04, D.CAB_FLOOR_Z), (wb[0] - 0.06, wb[1]), (cx1, wb[1]), (cx1, D.CAB_FLOOR_Z)], -cw, -cw + 0.02))
    lower.add(MS.box(0.02, cw, (wb[1] - D.CAB_FLOOR_Z) / 2).moved(cx1 + 0.02, 0, (wb[1] + D.CAB_FLOOR_Z) / 2))
    add('rear', 'cab_lower', 'wa_paint', lower)
    inner = MS.Mesh()  # floor, seat, steering column and wheel, side console
    inner.add(MS.box((wb[0] - cx1) / 2 - 0.05, cw - 0.03, 0.02).moved((wb[0] + cx1) / 2, 0, 1.62))
    inner.add(MS.rounded_box(0.24, 0.25, 0.08, e=0.4).moved(-0.62, 0, 1.92))
    inner.add(MS.rounded_box(0.07, 0.24, 0.32, e=0.4).moved(-0.86, 0, 2.28))
    inner.add(MS.box(0.06, 0.12, 0.25).moved(-0.6, 0, 1.75))
    inner.add(cyl_between((0.28, 0, 1.75), (0.12, 0, 2.18), 0.05))
    ax = np.array([-0.16, 0.0, 0.43]) / np.linalg.norm([-0.16, 0.0, 0.43])  # steering column direction
    u, v, c = np.array([ax[2], 0.0, -ax[0]]), np.array([0.0, 1.0, 0.0]), np.array([0.13, 0.0, 2.215])
    ring = [c + 0.19 * (math.cos(a) * u + math.sin(a) * v) for a in np.linspace(0, 2 * math.pi, 17)]
    for p0, p1 in zip(ring[:-1], ring[1:]):  # steering wheel rim (a ring: the driver looks over and through it)
        inner.add(cyl_between(tuple(p0), tuple(p1), 0.016))
    for a in (0.0, 2.1, 4.2):  # spokes
        inner.add(cyl_between(tuple(c), tuple(c + 0.19 * (math.cos(a) * u + math.sin(a) * v)), 0.012))
    inner.add(MS.box(0.25, 0.08, 0.12).moved(-0.4, -0.42, 1.95))
    inner.add(MS.box(0.12, 0.38, 0.12).moved(0.3, 0, 1.78))
    add('rear', 'cab_interior', 'wa_dark', inner)
    add('rear', 'cab_front_glass', 'wa_glass', _side([(wb[0], 1.72), (wb[0] + 0.005, wb[1]), (wb[0] - 0.035, wb[1]), (wb[0] - 0.035, 1.72)], -cw + 0.06, cw - 0.06))
    add('rear', 'cab_front_low', 'wa_black', _side([(wb[0] + 0.005, D.CAB_FLOOR_Z), (wb[0], 1.72), (wb[0] - 0.035, 1.72), (wb[0] - 0.035, D.CAB_FLOOR_Z)], -cw + 0.02, cw - 0.02))
    add('rear', 'cab_band', 'wa_black', _side([(wb[0] + 0.005, wb[1] - 0.14), (wb[0] - 0.015, wb[1] + 0.02),
                                              (cx1 - 0.005, wb[1] + 0.02), (cx1 - 0.005, wb[1] - 0.14)], -cw - 0.005, cw + 0.005))
    add('rear', 'cab_glass', 'wa_glass', _side([(wb[0] - 0.02, wb[1]), (cx1, wb[1]), (cx1, D.CAB_ROOF_Z - 0.05), (cx0, D.CAB_ROOF_Z - 0.05)],
                                               -cw + 0.01, cw - 0.01))
    pil = MS.Mesh()
    for s in (-1, 1):
        y = s * (cw - 0.025)
        pil.add(cyl_between((wb[0] - 0.02, y, wb[1]), (cx0, y, D.CAB_ROOF_Z - 0.04), 0.035))  # A pillar
        pil.add(MS.box(0.035, 0.03, (D.CAB_ROOF_Z - wb[1]) / 2).moved(cx1 + 0.03, y, (D.CAB_ROOF_Z + wb[1]) / 2))  # rear
        pil.add(MS.box(0.018, 0.02, (D.CAB_ROOF_Z - wb[1]) / 2).moved(D.CAB_DOOR_X[0], y, (D.CAB_ROOF_Z + wb[1]) / 2))  # door frame
        pil.add(MS.box(0.025, 0.025, (D.CAB_ROOF_Z - wb[1]) / 2).moved(D.CAB_DOOR_X[1], y, (D.CAB_ROOF_Z + wb[1]) / 2))
        pil.add(MS.box(0.35, 0.012, 0.012).moved(-0.5, y, 2.55))  # door bar
    pil.add(MS.box((cx0 - cx1) / 2 + 0.02, cw, 0.03).moved((cx0 + cx1) / 2, 0, D.CAB_ROOF_Z - 0.06))  # roof frame
    add('rear', 'cab_pillars', 'wa_black', pil)
    add('rear', 'cab_roof', 'wa_paint', loft([(cx1 - 0.02, cw + 0.01, D.CAB_ROOF_Z - 0.05, D.CAB_ROOF_Z + 0.01, 0.025),
                                              (cx0 + 0.0, cw + 0.01, D.CAB_ROOF_Z - 0.05, D.CAB_ROOF_Z + 0.01, 0.025)]))
    rl = MS.Mesh()
    for s in (-1, 1):  # cab work lamps on the front roof corners, antenna
        rl.add(MS.box(0.05, 0.06, 0.05).moved(0.12, s * (cw - 0.05), D.CAB_ROOF_Z - 0.12))
    rl.add(cyl_between((cx1 + 0.1, 0.3, D.CAB_ROOF_Z), (cx1 + 0.1, 0.3, D.CAB_ROOF_Z + 0.3), 0.006))
    add('rear', 'cab_lamps', 'wa_black', rl)
    st = MS.Mesh()  # steps and handrail at the cab door (both sides)
    for s in (-1, 1):
        y = s * 0.75
        for z in (0.55, 0.9, 1.22):
            st.add(MS.box(0.17, 0.09, 0.012).moved(-0.08, y, z))
        st.add(MS.box(0.02, 0.09, 0.4).moved(0.09, y, 0.88))  # side plates of the step assembly
        st.add(MS.box(0.02, 0.09, 0.4).moved(-0.26, y, 0.88))
        st.add(MS.box(0.17, 0.006, 0.4).moved(-0.08, s * 0.66, 0.88))  # back plate (closes the ladder against the frame)
        st.add(cyl_between((-0.95, s * (cw + 0.04), 1.45), (-0.95, s * (cw + 0.04), 2.2), 0.014))
    add('rear', 'steps', 'wa_black', st)
    under = MS.Mesh()
    under.add(MS.box(0.38, 0.3, 0.36).moved(-0.1, 0, 0.88))  # articulation, transmission, drive shaft housing
    under.add(MS.box(0.2, 0.2, 0.1).moved(0.25, 0, 0.62))
    add('rear', 'drive_line', 'wa_black', under)
    add('rear', 'cab_skirt', 'wa_black', MS.box((wb[0] - cx1) / 2, cw + 0.01, 0.05).moved((wb[0] + cx1) / 2, 0, D.CAB_FLOOR_Z + 0.04))
    # (inside the rear tyres: their inner faces are at |y| 0.655)
    add('rear', 'tank', 'wa_black', MS.box(0.4, 0.09, 0.25).moved(-1.0, 0.5, 0.8))
    add('rear', 'battery', 'wa_black', MS.box(0.4, 0.09, 0.25).moved(-1.0, -0.5, 0.8))
    dec = MS.Mesh()
    for s, n in ((1, '+y'), (-1, '-y')):
        dec.add(MS.decal(((D.DECAL_X[0] + D.DECAL_X[1]) / 2 + 0.12, s * (HW + 0.007), sum(D.DECAL_Z) / 2), 0.4, 0.1, n))
    add('rear', 'decal_plate', 'wa_plate', dec)
    dd = MS.Mesh()
    for s, n in ((1, '+y'), (-1, '-y')):
        dd.add(MS.decal((-0.4, s * (cw + 0.012), wb[1] - 0.07), 0.2, 0.2 / 5.0, n))
    add('rear', 'decal_door', 'wa_door', dd)

    # ===== front frame =====
    add('front', 'frame', 'wa_black', loft([(0.05, 0.36, 0.98, 1.25, 0.05), (0.5, 0.36, 0.98, 1.25, 0.05), (0.62, 0.36, 0.66, 1.25, 0.05),
                                            (1.95, 0.36, 0.66, 1.25, 0.05)]))
    tw = MS.Mesh()  # towers carrying the boom pins, inside the lift arms
    for s in (-1, 1):
        tw.add(MS.extrude([(0.12, 0.95), (1.05, 0.95), (0.86, 1.42), (0.76, 1.97), (0.42, 1.97), (0.3, 1.42)], s * 0.39 - 0.03, s * 0.39 + 0.03))
    add('front', 'towers', 'wa_black', tw)
    # dark grey plate between the towers (front view), sloping down to the front, bolts
    add('front', 'front_plate', 'wa_dark', MS.extrude([(0.3, 1.26), (1.25, 1.26), (0.95, 1.86), (0.5, 1.96)], -0.36, 0.36))
    fen = MS.Mesh()  # front fenders: yellow flat-topped covers over the tyres
    for s in (-1, 1):
        y0, y1 = sorted((s * 0.6, s * 1.08))
        # drawing: short flat top x 0.69..1.10 at z 1.42..1.50, and a strut from its rear down to (0.51, 0.94)
        fen.add(MS.extrude([(0.69, 1.5), (1.095, 1.5), (1.095, 1.455), (0.75, 1.42), (0.69, 1.455)], y0, y1))
        ys = sorted((s * 0.6, s * 0.68))
        fen.add(MS.extrude([(0.654, 1.47), (0.508, 1.007), (0.531, 0.937), (0.722, 1.399), (0.753, 1.422)], *ys))
    add('front', 'fenders', 'wa_paint', fen)
    flap = MS.Mesh()
    for s in (-1, 1):
        y0, y1 = sorted((s * 0.62, s * 1.12))
        flap.add(MS.extrude([(0.66, 0.45), (0.69, 0.45), (0.69, 1.42), (0.66, 1.42)], y0, y1))
    add('front', 'mud_flaps', 'wa_black', flap)
    wl = MS.Mesh()  # work lamps: black boxes on stalks out from the towers
    (lx0, lx1), (lz0, lz1) = D.WORK_LAMP
    for s in (-1, 1):
        y = s * 0.95
        wl.add(MS.box((lx1 - lx0) / 2, 0.11, (lz1 - lz0) / 2).moved((lx0 + lx1) / 2, y, (lz0 + lz1) / 2))
        wl.add(cyl_between((0.6, s * 0.42, 1.8), ((lx0 + lx1) / 2, y - s * 0.08, (lz0 + lz1) / 2 - 0.05), 0.025))
    add('front', 'work_lamps', 'wa_black', wl)
    wlens = MS.Mesh()
    for s in (-1, 1):
        wlens.add(MS.box(0.006, 0.07, 0.055).moved(lx1 + 0.004, s * 0.93, (lz0 + lz1) / 2))
    add('front', 'work_lamp_lens', 'wa_lamp', wlens)
    wtail = MS.Mesh()
    for s in (-1, 1):
        wtail.add(MS.box(0.04, 0.006, 0.04).moved(lx0 + 0.05, s * 1.062, (lz0 + lz1) / 2))
    add('front', 'work_lamp_tail', 'wa_tail', wtail)
    lug = MS.Mesh()  # lift cylinder base lugs, tilt cylinder base bracket
    for s in (-1, 1):
        for yy in (s * LIFT_Y - 0.08, s * LIFT_Y + 0.08):
            lug.add(MS.extrude([(LIFT_A[0] - 0.12, LIFT_A[1] - 0.14), (LIFT_A[0] + 0.12, LIFT_A[1] - 0.14),
                                (LIFT_A[0] + 0.07, LIFT_A[1] + 0.08), (LIFT_A[0] - 0.07, LIFT_A[1] + 0.08)], yy - 0.015, yy + 0.015))
        lug.add(cyl_between((LIFT_A[0], s * LIFT_Y - 0.1, LIFT_A[1]), (LIFT_A[0], s * LIFT_Y + 0.1, LIFT_A[1]), 0.03))
    for yy in (-0.1, 0.1):
        lug.add(MS.extrude([(TILT_A[0] - 0.12, TILT_A[1] - 0.18), (TILT_A[0] + 0.1, TILT_A[1] - 0.18),
                            (TILT_A[0] + 0.07, TILT_A[1] + 0.07), (TILT_A[0] - 0.08, TILT_A[1] + 0.07)], yy - 0.015, yy + 0.015))
    lug.add(cyl_between((TILT_A[0], -0.12, TILT_A[1]), (TILT_A[0], 0.12, TILT_A[1]), 0.03))
    add('front', 'cyl_mounts', 'wa_black', lug)

    # ===== boom (origin = PIVOT): twin arms traced from the drawing (lowered pose), cross member with the lever bracket
    o = PIVOT
    rel = lambda pts: [(x - o[0], z - o[1]) for x, z in pts]
    arm = rel([(0.5, 1.8), (0.56, 1.93), (0.93, 1.77), (1.82, 1.2), (1.97, 1.0), (2.12, 0.75), (2.3, 0.42), (2.45, 0.19),
               (2.5, 0.08), (2.44, 0.0), (2.32, 0.02), (2.2, 0.17), (2.0, 0.33), (1.92, 0.42), (1.62, 0.78), (1.4, 0.95),
               (1.17, 1.25), (0.87, 1.6), (0.62, 1.66)])
    arms = MS.Mesh()
    for s in (-1, 1):
        arms.add(MS.extrude(arm, s * ARM_Y - 0.045, s * ARM_Y + 0.045))
    add('boom', 'arms', 'wa_paint', arms)
    bos = MS.Mesh()
    for s in (-1, 1):
        for p in (PIVOT, HINGE):
            q = p - o
            bos.add(cyl_between((q[0], s * ARM_Y - 0.065, q[1]), (q[0], s * ARM_Y + 0.065, q[1]), 0.07))
    add('boom', 'bosses', 'wa_paint', bos)
    # cross member between the arms (front), with the lever bracket on top (drawing: bulge around the lever pivot)
    ct = np.array([2.02, 0.86]) - o
    br = MS.Mesh()
    br.add(cyl_between((ct[0], -ARM_Y, ct[1]), (ct[0], ARM_Y, ct[1]), 0.1))
    lp = LEVER_P - o
    bracket = [(1.93, 0.86), (1.97, 1.02), (2.05, 1.16), (2.2, 1.17), (2.28, 1.05), (2.2, 0.9), (2.08, 0.8)]
    for yy in (-0.11, 0.11):
        br.add(MS.extrude(rel(bracket), yy - 0.018, yy + 0.018))
    lb = LIFT_B - o  # lift cylinder rod-end lugs under the arms
    for s in (-1, 1):
        for yy in (s * LIFT_Y - 0.08, s * LIFT_Y + 0.08):
            br.add(MS.extrude([(lb[0] - 0.13, lb[1] + 0.16), (lb[0] + 0.13, lb[1] + 0.12), (lb[0] + 0.06, lb[1] - 0.07),
                               (lb[0] - 0.07, lb[1] - 0.06)], yy - 0.015, yy + 0.015))
        br.add(MS.box(0.13, (ARM_Y - LIFT_Y) / 2, 0.04).moved(lb[0], s * (ARM_Y + LIFT_Y) / 2, lb[1] + 0.13))
    add('boom', 'brackets', 'wa_paint', br)
    pins = MS.Mesh()
    pins.add(cyl_between((lp[0], -0.14, lp[1]), (lp[0], 0.14, lp[1]), 0.04))
    add('boom', 'pins', 'wa_black', pins)
    dk = MS.Mesh()
    ang = math.atan2(1.77 - 1.2, 1.82 - 0.93)  # along the upper edge of the arm (rot_y + turns +x down)
    for s, n in ((1, '+y'), (-1, '-y')):
        dk.add(_decal_rot((DECAL_ARM[0] - o[0], s * (ARM_Y + 0.047), DECAL_ARM[1] - o[1]), DECAL_ARM[3], DECAL_ARM[3] / 5.0, n, ang + DECAL_ARM[2]))
    add('boom', 'decal', 'wa_komatsu', dk)

    # ===== bucket (origin = HINGE): GP bucket traced from the drawing (lowered pose: floor on the ground) =====
    h = HINGE
    k_ = D.BUCKET_SCALE
    relb = lambda pts: [((x - h[0]) * k_, (z - h[1]) * k_) for x, z in pts]
    W = D.BUCKET_W / 2
    jp = relb([(3.21, 0.0), (2.74, 0.0), (2.6, 0.04), (2.49, 0.15), (2.44, 0.33), (2.46, 0.5), (2.56, 0.62), (2.66, 0.72)])
    add('bucket', 'shell', 'wa_paint', MS.extrude(MS.offset_band(jp, -0.025), -W + 0.02, W - 0.02))
    guard = relb([(2.66, 0.72), (2.72, 0.86), (2.82, 0.99), (2.95, 1.04), (3.03, 1.0)])  # spill guard leaning forward
    add('bucket', 'spill_guard', 'wa_paint', MS.extrude(MS.offset_band(guard, -0.02), -W + 0.02, W - 0.02))
    gside = relb([(2.83, 0.81), (3.03, 1.0), (2.95, 1.045), (2.82, 1.0), (2.72, 0.86), (2.66, 0.74)])
    gs = MS.Mesh()
    for s in (-1, 1):
        gs.add(MS.extrude(gside, s * W - 0.02, s * W + 0.02))
    add('bucket', 'guard_sides', 'wa_paint', gs)
    side = relb([(3.21, 0.0), (2.83, 0.81), (2.66, 0.76), (2.5, 0.62), (2.42, 0.48), (2.4, 0.3), (2.42, 0.12),
                 (2.5, 0.02), (2.74, -0.005)])
    sp = MS.Mesh()
    for s in (-1, 1):
        sp.add(MS.extrude(side, s * W - 0.02, s * W + 0.02))
    add('bucket', 'sides', 'wa_paint', sp)
    edge_band = MS.Mesh()
    for s in (-1, 1):
        edge_band.add(MS.extrude(MS.offset_band(relb([(3.21, 0.0), (2.83, 0.81)]), -0.07), s * W - 0.035, s * W + 0.035))
    add('bucket', 'side_edges', 'wa_paint', edge_band)
    e0 = (np.array([3.21, 0.0]) - h) * k_
    add('bucket', 'edge', 'wa_paint', MS.extrude([tuple(e0), tuple(e0 + (-0.2, 0.0)), tuple(e0 + (-0.2, -0.02)), tuple(e0 + (0.0, -0.015))],
                                                 -W - 0.02, W + 0.02))
    bolts = MS.Mesh()
    for k in range(12):
        y = -W + 0.1 + (2 * W - 0.2) * k / 11
        bolts.add(cyl_between((e0[0] - 0.1, y, e0[1] + 0.002), (e0[0] - 0.1, y, e0[1] + 0.015), 0.012))
    add('bucket', 'bolts', 'wa_black', bolts)
    ears = MS.Mesh()
    ld = LINK_D - h
    for s in (-1, 1):  # hinge ears at the arm tips (behind the back wall), link ears in the middle
        for yy in (s * ARM_Y - 0.07, s * ARM_Y + 0.07):
            ears.add(MS.extrude([(-0.1, -0.1), (0.12, -0.12), (0.12, 0.25), (-0.06, 0.12)], yy - 0.018, yy + 0.018))
    for yy in (-0.13, 0.13):
        ears.add(MS.extrude([(ld[0] - 0.08, ld[1] - 0.1), (0.1, -0.05), (0.1, ld[1] + 0.12), (ld[0] - 0.05, ld[1] + 0.08)],
                            yy - 0.018, yy + 0.018))
    add('bucket', 'ears', 'wa_paint', ears)
    pins = MS.Mesh()
    for s in (-1, 1):
        pins.add(cyl_between((0, s * ARM_Y - 0.1, 0), (0, s * ARM_Y + 0.1, 0), 0.03))
    pins.add(cyl_between((ld[0], -0.16, ld[1]), (ld[0], 0.16, ld[1]), 0.025))
    add('bucket', 'pins', 'wa_black', pins)

    # ===== wheel (one mesh for all four): L3 tyre with deep bars running down the sidewalls, yellow disc wheel =====
    tire_m, lugs_m = tire_l3(D.TYRE_R, D.TYRE_W / 2, bars=22)
    add('wheel', 'tire', 'wa_rubber', tire_m)
    add('wheel', 'tread', 'wa_rubber', lugs_m)
    add('wheel', 'shoulders', 'wa_rubber', _shoulder_lugs(D.TYRE_R, D.TYRE_W / 2, bars=22))
    for nm, mat, m in wheel_disc(D.RIM_R, D.TYRE_W / 2):
        if nm == 'disc_holes':
            continue
        if nm in ('hub', 'nuts'):
            mat = 'ld_paint'  # painted with the wheel in the photos
        add('wheel', nm, {'ld_paint': 'wa_paint', 'ld_black': 'wa_black', 'ld_steel': 'wa_steel', 'ld_grey': 'wa_dark', 'ld_chrome': 'wa_chrome'}.get(mat, mat), m)
    return P
