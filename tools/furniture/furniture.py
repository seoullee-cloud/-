"""Detailed furniture solids (real millimetres, z up). Every builder makes the piece in a local frame:
x = width (0..W), y = depth (0..D), the BACK (wall side) at y = D, the front at y = 0.
Minimum feature ~40 mm real (0.8 mm printed at 1:50)."""
import math
from manifold3d import Manifold as M

SEG = 32
MIN = 40.0

def box(x0, y0, z0, x1, y1, z1):
    return M.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

def rbox(x0, y0, z0, x1, y1, z1, r):
    r = max(1.0, min(r, (x1 - x0) / 2 - 0.01, (y1 - y0) / 2 - 0.01, (z1 - z0) / 2 - 0.01))
    s = M.sphere(r, 16)
    pts = [s.translate([x, y, z]) for x in (x0 + r, x1 - r) for y in (y0 + r, y1 - r) for z in (z0 + r, z1 - r)]
    return M.batch_hull(pts)

def rbox_v(x0, y0, z0, x1, y1, z1, r):
    """box rounded only on vertical edges (flat top/bottom)"""
    r = max(1.0, min(r, (x1 - x0) / 2 - 0.01, (y1 - y0) / 2 - 0.01))
    c = M.cylinder(z1 - z0, r, r, SEG)
    return M.batch_hull([c.translate([x, y, z0]) for x in (x0 + r, x1 - r) for y in (y0 + r, y1 - r)])

def cyl(cx, cy, z0, h, r, r2=None, seg=SEG):
    return M.cylinder(h, r, r if r2 is None else r2, seg).translate([cx, cy, z0])

def ellip(cx, cy, z0, h, rx, ry, seg=48):
    return M.cylinder(h, 1, 1, seg).scale([rx, ry, 1]).translate([cx, cy, z0])

def U(*ms):
    out = ms[0]
    for m in ms[1:]:
        out = out + m
    return out

def leg(x, y, z0, z1, a=MIN, taper=0.75):
    """square leg tapering towards the floor"""
    top = box(x - a / 2, y - a / 2, z1 - 1, x + a / 2, y + a / 2, z1)
    b = max(MIN * 0.9, a * taper)
    bot = box(x - b / 2, y - b / 2, z0, x + b / 2, y + b / 2, z0 + 1)
    return M.batch_hull([top, bot])

# ---------------------------------------------------------------- sleeping / seating
def bed(W, D, pillows=2):
    p = [leg(x, y, 0, 130, 60) for x in (70, W - 70) for y in (70, D - 70)]
    frame = rbox_v(0, 0, 120, W, D - 60, 300, 20)
    matt = rbox(25, 25, 290, W - 25, D - 90, 520, 60)
    head = rbox(0, D - 80, 120, W, D, 1000, 25)
    head = head - box(60, D - 120, 420, W - 60, D - 60, 940)            # upholstered panel recess
    head = head + rbox(70, D - 100, 430, W - 70, D - 50, 930, 30)       # padded panel
    pw = (W - 120 - 60 * (pillows - 1)) / pillows
    pil = [rbox(60 + i * (pw + 60), D - 520, 505, 60 + i * (pw + 60) + pw, D - 120, 630, 60) for i in range(pillows)]
    duvet = rbox(10, 10, 470, W - 10, D - 560, 565, 40)                  # throw over the edges
    fold = M.cylinder(W - 40, 45, 45, 24).rotate([0, 90, 0]).translate([20, D - 600, 545])
    return U(*p, frame, matt, head, *pil, duvet, fold)

def sofa(W, D, seats=2, arm=180, back=200, seat_h=430, back_h=820):
    legs = [leg(x, y, 0, 110, 50) for x in (60, W - 60) for y in (60, D - 60)]
    base = rbox_v(0, 0, 100, W, D, 300, 30)
    arms = [rbox(0, 0, 100, arm, D, 620, 50), rbox(W - arm, 0, 100, W, D, 620, 50)]
    backp = rbox(arm - 10, D - back, 280, W - arm + 10, D, back_h, 60)
    iw = W - 2 * arm; cw = (iw - 20 * (seats - 1)) / seats
    cush, bcush = [], []
    for i in range(seats):
        x0 = arm + i * (cw + 20)
        cush.append(rbox(x0, 20, 290, x0 + cw, D - back + 10, seat_h, 50))
        bcush.append(rbox(x0 + 10, D - back - 110, seat_h - 20, x0 + cw - 10, D - back + 30, back_h - 40, 60))
    return U(*legs, base, *arms, backp, *cush, *bcush)

def armchair(W, D, arm=150, back=180):
    return sofa(W, D, seats=1, arm=arm, back=back)

def dining_table(W, D, H=740):
    top = rbox_v(0, 0, H - 40, W, D, H, 25)
    edge = box(15, 15, H - 45, W - 15, D - 15, H - 39)
    apron = box(70, 70, H - 130, W - 70, D - 70, H - 40) - box(110, 110, H - 140, W - 110, D - 110, H - 30)
    legs = [leg(x, y, 0, H - 40, 60, 0.65) for x in (100, W - 100) for y in (100, D - 100)]
    return U(top, edge, apron, *legs)

def dining_chair(W, D, seat_h=450, H=860):
    inset = 30
    legs = [leg(x, y, 0, seat_h - 30, MIN, 0.9) for x in (inset + 20, W - inset - 20) for y in (inset + 20, D - inset - 20)]
    seat = rbox(inset, inset, seat_h - 40, W - inset, D - inset, seat_h, 15)
    posts = [box(x - 20, D - inset - 40, seat_h - 40, x + 20, D - inset, H - 30) for x in (inset + 20, W - inset - 20)]
    rail = rbox(inset, D - inset - 45, H - 160, W - inset, D - inset + 5, H, 15)
    slat = box(W / 2 - 25, D - inset - 40, seat_h, W / 2 + 25, D - inset, H - 150)
    stretch = [box(inset + 20, y - 20, 160, W - inset - 20, y + 20, 200) for y in (inset + 20, D - inset - 20)]
    return U(*legs, seat, *posts, rail, slat, *stretch)

def desk_chair(W, D, seat_h=470, H=980):
    cx, cy = W / 2, D / 2
    star = []
    for k in range(5):
        a = k * 72 + 90
        arm = box(0, -22, 60, min(W, D) / 2 - 20, 22, 100).rotate([0, 0, a]).translate([cx, cy, 0])
        caster = M.sphere(30, 16).translate([cx + (min(W, D) / 2 - 30) * math.cos(math.radians(a)),
                                             cy + (min(W, D) / 2 - 30) * math.sin(math.radians(a)), 30])
        star += [arm, caster]
    hub = cyl(cx, cy, 55, 60, 50)
    col = cyl(cx, cy, 100, seat_h - 160, 28)
    mech = box(cx - 100, cy - 90, seat_h - 90, cx + 100, cy + 90, seat_h - 50)
    seat = rbox(20, 20, seat_h - 60, W - 20, D - 40, seat_h + 10, 30)
    spine = box(cx - 30, D - 80, seat_h - 70, cx + 30, D - 40, seat_h + 160)
    back = rbox(50, D - 90, seat_h + 90, W - 50, D - 30, H, 40)
    lumbar = rbox(70, D - 120, seat_h + 160, W - 70, D - 60, seat_h + 260, 30)
    arms = [U(box(x - 20, cy - 40, seat_h - 30, x + 20, cy + 40, seat_h + 180),
              rbox(x - 30, cy - 160, seat_h + 170, x + 30, cy + 120, seat_h + 210, 15)) for x in (40, W - 40)]
    return U(*star, hub, col, mech, seat, spine, back, lumbar, *arms)

def lounge_chair(W, D):
    base = rbox_v(30, 30, 0, W - 30, D - 30, 60, 30)
    shell = rbox(0, 0, 60, W, D, 380, 80)
    back = rbox(0, D - 180, 200, W, D, 760, 90)
    arms = [rbox(0, 0, 200, 110, D, 560, 50), rbox(W - 110, 0, 200, W, D, 560, 50)]
    cush = rbox(110, 20, 330, W - 110, D - 170, 430, 50)
    return U(base, shell, back, *arms, cush)

def side_table(W, D, H=520):
    r = min(W, D) / 2
    return U(cyl(W / 2, D / 2, 0, 25, r * 0.6), cyl(W / 2, D / 2, 25, H - 55, 30), cyl(W / 2, D / 2, H - 30, 30, r, seg=48),
             cyl(W / 2, D / 2, H - 40, 12, r - 20, seg=48))

# ---------------------------------------------------------------- work
def monitor(x0, y0, x1, y1, z):
    """screen across the longer side of the plan rectangle"""
    w = x1 - x0; cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
    stand = rbox(cx - 120, cy - 90, z, cx + 120, cy + 90, z + 18, 8)
    neck = box(cx - 25, cy + 20, z + 15, cx + 25, cy + 60, z + 200)
    screen = rbox(x0, cy + 40, z + 140, x1, cy + 85, z + 140 + w * 0.6, 10)
    bezel = box(x0 + 20, cy + 30, z + 160, x1 - 20, cy + 45, z + 140 + w * 0.6 - 20)
    kb = rbox(cx - 220, y0 - 220, z, cx + 220, y0 - 80, z + 22, 6)
    return U(stand, neck, screen - bezel, kb)

def desk(W, D, mon=None, H=740):
    top = rbox_v(0, 0, H - 30, W, D, H, 10)
    side = [box(0, 20, 0, MIN, D - 20, H - 30), box(W - MIN, 20, 0, W, D - 20, H - 30)]
    modesty = box(MIN, D - 60, 300, W - MIN, D - 20, H - 30)
    pw = min(420, W * 0.33)
    ped = box(W - MIN - pw, 40, 40, W - MIN, D - 60, H - 30)
    for z in (40 + (H - 70) / 3, 40 + 2 * (H - 70) / 3):          # drawer seams
        ped = ped - box(W - MIN - pw - 1, 39, z - 12, W - MIN + 1, 60, z + 12)
    pulls = [box(W - MIN - pw / 2 - 80, 15, z, W - MIN - pw / 2 + 80, 40, z + 30)
             for z in (40 + (H - 70) / 6 + 20, 40 + (H - 70) / 2 + 20, 40 + 5 * (H - 70) / 6 + 20)]
    parts = [top, *side, modesty, ped, *pulls]
    if mon is not None:
        parts.append(monitor(*mon, H))
    return U(*parts)

# ---------------------------------------------------------------- storage
def doors(W, n=None):
    n = n or max(1, round(W / 550))
    return [W * i / n for i in range(1, n)], n

def cabinet(W, D, H, n_doors=None, plinth=100, handle='v', tall=False):
    body = box(0, 0, plinth, W, D, H)
    body = body + box(50, 50, 0, W - 50, D, plinth + 1)                # recessed plinth
    seams, n = doors(W, n_doors)
    for x in seams:
        body = body - box(x - 15, -1, plinth + 30, x + 15, 25, H - 30)
    body = body - box(-1, -1, H - 40, W + 1, 25, H - 25) if tall else body   # crown shadow line
    hs = []
    edges = [0] + seams + [W]
    for i in range(n):
        x0, x1 = edges[i], edges[i + 1]
        hx = x1 - 70 if i % 2 == 0 else x0 + 70
        if handle == 'v':
            zc = min(H * 0.55, 1100) if tall else H - 220
            hs.append(rbox(hx - 20, -30, zc - 220, hx + 20, 5, zc + 220, 10))
        else:
            hs.append(rbox((x0 + x1) / 2 - 140, -30, H - 120, (x0 + x1) / 2 + 140, 5, H - 80, 10))
    return U(body, *hs)

def wardrobe(W, D, H=2400):
    w = cabinet(W, D, H, tall=True)
    # top cupboard split (doors above 2000)
    return w - box(-1, -1, 1990, W + 1, 25, 2010)

def shoe_cabinet(W, D, H=1100):
    c = cabinet(W, D, H, handle='h', plinth=150)
    top = rbox_v(-10, -10, H, W + 10, D, H + 30, 8)
    return c + top

def tv_console(W, D, H=450):
    body = box(0, 0, 120, W, D, H)
    seams, n = doors(W, 3)
    for x in seams:
        body = body - box(x - 12, -1, 140, x + 12, 25, H - 20)
    body = body - box(W / 3 + 40, -1, 160, 2 * W / 3 - 40, D - 60, H - 40)      # open media niche
    legs = [leg(x, y, 0, 125, 50, 0.8) for x in (80, W - 80) for y in (60, D - 60)]
    return U(body, *legs)

# ---------------------------------------------------------------- kitchen
def kitchen(W, D, sink=None, tap=None, hob=None, burners=(), H=880):
    base = box(0, 60, 0, W, D, 100) + box(0, 0, 100, W, D, H - 40)
    seams, n = doors(W, max(1, round(W / 600)))
    for x in seams:
        base = base - box(x - 12, -1, 130, x + 12, 25, H - 70)
    base = base - box(-1, -1, H - 210, W + 1, 25, H - 190)                   # drawer line
    top = box(-20, -20, H - 40, W, D, H)
    if sink:
        x0, y0, x1, y1 = sink
        top = top - rbox(x0, y0, H - 220, x1, y1, H + 40, 60)
        top = top + rbox(x0 + (x1 - x0) / 2 - 30, y0 + (y1 - y0) / 2 - 30, H - 222, x0 + (x1 - x0) / 2 + 30, y0 + (y1 - y0) / 2 + 30, H - 205, 10)
    parts = [base, top]
    for x in seams + [W]:
        x0 = x - (W / n)
        parts.append(rbox(x0 + 60, -45, H - 160, x - 60, -20, H - 130, 8))        # bar handles
    if tap:
        tx, ty = tap
        parts += [cyl(tx, ty, H, 250, 22), box(tx - 20, ty - 190, H + 220, tx + 20, ty + 10, H + 260),
                  cyl(tx, ty - 180, H + 170, 70, 18)]
    if hob:
        x0, y0, x1, y1 = hob
        parts.append(rbox(x0, y0, H - 2, x1, y1, H + 10, 10))
        for bx, by, br in burners:
            parts += [cyl(bx, by, H + 10, 12, br * 0.9), cyl(bx, by, H + 22, 10, br * 0.45)]
    return U(*parts)

def fridge(W, D, H=1850):
    body = rbox_v(0, 0, 40, W, D, H, 25) + box(40, 60, 0, W - 40, D - 40, 41)
    body = body - box(-1, -1, H * 0.62 - 12, W + 1, 25, H * 0.62 + 12)       # freezer / fridge split
    body = body - box(-1, -1, 40, W + 1, 20, 120)                             # kick vent recess
    hs = [rbox(W - 90, -40, H * 0.30, W - 50, 5, H * 0.58, 10), rbox(W - 90, -40, H * 0.66, W - 50, 5, H * 0.85, 10)]
    return U(body, *hs)

def washer(W, D, H=850, dryer=False):
    body = rbox_v(0, 0, 0, W, D, H, 25)
    body = body - box(-1, -1, H - 150, W + 1, 20, H - 140)                    # control panel seam
    cx, cz = W / 2, (H - 150) / 2 + 20
    door = M.cylinder(45, 190, 190, 48).rotate([90, 0, 0]).translate([cx, 20, cz])
    glass = M.cylinder(60, 140, 140, 48).rotate([90, 0, 0]).translate([cx, 0, cz])
    knob = M.cylinder(40, 35, 35, 24).rotate([90, 0, 0]).translate([W - 110, 15, H - 75])
    disp = box(40, -10, H - 120, 260, 5, H - 40)
    return U(body, door - glass, knob, disp)

# ---------------------------------------------------------------- bathroom
def bathtub(W, D, tap=None, H=560):
    shell = rbox_v(0, 0, 0, W, D, H, 30)
    inner = rbox(80, 80, 140, W - 80, D - 80, H + 200, 160)
    rim = rbox_v(0, 0, H - 30, W, D, H, 30) - rbox_v(80, 80, H - 40, W - 80, D - 80, H + 10, 140)
    parts = [shell - inner, rim, cyl(W * 0.5, D / 2, 140, 8, 30)]
    if tap:
        tx, ty = tap
        parts += [cyl(tx, ty, H, 120, 22), box(tx - 160, ty - 20, H + 90, tx + 10, ty + 20, H + 130)]
    return U(*parts)

def toilet(W, D, tank_d=200):
    bd = D - tank_d
    ped = M.batch_hull([ellip(W / 2, bd * 0.55, 0, 1, W * 0.28, bd * 0.32), ellip(W / 2, bd * 0.5 + 10, 330, 1, W * 0.5 - 5, bd * 0.5 - 5)])
    bowl = ped - ellip(W / 2, bd * 0.48, 220, 200, W * 0.36, bd * 0.36)
    seat = ellip(W / 2, bd * 0.5 + 10, 330, 25, W * 0.5 - 5, bd * 0.5 - 5)
    lid = ellip(W / 2, bd * 0.5 + 10, 355, 20, W * 0.5 - 15, bd * 0.5 - 15)
    hinge = box(W / 2 - 100, bd - 30, 330, W / 2 + 100, bd + 30, 375)
    neck = box(W / 2 - W * 0.28, bd - 60, 0, W / 2 + W * 0.28, D - 30, 360)
    tank = rbox(10, D - tank_d, 360, W - 10, D, 780, 30)
    lidt = rbox(0, D - tank_d - 10, 770, W, D, 810, 15)
    btn = cyl(W / 2, D - tank_d / 2, 810, 10, 35)
    return U(bowl, seat, lid, hinge, neck, tank, lidt, btn)

def vanity(W, D, bowls, taps, H=850):
    cab = box(0, 0, 150, W, D, H - 40) + box(40, 40, 0, W - 40, D, 151)
    seams, n = doors(W, max(1, len(bowls)))
    for x in seams:
        cab = cab - box(x - 12, -1, 180, x + 12, 25, H - 70)
    top = rbox_v(-10, -10, H - 40, W, D, H, 10)
    for (x0, y0, x1, y1) in bowls:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        top = top - ellip(cx, cy, H - 160, 300, (x1 - x0) / 2, (y1 - y0) / 2)
        cab = cab - ellip(cx, cy, H - 160, 300, (x1 - x0) / 2, (y1 - y0) / 2)
        top = top + cyl(cx, cy, H - 162, 10, 25)
    parts = [cab, top]
    for tx, ty in taps:
        parts += [cyl(tx, ty, H, 180, 20), box(tx - 18, ty - 130, H + 150, tx + 18, ty + 10, H + 185)]
    hs = [rbox(x - 120, -40, H - 120, x + 120, -10, H - 90, 8) for x in [W * (i + 0.5) / n for i in range(n)]]
    return U(*parts, *hs)

def shower(W, D, H=2100):
    tray = rbox_v(0, 0, 0, W, D, 70, 20) - rbox(40, 40, 45, W - 40, D - 40, 200, 30)
    drain = cyl(W / 2, D / 2, 44, 6, 60)
    for k in range(-2, 3):
        drain = drain - box(W / 2 - 40, D / 2 + k * 20 - 5, 40, W / 2 + 40, D / 2 + k * 20 + 5, 60)
    # shower column on the wall side (y = D), at the centre
    cx = W / 2
    col = cyl(cx, D - 50, 70, H - 70, 22)
    arm = box(cx - 20, D - 300, H - 40, cx + 20, D - 30, H)
    head = cyl(cx, D - 300, H - 60, 25, 130)
    mixer = rbox(cx - 90, D - 90, 1000, cx + 90, D - 30, 1080, 15)
    # fixed glass screen on the front edge (y = 0), half width
    glass = box(0, 0, 70, W * 0.55, MIN, H - 100) + box(0, 0, H - 120, W * 0.55, 60, H - 90)
    return U(tray, drain, col, arm, head, mixer, glass)
