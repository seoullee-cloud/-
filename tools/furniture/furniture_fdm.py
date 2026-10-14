"""FDM-friendly furniture (Bambu P1S, 0.4 mm nozzle, PLA) at 1:50 — real-mm builders, z up, back at +y.
Rules (printed size = real / 50):
  posts >= 70 real (1.4 mm), walls >= 40 real (0.8 mm), grooves 35 wide x 20 deep (0.7 x 0.4 mm)
  no unsupported down-facing surface steeper than 45 deg except short flat bridges (< ~600 real / 12 mm)
  every piece has a flat bottom; tables/desks are printed upside down (see PRINT_FLIP)."""
import math
from manifold3d import Manifold as M
from furniture import box, rbox_v, cyl, ellip, U

POST, WALL, GW, GD = 70.0, 40.0, 35.0, 20.0

def rtop(x0, y0, z0, x1, y1, z1, r):
    """flat bottom, vertical sides rounded, top edges rounded (prints without support)"""
    r = max(1.0, min(r, (x1 - x0) / 2 - 0.01, (y1 - y0) / 2 - 0.01, (z1 - z0) - 0.02))
    base = rbox_v(x0, y0, z0, x1, y1, z1 - r, r)
    s = M.sphere(r, 16)
    tops = [s.translate([x, y, z1 - r]) for x in (x0 + r, x1 - r) for y in (y0 + r, y1 - r)]
    return M.batch_hull([base] + tops)

def chamfer_plinth(x0, y0, x1, y1, h, inset):
    """recessed base: inset at the floor, growing at 45 deg to the full outline at height = inset"""
    inset = min(inset, h)
    a = box(x0 + inset, y0 + inset, 0, x1 - inset, y1 - inset, 1)
    b = box(x0, y0, inset - 1, x1, y1, h)
    return M.batch_hull([a, box(x0, y0, inset - 1, x1, y1, inset)]) + b

def groove_v(x, z0, z1, y=0):
    return box(x - GW / 2, y - 1, z0, x + GW / 2, y + GD, z1)

def groove_h(x0, x1, z, y=0):
    return box(x0, y - 1, z - GW / 2, x1, y + GD, z + GW / 2)

def door_split(W, n=None):
    n = n or max(1, round(W / 550))
    return [W * i / n for i in range(1, n)], n

# ------------------------------------------------------------------ sleeping / seating
def bed(W, D, pillows=2):
    base = chamfer_plinth(0, 0, W, D - 60, 300, 80)
    matt = rtop(25, 25, 300, W - 25, D - 90, 520, 60)
    head = rtop(0, D - 90, 0, W, D, 1000, 30)
    head = head - box(80, D - 91, 450, W - 80, D - 70, 940)                    # framed panel
    head = head + rtop(110, D - 90, 450, W - 110, D - 60, 930, 25)
    pw = (W - 120 - 60 * (pillows - 1)) / pillows
    pil = [rtop(100 + i * (pw + 20), D - 520, 505, 100 + i * (pw + 20) + pw - 40, D - 160, 640, 55) for i in range(pillows)]
    duvet = rtop(10, 10, 300, W - 10, D - 560, 565, 40)                      # drapes to the frame top
    fold = rtop(60, D - 700, 545, W - 60, D - 610, 610, 30)                  # turned-down edge
    return U(base, matt, head, *pil, duvet, fold)

def sofa(W, D, seats=2, arm=180, back=200, seat_h=430, back_h=820):
    base = chamfer_plinth(0, 0, W, D, 300, 70)
    arms = [rtop(0, 0, 0, arm, D, 620, 60), rtop(W - arm, 0, 0, W, D, 620, 60)]
    backp = rtop(arm - 10, D - back, 0, W - arm + 10, D, back_h, 60)
    iw = W - 2 * arm; cw = (iw - GW * (seats - 1)) / seats
    cush, bcush = [], []
    for i in range(seats):
        x0 = arm + i * (cw + GW)
        cush.append(rtop(x0, 20, 300, x0 + cw, D - back, seat_h, 50))
        bcush.append(rtop(x0 + 10, D - back - 120, seat_h, x0 + cw - 10, D - back + 5, back_h - 40, 55))
    return U(base, *arms, backp, *cush, *bcush)

def armchair(W, D, arm=150, back=180):
    return sofa(W, D, seats=1, arm=arm, back=back)

def dining_chair(W, D, seat_h=450, H=860):
    """panel-sided chair: two side frames with a 45-deg pointed opening, seat bridges between them"""
    side_t = POST
    parts = []
    for x in (0, W - side_t):
        p = box(x, 20, 0, x + side_t, D - 20, seat_h)
        # pointed opening: rectangle + 45 deg roof (hull with a ridge line)
        o0, o1 = 90, D - 110
        open_ = M.batch_hull([box(x - 1, o0, -1, x + side_t + 1, o1, seat_h - 200),
                              box(x - 1, (o0 + o1) / 2 - 1, -1, x + side_t + 1, (o0 + o1) / 2 + 1, seat_h - 200 + (o1 - o0) / 2)])
        parts.append(p - open_)
    seat = rtop(0, 20, seat_h - 50, W, D - 20, seat_h, 15)
    backp = box(0, D - 20 - POST, seat_h - 50, W, D - 20, H)
    backp = backp - box(side_t + 40, D - 21 - POST, seat_h + 120, W - side_t - 40, D - 19, H - 140)   # open back (bridge at top)
    rail = rtop(0, D - 20 - POST - 10, H - 140, W, D - 15, H, 15)
    return U(*parts, seat, backp, rail)

def desk_chair(W, D, seat_h=470, H=980):
    cx, cy = W / 2, D / 2
    R = min(W, D) / 2 - 10
    spokes = [box(0, -POST / 2, 0, R, POST / 2, 70).rotate([0, 0, k * 72 + 90]).translate([cx, cy, 0]) for k in range(5)]
    feet = [cyl(cx + (R - 35) * math.cos(math.radians(k * 72 + 90)), cy + (R - 35) * math.sin(math.radians(k * 72 + 90)), 0, 70, 38)
            for k in range(5)]
    hub = cyl(cx, cy, 0, 110, 70, 55)
    col = cyl(cx, cy, 100, seat_h - 250, 42)
    # mechanism as a 45-deg cone that carries the seat
    reach = math.hypot(W / 2 - 20, D / 2 - 20)
    cone = M.batch_hull([cyl(cx, cy, seat_h - 60 - reach, 1, 42), box(20, 20, seat_h - 61, W - 20, D - 40, seat_h - 60)])
    seat = rtop(20, 20, seat_h - 60, W - 20, D - 40, seat_h, 30)
    back = rtop(50, D - 40 - 60, seat_h - 60, W - 50, D - 40, H, 40)
    back = back - box(90, D - 101, seat_h + 60, W - 90, D - 85, H - 60)          # shallow mesh panel recess
    lumbar = rtop(80, D - 40 - 100, seat_h, W - 80, D - 80, seat_h + 220, 25)
    return U(*spokes, *feet, hub, col, cone, seat, back, lumbar)

def lounge_chair(W, D):
    base = chamfer_plinth(20, 20, W - 20, D - 20, 380, 60)
    back = rtop(0, D - 180, 0, W, D, 760, 80)
    arms = [rtop(0, 0, 0, 110, D, 560, 50), rtop(W - 110, 0, 0, W, D, 560, 50)]
    cush = rtop(110, 20, 380, W - 110, D - 170, 450, 45)
    return U(base, back, *arms, cush)

def side_table(W, D, H=520):
    """tulip table: base disc, stem, 45-deg cone under the top"""
    r = min(W, D) / 2; cx, cy = W / 2, D / 2
    return U(cyl(cx, cy, 0, 30, r * 0.62, r * 0.55, 48), cyl(cx, cy, 0, H, 40),
             cyl(cx, cy, H - 40 - (r - 40), r - 40, 40, r, 48), cyl(cx, cy, H - 40, 40, r, r, 48))

# ------------------------------------------------------------------ tables (printed upside down)
def dining_table(W, D, H=740):
    top = rbox_v(0, 0, H - 45, W, D, H, 25)
    apron = box(70, 70, H - 140, W - 70, D - 70, H - 45) - box(70 + WALL, 70 + WALL, H - 141, W - 70 - WALL, D - 70 - WALL, H - 44)
    legs = [M.batch_hull([box(x - 45, y - 45, H - 46, x + 45, y + 45, H - 45),
                          box(x - 30, y - 30, 0, x + 30, y + 30, 1)]) for x in (100, W - 100) for y in (100, D - 100)]
    return U(top, apron, *legs)

def desk(W, D, mon=None, H=740):
    top = rbox_v(0, 0, H - 35, W, D, H, 10)
    side = [box(0, 20, 0, WALL + 10, D - 20, H - 35), box(W - WALL - 10, 20, 0, W, D - 20, H - 35)]
    modesty = box(WALL, D - 20 - WALL, 300, W - WALL, D - 20, H - 35)
    pw = min(420, W * 0.33); x0 = W - WALL - 10 - pw
    ped = box(x0, 30, 0, W - WALL - 10, D - 60, H - 35)
    zs = [(H - 35) / 3, 2 * (H - 35) / 3]
    for z in zs:
        ped = ped - groove_h(x0 - 1, W - WALL - 9, z, y=30)
    pulls = [box(x0 + pw / 2 - 90, 5, z, x0 + pw / 2 + 90, 31, z + 35) for z in ((H - 35) / 6 + 30, (H - 35) / 2 + 30, 5 * (H - 35) / 6 + 30)]
    return U(top, *side, modesty, ped, *pulls)

def monitor(Wm):
    """separate part: stand + screen resting on the stand (no overhang)"""
    w = max(Wm, 450)
    stand = rtop(w / 2 - 130, 0, 0, w / 2 + 130, 200, 30, 10)
    screen = rtop(0, 80, 30, w, 80 + 60, 30 + w * 0.58, 12)
    back = M.batch_hull([box(w / 2 - 80, 140, 30, w / 2 + 80, 141, 260), box(w / 2 - 50, 140, 30, w / 2 + 50, 190, 120)])
    return U(stand, screen, back)

# ------------------------------------------------------------------ storage
def cabinet(W, D, H, n_doors=None, plinth=100, handle='v', tall=False):
    body = chamfer_plinth(0, 0, W, D, H, min(plinth, 60))
    seams, n = door_split(W, n_doors)
    for x in seams:
        body = body - groove_v(x, plinth, H - 40)
    body = body - groove_h(-1, W + 1, plinth)                                   # plinth line
    if tall:
        body = body - groove_h(-1, W + 1, H - 400)                               # top cupboard line
    edges = [0] + seams + [W]
    hs = []
    for i in range(n):
        a, b = edges[i], edges[i + 1]
        hx = b - 80 if i % 2 == 0 else a + 80
        if handle == 'v':
            zc = min(H * 0.5, 1050) if tall else H - 250
            hs.append(box(hx - POST / 2, -30, zc - 250, hx + POST / 2, 1, zc + 250))
        else:
            hs.append(box((a + b) / 2 - 150, -30, H - 140, (a + b) / 2 + 150, 1, H - 90))
    return U(body, *hs)

def wardrobe(W, D, H=2400):
    return cabinet(W, D, H, tall=True)

def shoe_cabinet(W, D, H=1100):
    return cabinet(W, D, H, handle='h', plinth=120) + box(-15, -15, H - 35, W + 15, D, H)

def tv_console(W, D, H=450):
    body = chamfer_plinth(0, 0, W, D, H, 80)
    seams, n = door_split(W, 3)
    for x in seams:
        body = body - groove_v(x, 100, H - 30)
    body = body - box(W / 3 + 50, -1, 140, 2 * W / 3 - 50, D - 80, H - 60)     # media niche (bridge ~11 mm)
    return body

# ------------------------------------------------------------------ kitchen
def spout(tx, ty, z, rise, reach):
    """faucet leaning towards the basin (-y); front face <= 45 deg so it prints without support"""
    reach = min(reach, rise - POST)
    base = cyl(tx, ty, z, 30, POST * 0.8)
    body = M.batch_hull([box(tx - POST / 2, ty - POST / 2, z, tx + POST / 2, ty + POST / 2, z + 1),
                         box(tx - POST / 2, ty - reach - POST / 2, z + rise - POST, tx + POST / 2, ty + POST / 2, z + rise)])
    return base + body

def kitchen(W, D, sink=None, tap=None, hob=None, burners=(), H=880):
    base = chamfer_plinth(0, 0, W, D, H - 40, 60)
    seams, n = door_split(W, max(1, round(W / 600)))
    for x in seams:
        base = base - groove_v(x, 120, H - 60)
    base = base - groove_h(-1, W + 1, H - 220)
    base = base - groove_h(-1, W + 1, 100)
    top = box(-20, -20, H - 40, W, D, H)
    parts = []
    if sink:
        x0, y0, x1, y1 = sink
        top = top - rbox_v(x0, y0, H - 200, x1, y1, H + 1, 50)
        base = base - rbox_v(x0, y0, H - 200, x1, y1, H + 1, 50)
    edges = [0] + seams + [W]
    for i in range(n):
        a, b = edges[i], edges[i + 1]
        parts.append(box(a + 70, -30, H - 175, b - 70, 1, H - 135))
    if tap:
        parts.append(spout(tap[0], tap[1], H, 230, 150))
    if hob:
        x0, y0, x1, y1 = hob
        parts.append(box(x0, y0, H - 1, x1, y1, H + 15))
        for bx, by, br in burners:
            parts.append(cyl(bx, by, H + 15, 20, br * 0.85) - cyl(bx, by, H + 25, 20, br * 0.55))
            parts.append(cyl(bx, by, H + 15, 25, br * 0.3))
    return U(base, top, *parts)

def fridge(W, D, H=1850):
    body = chamfer_plinth(0, 0, W, D, H, 50)
    body = body - groove_h(-1, W + 1, H * 0.62)
    hs = [box(W - 110, -30, H * 0.32, W - 110 + POST, 1, H * 0.58), box(W - 110, -30, H * 0.66, W - 110 + POST, 1, H * 0.86)]
    return U(body, *hs)

def washer(W, D, H=850):
    body = rbox_v(0, 0, 0, W, D, H, 25)
    body = body - groove_h(-1, W + 1, H - 150)
    cx, cz = W / 2, (H - 150) / 2 + 20
    ring = M.cylinder(40, 200, 200, 48).rotate([90, 0, 0]).translate([cx, 20, cz]) - \
           M.cylinder(60, 140, 140, 48).rotate([90, 0, 0]).translate([cx, 30, cz])
    glass = M.cylinder(30, 141, 141, 48).rotate([90, 0, 0]).translate([cx, 29, cz])  # recessed glass
    body = body - glass
    knob = M.cylinder(30, 40, 40, 24).rotate([90, 0, 0]).translate([W - 110, 10, H - 75])
    disp = box(40, -20, H - 115, 260, 1, H - 45)
    return U(body, ring, knob, disp)

# ------------------------------------------------------------------ bathroom
def bathtub(W, D, tap=None, H=560):
    shell = rbox_v(0, 0, 0, W, D, H, 30)
    inner = rbox_v(90, 90, 160, W - 90, D - 90, H + 1, 150)
    rim = shell - inner
    rim = rim - groove_h(-1, W + 1, 110) - groove_h(-1, W + 1, H - 80)
    parts = [rim, cyl(W * 0.12 if tap is None else tap[0], D / 2, 160, 10, 40)]
    if tap:
        parts.append(spout(tap[0], tap[1], H, 140, 120))
    return U(*parts)

def toilet(W, D, tank_d=200):
    bd = D - tank_d
    ped = M.batch_hull([ellip(W / 2, bd * 0.55, 0, 1, W * 0.30, bd * 0.34), ellip(W / 2, bd * 0.5 + 10, 330, 1, W * 0.5 - 5, bd * 0.5 - 5)])
    bowl = ped - ellip(W / 2, bd * 0.48, 230, 200, W * 0.36, bd * 0.36)
    seat = ellip(W / 2, bd * 0.5 + 10, 330, 30, W * 0.5 - 5, bd * 0.5 - 5) - ellip(W / 2, bd * 0.48, 329, 40, W * 0.30, bd * 0.30)
    lid_open = box(15, bd - 40, 360, W - 15, bd + 20, 760)                       # lid raised against the tank
    tank = rtop(10, D - tank_d, 0, W - 10, D, 780, 25)                           # tank down to the floor = solid back
    tank = tank - groove_h(9, W - 9, 400, y=D - tank_d)
    lidt = rtop(0, D - tank_d - 15, 760, W, D, 810, 15)
    btn = cyl(W / 2, D - tank_d / 2, 805, 20, 40)
    return U(bowl, seat, lid_open, tank, lidt, btn)

def vanity(W, D, bowls, taps, H=850):
    cab = chamfer_plinth(0, 0, W, D, H - 40, 70)
    seams, n = door_split(W, max(1, len(bowls)))
    for x in seams:
        cab = cab - groove_v(x, 150, H - 60)
    cab = cab - groove_h(-1, W + 1, 150)
    top = rbox_v(-10, -10, H - 40, W, D, H, 10)
    for (x0, y0, x1, y1) in bowls:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cut = ellip(cx, cy, H - 160, 200, (x1 - x0) / 2, (y1 - y0) / 2)
        top, cab = top - cut, cab - cut
    parts = [cab, top]
    for tx, ty in taps:
        parts.append(spout(tx, ty, H, 190, 110))
    for x in [W * (i + 0.5) / n for i in range(n)]:
        parts.append(box(x - 130, -30, H - 135, x + 130, 1, H - 95))
    return U(*parts)

def shower(W, D, H=2100):
    tray = rbox_v(0, 0, 0, W, D, 80, 20) - rbox_v(50, 50, 50, W - 50, D - 50, 81, 30)
    drain = cyl(W / 2, D / 2, 49, 6, 70) - cyl(W / 2, D / 2, 52, 6, 40)
    cx = W / 2
    plate = box(cx - 160, D - WALL, 80, cx + 160, D, H)                            # wall rail plate (stabilises the column)
    col = box(cx - POST / 2, D - WALL - POST, 80, cx + POST / 2, D - WALL, H - 60)
    head = M.batch_hull([box(cx - POST / 2, D - WALL - POST, H - 60 - 420, cx + POST / 2, D - WALL, H),
                         cyl(cx, D - 300, H - 40, 40, 130, 130)])                  # 45-deg arm + head
    mixer = rtop(cx - 110, D - WALL - 80, 1000, cx + 110, D - WALL, 1090, 15)
    glass = box(0, 0, 80, W * 0.55, 60, H - 100)
    return U(tray, drain, plate, col, head, mixer, glass)
