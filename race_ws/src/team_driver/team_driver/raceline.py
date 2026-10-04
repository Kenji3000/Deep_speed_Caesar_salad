#!/usr/bin/env python3
"""Racing line generation, computed from the map every time the driver starts.

Pipeline (each step is a function below):
  1. load_map          read the occupancy grid the simulator uses
  2. seal_cone_rows    join neighbouring cones into solid walls (rule 11)
  3. trace_lap         shortest closed lap through open track (Dijkstra),
                       crossing the finish line in the racing direction
  4. smooth/resample   evenly spaced, smooth reference line
  5. corridor_bounds   how far the car may move left/right of the reference
                       at each point while keeping `margin` from every wall
  6. min_curvature     choose those sideways offsets to make the line as
                       straight as possible (a bounded least-squares problem)
  7. speed_profile     fastest speed at each point given grip, braking and
                       acceleration limits

Only numpy, scipy and yaml are needed; all three are in the container.
"""

import math
import os

import numpy as np
import yaml
from scipy import ndimage, sparse
from scipy.optimize import lsq_linear
from scipy.sparse.csgraph import dijkstra


# ---------------------------------------------------------------------------
# 1. Map
# ---------------------------------------------------------------------------
class GridMap:
    """Occupancy grid in numpy form. Row 0 is the TOP of the image, as in the
    file; world y grows upwards, so rows are flipped in to_cell/to_world."""

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        image = meta['image']
        if not os.path.isabs(image):
            image = os.path.join(os.path.dirname(yaml_path), image)
        pixels = read_pgm(image)
        self.res = float(meta['resolution'])
        self.ox, self.oy = float(meta['origin'][0]), float(meta['origin'][1])
        negate = bool(int(meta.get('negate', 0)))
        occ = pixels / 255.0 if negate else (255.0 - pixels) / 255.0
        self.occupied = occ > float(meta.get('occupied_thresh', 0.65))
        self.h, self.w = self.occupied.shape
        self.clearance = None

    def to_cell(self, x, y):
        col = np.floor((np.asarray(x) - self.ox) / self.res).astype(int)
        row = self.h - 1 - np.floor((np.asarray(y) - self.oy) / self.res).astype(int)
        return row, col

    def to_world(self, row, col):
        return (self.ox + (np.asarray(col) + 0.5) * self.res,
                self.oy + (self.h - 1 - np.asarray(row) + 0.5) * self.res)

    def update_clearance(self):
        """Distance in metres from every cell to the nearest obstacle."""
        self.clearance = ndimage.distance_transform_edt(~self.occupied) * self.res

    def clearance_at(self, x, y):
        row, col = self.to_cell(x, y)
        inside = (row >= 0) & (row < self.h) & (col >= 0) & (col < self.w)
        out = np.zeros(np.shape(row))
        out[inside] = self.clearance[row[inside], col[inside]]
        return out


def read_pgm(path):
    """Minimal binary (P5) PGM reader."""
    with open(path, 'rb') as f:
        data = f.read()
    tokens, pos = [], 0
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            pos = data.index(b'\n', pos) + 1
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        tokens.append(data[pos:end])
        pos = end
    if tokens[0] != b'P5':
        raise ValueError(f'{path}: only binary PGM (P5) is supported')
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    pos += 1
    dtype = np.uint8 if maxval < 256 else '>u2'
    img = np.frombuffer(data, dtype=dtype, count=w * h, offset=pos).reshape(h, w)
    return img.astype(np.float64) * (255.0 / maxval)


# ---------------------------------------------------------------------------
# 2. Cone rows are walls
# ---------------------------------------------------------------------------
def seal_cone_rows(grid, max_span=0.9, link_distance=1.1):
    """Find small free-standing obstacles (cones) and draw a wall between any
    two that are closer than link_distance. Returns the number of links."""
    labels, n = ndimage.label(grid.occupied, structure=np.ones((3, 3)))
    centres = []
    for sl in ndimage.find_objects(labels):
        span = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * grid.res
        if 0.1 < span < max_span:
            r = (sl[0].start + sl[0].stop - 1) / 2.0
            c = (sl[1].start + sl[1].stop - 1) / 2.0
            centres.append(grid.to_world(r, c))
    links = 0
    for i, a in enumerate(centres):
        for b in centres[i + 1:]:
            d = math.dist(a, b)
            if d <= link_distance:
                t = np.linspace(0.0, 1.0, max(2, int(d / (grid.res / 2)) + 1))
                rows, cols = grid.to_cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr = np.clip(rows + dr, 0, grid.h - 1)
                        cc = np.clip(cols + dc, 0, grid.w - 1)
                        grid.occupied[rr, cc] = True
                links += 1
    return links


# ---------------------------------------------------------------------------
# 3. One lap through open track
# ---------------------------------------------------------------------------
def signed_side(a, b, x, y):
    return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])


def trace_lap(grid, finish_line, start_pose, half_width=0.35):
    """Cheapest closed lap (Dijkstra over the grid) that leaves the finish line
    in the racing direction and comes back to it. Cells near walls cost more,
    so the path runs down the middle of the corridor."""
    a, b = finish_line
    # Racing direction: the start pose is behind the line, facing across it.
    sx, sy, sth = start_pose
    ahead = (sx + math.cos(sth), sy + math.sin(sth))
    direction = np.sign(signed_side(a, b, *ahead) - signed_side(a, b, sx, sy))

    clear = grid.clearance
    drivable = clear >= half_width
    # Cut the finish line out of the grid so the search has to go round.
    t = np.linspace(0, 1, int(math.dist(a, b) / (grid.res / 3)) + 2)
    rows, cols = grid.to_cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    cut = np.zeros_like(drivable)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            cut[np.clip(rows + dr, 0, grid.h - 1), np.clip(cols + dc, 0, grid.w - 1)] = True
    usable = drivable & ~cut

    H, W = grid.h, grid.w
    idx = np.arange(H * W).reshape(H, W)
    src, dst, wgt = [], [], []
    for dr, dc, step in ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                         (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)):
        r0, r1 = max(0, -dr), H - max(0, dr)
        c0, c1 = max(0, -dc), W - max(0, dc)
        ok = usable[r0:r1, c0:c1] & usable[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        i = idx[r0:r1, c0:c1][ok]
        j = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc][ok]
        cj = clear[r0 + dr:r1 + dr, c0 + dc:c1 + dc][ok]
        src.append(i)
        dst.append(j)
        wgt.append(step * grid.res * (1.0 + 1.2 / np.maximum(cj, 0.05)))
    graph = sparse.csr_matrix((np.concatenate(wgt), (np.concatenate(src), np.concatenate(dst))),
                              shape=(H * W, H * W))

    # Cells touching the cut: in front of it (starts) and behind it (goals).
    near = ndimage.binary_dilation(cut, np.ones((3, 3))) & usable
    nr, nc = np.nonzero(near)
    wx, wy = grid.to_world(nr, nc)
    side = signed_side(a, b, wx, wy) * direction
    starts = idx[nr[side > 0], nc[side > 0]]
    goals = idx[nr[side < 0], nc[side < 0]]
    if len(starts) == 0 or len(goals) == 0:
        raise RuntimeError('could not find cells either side of the finish line')

    dist, pred, _ = dijkstra(graph, indices=starts, min_only=True, return_predecessors=True)
    goal = goals[np.argmin(dist[goals])]
    if not np.isfinite(dist[goal]):
        raise RuntimeError('no closed lap found through the map')
    path = []
    node = goal
    while node >= 0:
        path.append(node)
        node = pred[node]
    path.reverse()
    r, c = np.divmod(np.array(path), W)
    x, y = grid.to_world(r, c)
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# 4. Smooth and resample a closed loop
# ---------------------------------------------------------------------------
def smooth_closed(xy, window):
    k = np.ones(window) / window
    pad = window
    ext = np.vstack([xy[-pad:], xy, xy[:pad]])
    out = np.column_stack([np.convolve(ext[:, 0], k, 'same'), np.convolve(ext[:, 1], k, 'same')])
    return out[pad:-pad]


def resample_closed(xy, spacing):
    loop = np.vstack([xy, xy[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(8, int(round(s[-1] / spacing)))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, loop[:, 0]), np.interp(t, s, loop[:, 1])])


def normals_of(xy):
    d = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.column_stack([-d[:, 1], d[:, 0]])      # left-pointing


# ---------------------------------------------------------------------------
# 5. How far left and right the line may move
# ---------------------------------------------------------------------------
def corridor_bounds(grid, xy, normals, margin, max_offset=1.5, step=None):
    """For each point, march along the normal (both ways) while the clearance
    stays above `margin`. Returns (left, right) as positive distances."""
    step = step or grid.res / 2
    offsets = np.arange(0.0, max_offset + step, step)
    bounds = []
    for sign in (1.0, -1.0):
        px = xy[:, 0:1] + sign * normals[:, 0:1] * offsets
        py = xy[:, 1:2] + sign * normals[:, 1:2] * offsets
        ok = grid.clearance_at(px, py) >= margin
        first_bad = np.where(ok.all(1), len(offsets), np.argmin(ok, axis=1))
        bounds.append(offsets[np.maximum(first_bad - 1, 0)])
    return bounds[0], bounds[1]


# ---------------------------------------------------------------------------
# 6. Minimum-curvature line
# ---------------------------------------------------------------------------
def min_curvature(grid, ref, margin, iterations=3):
    """Move each point sideways by a_i (within the corridor) so that the sum of
    squared second differences - a discrete measure of curvature - is as small
    as possible. That is a bounded linear least-squares problem:
        minimise || D2 (ref + a * n) ||^2   subject to  -right <= a <= left
    Re-linearised a few times, because moving points changes their spacing."""
    line = ref.copy()
    n_pts = len(line)
    eye = sparse.identity(n_pts, format='csr')
    D2 = (sparse.eye(n_pts, k=-1) - 2 * eye + sparse.eye(n_pts, k=1)).tolil()
    D2[0, n_pts - 1] = 1.0
    D2[n_pts - 1, 0] = 1.0
    D2 = D2.tocsr()
    for _ in range(iterations):
        nrm = normals_of(line)
        left, right = corridor_bounds(grid, line, nrm, margin)
        A = sparse.vstack([D2 @ sparse.diags(nrm[:, 0]), D2 @ sparse.diags(nrm[:, 1])])
        b = -np.concatenate([D2 @ line[:, 0], D2 @ line[:, 1]])
        lo, hi = -right, left
        hi = np.maximum(hi, lo + 1e-6)
        res = lsq_linear(A, b, bounds=(lo, hi), method='trf', lsmr_tol='auto', max_iter=500)
        line = line + res.x[:, None] * nrm
        line = resample_closed(line, np.mean(np.linalg.norm(np.diff(line, axis=0), axis=1)))
    return line




# ---------------------------------------------------------------------------
# Car limits, measured from the judged simulator (f1tenth_gym single-track
# model with the f1tenth parameters). Its tyres are linear, so sideways grip
# is effectively unlimited; what limits a corner is STEERING. Holding full
# lock at speed v, the car follows a curve of at most
#     k_max(v) = K0 / (1 + KC * v^2)      [1/m]
# (fitted to the simulator: 1.23 at 2 m/s, 0.90 at 7 m/s, 0.69 at 10 m/s).
# Acceleration is capped at A_MAX, and above V_SWITCH at A_MAX * V_SWITCH / v.
# ---------------------------------------------------------------------------
K0, KC = 1.269, 0.0084
A_MAX, V_SWITCH = 9.51, 7.319
LR = 0.17145          # rear axle to centre of gravity [m]


def speed_limit_sq(k, v_max, a_lat, steer_use):
    """Highest v^2 at each point from: top speed, a lateral-acceleration cap,
    and using at most `steer_use` (0..1) of the steering available."""
    k = np.maximum(k, 1e-6)
    steer = (steer_use * K0 / k - 1.0) / KC
    return np.maximum(np.minimum.reduce([np.full_like(k, v_max ** 2), a_lat / k, steer]), 0.25)

# ---------------------------------------------------------------------------
# 6b. Minimum-TIME line (what a racing driver actually does)
# ---------------------------------------------------------------------------
def fast_speed_squared(xy, v_max, a_lat, a_brake, a_accel, steer_use=0.8):
    """Vectorised speed profile, returned as v^2. Same physics as
    speed_profile() but written as prefix-minimum scans so it takes
    microseconds, which lets the optimiser call it thousands of times.
    Backward pass:  v_i^2 = min_j>=i ( vlim_j^2 + 2 a_brake (s_j - s_i) )
    Forward pass:   v_i^2 = min_j<=i ( v_j^2 + 2 a_accel (s_i - s_j) )
    The arrays are doubled so the limits wrap across the finish line."""
    n = len(xy)
    k = curvature_of(xy)
    ds = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)
    lim = speed_limit_sq(k, v_max, a_lat, steer_use)
    a_accel = min(a_accel, A_MAX * V_SWITCH / max(math.sqrt(v_max ** 2), V_SWITCH))
    s2 = np.concatenate([[0.0], np.cumsum(np.tile(ds, 2))])[:-1]
    l2 = np.tile(lim, 2)
    back = np.minimum.accumulate((l2 + 2 * a_brake * s2)[::-1])[::-1] - 2 * a_brake * s2
    back = np.tile(back[:n], 2)
    fwd = 2 * a_accel * s2 + np.minimum.accumulate(back - 2 * a_accel * s2)
    return fwd[n:], ds


def lap_time(xy, v_max, a_lat, a_brake, a_accel, steer_use=0.8):
    w, ds = fast_speed_squared(xy, v_max, a_lat, a_brake, a_accel, steer_use)
    v = np.sqrt(np.maximum(w, 0.01))
    return float(np.sum(2.0 * ds / (v + np.roll(v, -1))))


def periodic_spline_basis(n, every):
    """n x m matrix B: offsets at all n points = B @ offsets at m control
    points, joined by a smooth periodic cubic spline."""
    from scipy.interpolate import CubicSpline
    ctrl = np.arange(0, n, every)
    m = len(ctrl)
    t = np.arange(n)
    B = np.zeros((n, m))
    for j in range(m):
        e = np.zeros(m + 1)
        e[j] = 1.0
        if j == 0:
            e[m] = 1.0
        B[:, j] = CubicSpline(np.append(ctrl, n), e, bc_type='periodic')(t)
    return B, ctrl


def min_time(grid, ref, margin, v_max, a_lat, a_brake, a_accel, steer_use=0.8,
             every=4, time_budget=25.0, log=print):
    """Choose sideways offsets (smooth spline, inside the corridor) that make
    the predicted lap time as small as possible. Starts from the
    minimum-curvature answer and lets L-BFGS-B trade curvature against where
    in the lap it matters: late apexes, braking deeper, using all the exit
    width to get on the power early."""
    import time as _time
    from scipy.optimize import minimize
    centre = ref
    n = len(centre)
    nrm = normals_of(centre)
    left, right = corridor_bounds(grid, centre, nrm, margin)
    lo, hi = -right, np.maximum(left, -right + 1e-3)
    B, ctrl = periodic_spline_basis(n, every)
    m = len(ctrl)
    # Control-point bounds: the tightest corridor between neighbouring controls.
    lo_c, hi_c = np.empty(m), np.empty(m)
    for j in range(m):
        span = np.arange(ctrl[j] - every // 2, ctrl[j] + every // 2 + 1) % n
        lo_c[j], hi_c[j] = lo[span].max(), hi[span].min()
    hi_c = np.maximum(hi_c, lo_c + 1e-3)

    # Start from the reference itself (the minimum-curvature line): u = 0.
    u0 = np.zeros(m)

    def line_of(u):
        return centre + np.clip(B @ u, lo, hi)[:, None] * nrm

    def cost(u):
        a = B @ u
        over = np.maximum(a - hi, 0) + np.maximum(lo - a, 0)
        return lap_time(centre + a[:, None] * nrm, v_max, a_lat, a_brake, a_accel, steer_use) \
            + 200.0 * float(np.sum(over ** 2))

    t0 = _time.time()
    best = {'u': u0, 'f': cost(u0)}
    start_f = best['f']

    class Budget(Exception):
        pass

    def tracked(u):
        f = cost(u)
        if f < best['f']:
            best['u'], best['f'] = u.copy(), f
        if _time.time() - t0 > time_budget:
            raise Budget
        return f

    try:
        minimize(tracked, u0, method='L-BFGS-B', bounds=list(zip(lo_c, hi_c)),
                 options={'maxiter': 400, 'eps': 1e-4})
    except Budget:
        log(f'min-time: stopped at the {time_budget:.0f} s time budget')
    log(f'min-time: predicted lap {start_f:.2f} s (min curvature) -> {best["f"]:.2f} s '
        f'in {_time.time() - t0:.1f} s')
    return line_of(best['u'])


# ---------------------------------------------------------------------------
# 7. Speed profile
# ---------------------------------------------------------------------------
def curvature_of(xy, smooth=True):
    prev, nxt = np.roll(xy, 1, 0), np.roll(xy, -1, 0)
    a = np.linalg.norm(xy - prev, axis=1)
    b = np.linalg.norm(nxt - xy, axis=1)
    c = np.linalg.norm(nxt - prev, axis=1)
    d1, d2 = xy - prev, nxt - xy
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    k = 2.0 * np.abs(cross) / np.maximum(a * b * c, 1e-9)
    if smooth:
        k = np.convolve(np.r_[k[-3:], k, k[:3]], np.ones(7) / 7, 'valid')
    return k


def speed_profile(xy, v_max, v_min, a_lat, a_brake, a_accel, steer_use=0.8, trail=1.0):
    """Target speed at every point of the line.
      1. corner limit: the speed at which the car can still follow this
         curvature using `steer_use` of its steering (and a_lat, if lower);
      2. backward pass: brake so every corner is reached slowly enough. Hard
         braking while turning unloads the rear wheels and the tail slides, so
         the braking allowed shrinks as the corner uses up its limit:
             brake = a_brake * sqrt(1 - (trail * v^2 / v_corner^2)^2)
         i.e. brake hard in a straight line, ease off while turning in.
         trail = 0 switches this off (full braking right up to the apex).
      3. forward pass: accelerate at most a_accel, and less above V_SWITCH,
         as the simulator's motor does.
    Both passes run twice so the limits carry across the finish line."""
    n = len(xy)
    k = curvature_of(xy)
    ds = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)
    vc = np.sqrt(speed_limit_sq(k, 1e3, a_lat, steer_use))   # corner limit, uncapped
    v = np.minimum(vc, v_max)
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            used = min(1.0, trail * (v[j] / vc[i]) ** 2)
            brake = a_brake * max(0.12, math.sqrt(1.0 - used * used))
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2.0 * brake * ds[i]))
        for i in range(n):
            j = (i + 1) % n
            acc = min(a_accel, A_MAX * V_SWITCH / max(v[i], V_SWITCH))
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2.0 * acc * ds[i]))
    return np.maximum(v, v_min)


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------
def load_track(tracks_yaml, name=None):
    with open(tracks_yaml) as f:
        cfg = yaml.safe_load(f)
    name = name or cfg.get('default')
    t = cfg['tracks'][name]
    map_yaml = t['map_path'] + '.yaml'
    if not os.path.isfile(map_yaml):     # fall back to the file next to tracks.yaml
        map_yaml = os.path.join(os.path.dirname(tracks_yaml), os.path.basename(map_yaml))
    return map_yaml, [tuple(p) for p in t['finish_line']], tuple(t['start_pose'])


def load_grid(tracks_yaml):
    """The map with cone rows sealed and the wall-distance field computed."""
    map_yaml, finish, start = load_track(tracks_yaml)
    grid = GridMap(map_yaml)
    grid.cone_links = seal_cone_rows(grid)
    grid.update_clearance()
    return grid, finish, start


def build_raceline(tracks_yaml, margin=0.35, spacing=0.2, blend=1.0, log=print,
                   optimise_time=True, car=None, time_budget=25.0, grid=None):
    """Returns (centreline, raceline) as N x 2 arrays in the racing direction.
    blend: 0 = centreline, 1 = full minimum-curvature line."""
    if grid is None:
        grid, finish, start = load_grid(tracks_yaml)
    else:
        _, finish, start = load_track(tracks_yaml)
    links = grid.cone_links
    lap = trace_lap(grid, finish, start)
    centre = resample_closed(smooth_closed(lap, 9), spacing)
    race = min_curvature(grid, centre, margin)
    if optimise_time:
        car = car or dict(v_max=10.0, a_lat=30.0, a_brake=8.0, a_accel=8.0, steer_use=0.8)
        race = resample_closed(min_time(grid, race, margin, time_budget=time_budget, log=log, **car),
                               spacing)
    if blend < 1.0:
        idx = np.array([np.argmin(np.hypot(*(race - p).T)) for p in centre])
        race = resample_closed(centre + blend * (race[idx] - centre), spacing)
    # Start the list at the point nearest the start pose, so index 0 is the grid.
    i0 = int(np.argmin(np.hypot(race[:, 0] - start[0], race[:, 1] - start[1])))
    race = np.roll(race, -i0, axis=0)
    clear = grid.clearance_at(race[:, 0], race[:, 1])
    log(f'raceline: sealed {links} cone gaps, centreline {lap_length(centre):.1f} m, '
        f'raceline {lap_length(race):.1f} m, tightest clearance {clear.min():.2f} m')
    return centre, race


def load_csv(path):
    """x, y from the first two columns of a CSV ('#' lines are comments)."""
    return np.loadtxt(path, delimiter=',', comments='#')[:, :2]


def lap_length(xy):
    return float(np.sum(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)))


if __name__ == '__main__':
    # Stand-alone: python3 raceline.py [tracks.yaml] [out.csv]  - writes the line
    # to a CSV and a picture, so you can look at it without running ROS.
    import sys
    tracks = sys.argv[1] if len(sys.argv) > 1 else '/hackathon/maps/tracks.yaml'
    centre, race = build_raceline(tracks)
    v = speed_profile(race, 9.0, 1.5, 5.0, 8.5, 9.5, 1.0)
    out = sys.argv[2] if len(sys.argv) > 2 else 'raceline.csv'
    np.savetxt(out, np.column_stack([race, v]), delimiter=',', fmt='%.4f',
               header='x_m, y_m, v_mps  (generated by raceline.py)')
    print('wrote', out)