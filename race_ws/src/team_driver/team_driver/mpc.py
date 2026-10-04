#!/usr/bin/env python3
"""Model Predictive Control for steering (sampling-based MPC, "MPPI").

Every control cycle:
  1. PREDICT  - take K candidate steering plans for the next `horizon` seconds
                and simulate all of them at once with the simulator's own
                single-track equations (same car parameters, same steering
                and speed controllers, same limits).
  2. SCORE    - each predicted path is charged for distance from the racing
                line, for getting close to a wall, for jerky steering, and for
                sliding too far sideways (a drift that could become a spin).
  3. BLEND    - the plans are averaged, weighted by exp(-cost / temperature),
                so the good ones dominate. That average is the new plan.
  4. ACT      - send the first steering command of the plan, then repeat next
                cycle starting from the plan just found (warm start).

Why this beats Pure Pursuit here: the prediction knows that the wheels need
time to turn (3.2 rad/s), that the car understeers more the faster it goes,
and that it slides - so it starts turning before the corner instead of in it.

Speed matters: the simulator does not wait, so time spent computing is delay.
The prediction loop is compiled with Numba when it is available (it ships with
the simulator) and checked against a numpy version at start-up; if Numba is
missing or disagrees, the numpy version is used instead.
"""

import math

import numpy as np

# --- the simulated car (f1tenth parameters of the f1tenth_gym ST model) ------
MU, C_SF, C_SR = 1.0489, 4.718, 5.4562
LF, LR, H_CG, MASS, INERTIA = 0.15875, 0.17145, 0.074, 3.74, 0.04712
WHEELBASE = LF + LR
S_MAX, SV_MAX = 0.4189, 3.2
A_MAX, V_SWITCH, V_MIN_REV, V_TOP = 9.51, 7.319, -5.0, 20.0
G = 9.81
STEER_KP = 10.0 * SV_MAX / (2.0 * S_MAX)      # the simulator's steering P gain
KP_ACCEL = 10.0 * A_MAX / V_TOP               # speed P gain when speeding up
KP_BRAKE = 10.0 * A_MAX / (-V_MIN_REV)        # speed P gain when braking
KC = 0.0084                                   # understeer: k_max(v) ~ 1/(1+KC v^2)


# ---------------------------------------------------------------------------
# The prediction loop, written with plain loops so Numba can compile it to
# machine code (about 10x faster than the numpy version below, which matters
# because the simulator does not wait for us: computing time is delay).
# It must produce the same costs as MPC._rollout_numpy; MPC checks that at
# start-up and uses the numpy version if Numba is missing or disagrees.
# ---------------------------------------------------------------------------
def _rollout_loops(knots, B, x, y, delta, v, yaw, yaw_rate, beta, idx,
                   lx, ly, tx, ty, theta, kappa, speed_ref, clear, res, ox, oy,
                   dt, ds, preview, w_line, w_wall, w_rate, w_end, w_slip,
                   slip_ok, wall_soft, wall_hard):
    K, M = knots.shape
    H = B.shape[0]
    n = lx.shape[0]
    gh, gw = clear.shape
    cost = np.zeros(K)
    c_yaw = MU * MASS / (INERTIA * WHEELBASE)
    for k in range(K):
        X = x
        Y = y
        D = delta
        V = v if v > 0.0 else 0.0
        PSI = yaw
        R = yaw_rate
        BETA = beta
        S = 0.0
        prev_cmd = D
        ii = idx
        total = 0.0
        for h in range(H):
            off = 0.0
            for m in range(M):
                off += knots[k, m] * B[h, m]
            cmd = math.atan(WHEELBASE * kappa[ii]) * (1.0 + KC * V * V) + off
            if cmd > S_MAX:
                cmd = S_MAX
            elif cmd < -S_MAX:
                cmd = -S_MAX
            sv = STEER_KP * (cmd - D)
            if sv > SV_MAX:
                sv = SV_MAX
            elif sv < -SV_MAX:
                sv = -SV_MAX
            dv = speed_ref[(ii + preview) % n] - V
            acc = KP_ACCEL * dv if dv > 0.0 else KP_BRAKE * dv
            upper = A_MAX * V_SWITCH / max(V, 1e-3) if V > V_SWITCH else A_MAX
            if acc > upper:
                acc = upper
            elif acc < -A_MAX:
                acc = -A_MAX
            glr = G * LR - acc * H_CG
            glf = G * LF + acc * H_CG
            Vs = V if V > 1.5 else 1.5
            r_dot = c_yaw * (
                LF * C_SF * glr * D + (LR * C_SR * glf - LF * C_SF * glr) * BETA
                - (LF * LF * C_SF * glr + LR * LR * C_SR * glf) * (R / Vs))
            b_dot = (MU / (Vs * WHEELBASE)) * (
                C_SF * glr * D - (C_SR * glf + C_SF * glr) * BETA
                + (C_SR * glf * LR - C_SF * glr * LF) * (R / Vs)) - R
            R_new = R + r_dot * dt
            B_new = BETA + b_dot * dt
            if V < 1.5:
                tan_d = math.tan(D)
                bk = math.atan(tan_d * LR / WHEELBASE)
                R_new = V * math.cos(bk) * tan_d / WHEELBASE
                B_new = bk
            PSI_new = PSI + 0.5 * (R + R_new) * dt
            course = 0.5 * (PSI + BETA + PSI_new + B_new)
            V_new = V + acc * dt
            if V_new < 0.0:
                V_new = 0.0
            Vm = 0.5 * (V + V_new)
            X += Vm * math.cos(course) * dt
            Y += Vm * math.sin(course) * dt
            along = math.cos(course - theta[ii])
            S += Vm * along * dt
            PSI = PSI_new
            R = R_new
            BETA = B_new
            V = V_new
            D = D + sv * dt
            if D > S_MAX:
                D = S_MAX
            elif D < -S_MAX:
                D = -S_MAX
            ii = (idx + int(S / ds)) % n
            e = (X - lx[ii]) * (-ty[ii]) + (Y - ly[ii]) * tx[ii]
            cth = math.cos(PSI)
            sth = math.sin(PSI)
            cmin = 1e9
            for q in range(3):
                if q == 0:
                    px = X
                    py = Y
                elif q == 1:
                    px = X + 0.27 * cth
                    py = Y + 0.27 * sth
                else:
                    px = X - 0.25 * cth
                    py = Y - 0.25 * sth
                col = int((px - ox) / res)
                row = gh - 1 - int((py - oy) / res)
                if col < 0:
                    col = 0
                elif col > gw - 1:
                    col = gw - 1
                if row < 0:
                    row = 0
                elif row > gh - 1:
                    row = gh - 1
                c = clear[row, col]
                if c < cmin:
                    cmin = c
            near = wall_soft - cmin
            if near < 0.0:
                near = 0.0
            w = w_end if h == H - 1 else 1.0
            step = w * w_line * e * e + w_wall * near * near + w_rate * (cmd - prev_cmd) ** 2
            if cmin < wall_hard:
                step += 50.0
            if along < 0.0:
                step += 20.0
            slide = abs(BETA) - slip_ok
            if slide > 0.0:
                step += w_slip * slide * slide
            total += step
            prev_cmd = cmd
        cost[k] = total / H
    return cost


try:
    from numba import njit
    try:                   # keep the compiled code between runs when we can
        _rollout_fast = njit(cache=True)(_rollout_loops)
    except Exception:
        _rollout_fast = njit(cache=False)(_rollout_loops)
except Exception:          # Numba not installed, or it cannot compile here
    _rollout_fast = None


class MPC:
    def __init__(self, line, line_speed, clearance, grid_meta, *,
                 samples=192, horizon=0.9, dt=0.03, knots=7,
                 w_line=4.0, w_wall=60.0, w_rate=0.4, w_end=3.0,
                 wall_soft=0.30, wall_hard=0.17, noise=0.07, temperature=0.4,
                 w_slip=30.0, slip_ok=0.30,
                 speed_preview=0.0, seed=0, engine='auto'):
        self.line = np.asarray(line, dtype=float)
        self.n = len(self.line)
        self.speed_ref = np.asarray(line_speed, dtype=float)
        nxt = np.roll(self.line, -1, 0)
        seg = nxt - self.line
        self.ds = float(np.mean(np.linalg.norm(seg, axis=1)))
        tang = np.roll(self.line, -1, 0) - np.roll(self.line, 1, 0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        self.tx, self.ty = tang[:, 0].copy(), tang[:, 1].copy()
        self.theta = np.arctan2(self.ty, self.tx)
        self.lx, self.ly = self.line[:, 0].copy(), self.line[:, 1].copy()
        # signed curvature of the line, for the feed-forward starting guess
        d1 = self.line - np.roll(self.line, 1, 0)
        d2 = nxt - self.line
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        a = np.linalg.norm(d1, axis=1)
        b = np.linalg.norm(d2, axis=1)
        c = np.linalg.norm(nxt - np.roll(self.line, 1, 0), axis=1)
        k = 2.0 * cross / np.maximum(a * b * c, 1e-9)
        self.kappa = np.convolve(np.r_[k[-3:], k, k[:3]], np.ones(7) / 7, 'valid')

        # wall distance map (cone rows already sealed)
        self.clear = np.asarray(clearance, dtype=np.float32)
        self.res, self.ox, self.oy = grid_meta
        self.gh, self.gw = self.clear.shape

        self.K, self.dt = int(samples), float(dt)
        self.H = max(4, int(round(horizon / dt)))
        self.M = int(knots)
        # knots -> per-step commands by linear interpolation
        t = np.linspace(0.0, self.M - 1.0, self.H)
        i0 = np.minimum(t.astype(int), self.M - 2)
        f = t - i0
        B = np.zeros((self.H, self.M))
        B[np.arange(self.H), i0] = 1.0 - f
        B[np.arange(self.H), i0 + 1] = f
        self.B = B
        self.w_line, self.w_wall, self.w_rate, self.w_end = w_line, w_wall, w_rate, w_end
        self.wall_soft, self.wall_hard = wall_soft, wall_hard
        self.w_slip, self.slip_ok = w_slip, slip_ok
        self.noise, self.temp = noise, temperature
        self.preview_pts = speed_preview
        self.rng = np.random.default_rng(seed)
        self.plan = None                    # steering offsets at the knots (M,)
        self.last_cost = 0.0
        self.engine, self.engine_note = self.choose_engine(engine)

    # ------------------------------------------------------------------
    def nearest(self, x, y, hint=None):
        if hint is None:
            return int(np.argmin(np.hypot(self.lx - x, self.ly - y)))
        w = (hint + np.arange(-5, 40)) % self.n
        i = int(w[np.argmin(np.hypot(self.lx[w] - x, self.ly[w] - y))])
        if math.hypot(self.lx[i] - x, self.ly[i] - y) > 1.0:
            return int(np.argmin(np.hypot(self.lx - x, self.ly - y)))
        return i

    def clearance_at(self, x, y):
        col = ((x - self.ox) / self.res).astype(np.int32)
        row = self.gh - 1 - ((y - self.oy) / self.res).astype(np.int32)
        np.clip(col, 0, self.gw - 1, out=col)
        np.clip(row, 0, self.gh - 1, out=row)
        return self.clear[row, col]

    # ------------------------------------------------------------------
    def solve(self, x, y, delta, v, yaw, yaw_rate, beta, idx):
        """State of the car's centre of gravity -> (steering command, cost).
        x, y: CoG position; delta: current steering angle; beta: slip angle."""
        K, H, dt = self.K, self.H, self.dt
        if self.plan is None:
            self.plan = np.zeros(self.M)
        # candidate plans = current plan + noise (sample 0 is the plan itself)
        eps = self.rng.normal(0.0, self.noise, size=(K, self.M))
        eps[0] = 0.0
        eps[1] = -self.plan                 # pure feed-forward, as a safe fallback
        knots = self.plan[None, :] + eps
        if self.engine == 'numba':
            cost = self._rollout_numba(knots, x, y, delta, v, yaw, yaw_rate, beta, idx)
        else:
            cost = self._rollout_numpy(knots, x, y, delta, v, yaw, yaw_rate, beta, idx)
        cmin_ = float(cost.min())
        # temperature is relative to how much the plans differ this cycle
        spread = max(float(cost.std()), 1e-9)
        wts = np.exp(-(cost - cmin_) / (self.temp * spread))
        wts /= wts.sum()
        self.plan = wts @ knots
        self.last_cost = cmin_
        # steering to send now: feed-forward at the car plus the plan's first offset
        cmd0 = math.atan(WHEELBASE * self.kappa[idx]) * (1.0 + KC * v * v) + float(self.plan[0])
        return max(-S_MAX, min(S_MAX, cmd0)), cmin_

    # ------------------------------------------------------------------
    def _args(self):
        return (self.lx, self.ly, self.tx, self.ty, self.theta, self.kappa, self.speed_ref,
                self.clear, float(self.res), float(self.ox), float(self.oy), float(self.dt),
                float(self.ds), int(self.preview_pts), float(self.w_line), float(self.w_wall),
                float(self.w_rate), float(self.w_end), float(self.w_slip), float(self.slip_ok),
                float(self.wall_soft), float(self.wall_hard))

    def _rollout_numba(self, knots, x, y, delta, v, yaw, yaw_rate, beta, idx):
        return _rollout_fast(np.ascontiguousarray(knots), self.B, float(x), float(y),
                             float(delta), float(v), float(yaw), float(yaw_rate), float(beta),
                             int(idx), *self._args())

    def choose_engine(self, wanted):
        """Use the compiled prediction only if it exists AND gives the same
        costs as the numpy one on a test case. Returns (engine, note)."""
        if wanted == 'numpy':
            return 'numpy', 'numpy (as requested)'
        if _rollout_fast is None:
            return 'numpy', 'numpy (Numba is not available)'
        try:
            import time
            rng = np.random.default_rng(123)
            knots = rng.normal(0.0, self.noise, size=(self.K, self.M))
            i = self.n // 3
            state = (self.lx[i] + 0.05, self.ly[i] - 0.05, 0.05, 6.0, self.theta[i] + 0.05,
                     0.5, -0.05, i)
            t0 = time.perf_counter()
            ref = self._rollout_numpy(knots, *state)
            ms_numpy = (time.perf_counter() - t0) * 1e3
            got = self._rollout_numba(knots, *state)          # compiles here
            t0 = time.perf_counter()
            for _ in range(5):
                self._rollout_numba(knots, *state)
            ms = (time.perf_counter() - t0) / 5 * 1e3
            diff = np.abs(got - ref)
            agree = float(np.mean(diff <= 1e-3 + 1e-3 * np.abs(ref)))
            if not np.all(np.isfinite(got)) or agree < 0.97:
                return 'numpy', f'numpy (compiled version disagreed on {100 * (1 - agree):.0f}% of plans)'
            if ms > ms_numpy:
                return 'numpy', f'numpy ({ms_numpy:.1f} ms; compiled version was slower: {ms:.1f} ms)'
            return 'numba', (f'numba, compiled ({ms:.2f} ms per solve instead of '
                             f'{ms_numpy:.1f} ms, same results)')
        except Exception as e:
            return 'numpy', f'numpy (Numba failed: {type(e).__name__}: {e})'

    def _rollout_numpy(self, knots, x, y, delta, v, yaw, yaw_rate, beta, idx):
        """The same prediction with numpy arrays: all K plans advance together."""
        K, H, dt = len(knots), self.H, self.dt
        offs = knots @ self.B.T             # (K, H) steering offsets

        X = np.full(K, x); Y = np.full(K, y)
        D = np.full(K, delta); V = np.full(K, max(v, 0.0))
        PSI = np.full(K, yaw); R = np.full(K, yaw_rate); BETA = np.full(K, beta)
        S = np.zeros(K)
        cost = np.zeros(K)
        prev_cmd = D.copy()
        n, ds = self.n, self.ds
        ii = np.full(K, idx, dtype=np.int64)
        for h in range(H):
            # feed-forward steering for the line's curvature here, plus the offset
            kap = self.kappa[ii]
            cmd = np.arctan(WHEELBASE * kap) * (1.0 + KC * V * V) + offs[:, h]
            np.clip(cmd, -S_MAX, S_MAX, out=cmd)
            # --- the simulator's actuators ---
            sv = np.clip(STEER_KP * (cmd - D), -SV_MAX, SV_MAX)
            vref = self.speed_ref[(ii + int(self.preview_pts)) % n]
            dv = vref - V
            acc = np.where(dv > 0.0, KP_ACCEL * dv, KP_BRAKE * dv)
            upper = np.where(V > V_SWITCH, A_MAX * V_SWITCH / np.maximum(V, 1e-3), A_MAX)
            acc = np.clip(acc, -A_MAX, upper)
            # --- single-track dynamics (kinematic below 1.5 m/s, as it is stiff there) ---
            glr = G * LR - acc * H_CG
            glf = G * LF + acc * H_CG
            Vs = np.maximum(V, 1.5)
            r_dot = (MU * MASS / (INERTIA * WHEELBASE)) * (
                LF * C_SF * glr * D + (LR * C_SR * glf - LF * C_SF * glr) * BETA
                - (LF * LF * C_SF * glr + LR * LR * C_SR * glf) * (R / Vs))
            b_dot = (MU / (Vs * WHEELBASE)) * (
                C_SF * glr * D - (C_SR * glf + C_SF * glr) * BETA
                + (C_SR * glf * LR - C_SF * glr * LF) * (R / Vs)) - R
            slow = V < 1.5
            # second-order update: new yaw rate and slip first, then move along
            # the average direction of travel over the step
            R_new = R + r_dot * dt
            B_new = BETA + b_dot * dt
            if slow.any():
                tan_d = np.tan(D)
                bk = np.arctan(tan_d * LR / WHEELBASE)
                R_new = np.where(slow, V * np.cos(bk) * tan_d / WHEELBASE, R_new)
                B_new = np.where(slow, bk, B_new)
            PSI_new = PSI + 0.5 * (R + R_new) * dt
            course = 0.5 * (PSI + BETA + PSI_new + B_new)
            V_new = np.maximum(V + acc * dt, 0.0)
            Vm = 0.5 * (V + V_new)
            X += Vm * np.cos(course) * dt
            Y += Vm * np.sin(course) * dt
            along = np.cos(course - self.theta[ii])
            S += Vm * along * dt
            PSI, R, BETA, V = PSI_new, R_new, B_new, V_new
            D = np.clip(D + sv * dt, -S_MAX, S_MAX)
            ii = (idx + (S / ds).astype(np.int64)) % n
            # --- score this step ---
            e = (X - self.lx[ii]) * (-self.ty[ii]) + (Y - self.ly[ii]) * self.tx[ii]
            cth, sth = np.cos(PSI), np.sin(PSI)
            c_mid = self.clearance_at(X, Y)
            c_front = self.clearance_at(X + 0.27 * cth, Y + 0.27 * sth)
            c_rear = self.clearance_at(X - 0.25 * cth, Y - 0.25 * sth)
            cmin = np.minimum(np.minimum(c_mid, c_front), c_rear)
            near = np.maximum(self.wall_soft - cmin, 0.0)
            w = self.w_end if h == H - 1 else 1.0
            cost += w * self.w_line * e * e + self.w_wall * near * near \
                + 50.0 * (cmin < self.wall_hard) + self.w_rate * (cmd - prev_cmd) ** 2 \
                + 20.0 * (along < 0.0)          # never drive the wrong way
            # sliding: a little slip is normal, a lot is a drift about to become a spin
            slide = np.maximum(np.abs(BETA) - self.slip_ok, 0.0)
            cost += self.w_slip * slide * slide
            prev_cmd = cmd
        return cost / H

    def shift(self, elapsed):
        """Warm start: slide the plan forward by the time that has passed."""
        if self.plan is None:
            return
        knot_dt = self.H * self.dt / (self.M - 1)
        f = min(max(elapsed / knot_dt, 0.0), 1.0)
        nxt = np.r_[self.plan[1:], 0.0]
        self.plan = (1.0 - f) * self.plan + f * nxt

    def reset(self):
        self.plan = None