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

Only numpy is needed. All K plans are simulated as numpy arrays, so one cycle
costs a few milliseconds.
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


class MPC:
    def __init__(self, line, line_speed, clearance, grid_meta, *,
                 samples=192, horizon=0.9, dt=0.03, knots=7,
                 w_line=4.0, w_wall=60.0, w_rate=0.4, w_end=3.0,
                 wall_soft=0.30, wall_hard=0.17, noise=0.07, temperature=0.4,
                 w_slip=30.0, slip_ok=0.30,
                 speed_preview=0.0, seed=0):
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
        cost = cost / H
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