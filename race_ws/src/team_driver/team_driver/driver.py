#!/usr/bin/env python3
"""Team driver: Model Predictive Control along a minimum-time racing line.

How it works
  1. At start-up, raceline.py reads the track map, seals the cone rows, traces
     a lap, bends it into a minimum-curvature line and then optimises it for
     minimum LAP TIME, staying `margin` metres from every wall. Every point
     gets a target speed from the simulator's real limits: steering
     capability, braking, acceleration.
  2. Steering comes from mpc.py: about 50 times a second it simulates a few
     hundred candidate steering plans one second ahead with the simulator's
     own physics, scores them (distance from the line, closeness to walls,
     smoothness) and sends the first command of the best blend.
  3. Speed is the line's target speed at the car's position; the simulator's
     own speed controller does the rest.
     The MPC also allows for the delay between deciding and the car acting
     (`mpc_latency`), and keeps its own estimate of the wheel angle, which the
     simulator does not publish.
  4. Message handling matters as much as the maths. The simulator takes one
     command per physics step from a 10-deep queue, so the driver sends at
     most one command every two steps, ignores repeated or stale odometry,
     and measures the real command delay from how the car's speed responds.
  5. Fallback: `controller:=pure_pursuit` (or any MPC error) switches to Pure
     Pursuit with understeer compensation on a slower, safer speed plan.

Defaults were chosen in a replica of the judged simulator's physics with
irregular command delays (10-90 ms): about 10.6 s per lap, 0 contacts over 10
laps. The main speed knob is `a_lat`: 17 is the safe default; 20 gave about
10.0 s but can slide into a wall when the machine's timing is poor. Watch the
"MPC: ..." log line: it shows how long each solve takes and the biggest slide.

Tune without editing code, e.g.:
    ros2 run team_driver driver --ros-args -p a_lat:=18.0 -p margin:=0.40
"""

import collections
import math
import os
import threading
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from team_driver import mpc, raceline


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        # ---- parameters (all tunable with -p name:=value or a params YAML) ----
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('max_range', 8.0)       # [m] clip the scan here
        self.declare_parameter('controller', 'mpc')    # 'mpc' or 'pure_pursuit'
        # Line
        self.declare_parameter('tracks_yaml', '/hackathon/maps/tracks.yaml')
        self.declare_parameter('line_csv', '')         # set to a CSV to use that instead
        self.declare_parameter('margin', 0.35)         # [m] line centre to wall, minimum
        self.declare_parameter('line_blend', 1.0)      # 0 = centreline, 1 = racing line
        self.declare_parameter('optimise_time', True)  # min-time line (else min-curvature)
        # Speed plan (the simulated car brakes/accelerates up to 9.51 m/s^2;
        # corners are limited by steering, not grip - see raceline.py)
        self.declare_parameter('v_max', 11.0)          # [m/s] top speed
        self.declare_parameter('v_min', 1.5)           # [m/s] slowest target speed
        self.declare_parameter('a_lat', 17.0)          # [m/s^2] cap on sideways acceleration
        self.declare_parameter('steer_use', 0.95)      # share of full steering a corner may need
        self.declare_parameter('a_brake', 8.0)         # [m/s^2] braking before corners
        self.declare_parameter('trail', 1.0)           # 0..1: ease off the brakes while turning in
        self.declare_parameter('a_accel', 9.5)         # [m/s^2] acceleration out of corners
        self.declare_parameter('speed_preview', 0.0)   # [s] ask for the speed this far ahead
        # MPC
        self.declare_parameter('mpc_period', 0.02)     # [s] time between commands (and MPC solves)
        self.declare_parameter('mpc_samples', 192)     # candidate plans per solve
        self.declare_parameter('mpc_horizon', 0.9)     # [s] how far ahead it predicts
        self.declare_parameter('mpc_dt', 0.03)         # [s] prediction step
        self.declare_parameter('mpc_knots', 7)         # steering plan resolution
        self.declare_parameter('mpc_w_line', 8.0)      # weight: distance from the line
        self.declare_parameter('mpc_w_wall', 60.0)     # weight: closeness to walls
        self.declare_parameter('mpc_w_rate', 0.4)      # weight: jerky steering
        self.declare_parameter('mpc_w_slip', 30.0)     # weight: sliding sideways too much
        self.declare_parameter('mpc_slip_ok', 0.30)    # [rad] slide angle that is still fine
        self.declare_parameter('mpc_wall_soft', 0.30)  # [m] start avoiding walls inside this
        self.declare_parameter('mpc_noise', 0.13)      # [rad] spread of candidate plans
        self.declare_parameter('mpc_temperature', 0.1) # lower = trust only the best plans
        self.declare_parameter('mpc_latency', 0.02)    # [s] command delay assumed until it is measured
        self.declare_parameter('delay_auto', True)     # measure the real command delay while driving
        self.declare_parameter('mpc_seed', 0)          # random seed of the plan sampler
        self.declare_parameter('mpc_engine', 'auto')   # 'auto' (compiled if possible) or 'numpy'
        # Pure Pursuit (fallback)
        self.declare_parameter('pp_a_lat', 5.0)        # [m/s^2] safer speed plan for the fallback
        self.declare_parameter('pp_v_max', 9.0)
        self.declare_parameter('ld_base', 0.45)        # [m] lookahead = ld_base + ld_gain*speed
        self.declare_parameter('ld_gain', 0.15)
        self.declare_parameter('max_steer', 0.4189)    # [rad] car limit
        self.declare_parameter('understeer', 1.0)      # 0 = off, 1 = full understeer compensation
        # Safety
        self.declare_parameter('esc_slip', 0.40)       # [rad] sliding more than this -> stop braking hard
        self.declare_parameter('esc_brake', 2.0)       # [m/s^2] braking still allowed in a slide
        self.declare_parameter('stop_dist', 0.35)      # [m] wall this close dead ahead -> crawl
        self.declare_parameter('soft_start', 1.0)      # [s] limit speed to 3 m/s this long
        # Black box: on a collision, save the last 10 s of the car's state
        self.declare_parameter('record', True)
        self.declare_parameter('record_dir', '/hackathon/results/blackbox')

        self._raw = lambda n: self.get_parameter(n).value
        g = self.safe_param
        self.max_range = g('max_range')
        self.ld_base, self.ld_gain = g('ld_base'), g('ld_gain')
        self.max_steer, self.stop_dist = g('max_steer'), g('stop_dist')
        self.speed_preview = g('speed_preview')
        self.understeer = g('understeer')
        self.soft_start = g('soft_start')
        self.esc_slip, self.esc_brake = g('esc_slip'), g('esc_brake')
        self.wheelbase = 0.33
        self.controller = str(self._raw('controller'))

        # ---- the line and its speeds (computed once) ----
        self.grid = None
        self.line = self.make_line(g)
        self.line_speed = raceline.speed_profile(
            self.line, g('v_max'), g('v_min'), g('a_lat'), g('a_brake'), g('a_accel'),
            g('steer_use'), g('trail'))
        self.pp_speed = raceline.speed_profile(
            self.line, min(g('pp_v_max'), g('v_max')), g('v_min'), min(g('pp_a_lat'), g('a_lat')),
            g('a_brake'), g('a_accel'), 1.0, g('trail'))
        self.spacing = raceline.lap_length(self.line) / len(self.line)
        self.get_logger().info(
            f'{len(self.line)} line points, {raceline.lap_length(self.line):.1f} m, '
            f'speeds {self.line_speed.min():.1f}-{self.line_speed.max():.1f} m/s')

        self.mpc = None
        if self.controller == 'mpc':
            if self.grid is None:
                self.get_logger().error('No map available for MPC; using Pure Pursuit')
                self.controller = 'pure_pursuit'
            else:
                self.mpc = mpc.MPC(
                    self.line, self.line_speed, self.grid.clearance,
                    (self.grid.res, self.grid.ox, self.grid.oy),
                    samples=int(g('mpc_samples')), horizon=g('mpc_horizon'), dt=g('mpc_dt'),
                    knots=int(g('mpc_knots')), w_line=g('mpc_w_line'), w_wall=g('mpc_w_wall'),
                    w_rate=g('mpc_w_rate'), wall_soft=g('mpc_wall_soft'), noise=g('mpc_noise'),
                    temperature=g('mpc_temperature'), w_slip=g('mpc_w_slip'),
                    slip_ok=g('mpc_slip_ok'), seed=int(self._raw('mpc_seed')),
                    engine=str(self._raw('mpc_engine')))
                self.get_logger().info(f'MPC engine: {self.mpc.engine_note}')
        self.mpc_period, self.mpc_latency = g('mpc_period'), g('mpc_latency')
        self.delay = self.mpc_latency  # [s] delay between deciding and the car acting
        self.delay_auto = bool(self._raw('delay_auto'))
        self.delay_err = [0.0] * 12    # running prediction error for delays of 1..12 steps
        self.solve_ema = 0.0           # [s] how long an MPC solve takes on this machine
        self.rtf = 1.0                 # simulated seconds per real second on this machine
        self._wall_prev = None
        self.get_logger().info(f'Controller: {self.controller}')

        # Latest ground-truth state (allowed by the rules)
        self.position = None           # rear axle (base_link), map frame
        self.yaw = 0.0
        self.yaw_rate = 0.0            # [rad/s]
        self.beta = 0.0                # [rad] slip angle: travel direction minus yaw
        self.speed = 0.0
        self.sim_time = None           # [s] simulated time of the latest odometry
        self.ranges = np.zeros(0)      # latest LiDAR scan (for the safety net)
        self.angles = np.zeros(0)
        self.reset_state()

        # Queue depth 1 everywhere. The simulator takes ONE /drive message per
        # physics step from a 10-deep queue, so any extra message we send
        # stays in that queue and makes every later command 10 ms staler - for
        # good. We therefore never work through a backlog and never send more
        # often than every second step (see plan()).
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, g('drive_topic'), 1)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)
        self.create_subscription(LaserScan, g('scan_topic'), self.scan_callback, 1)
        self.create_subscription(Odometry, g('odom_topic'), self.odom_callback, 1)
        self.record = bool(self._raw('record'))
        self.record_dir = str(self._raw('record_dir'))
        self.rec = collections.deque(maxlen=1000)      # one row per control step (10 ms)
        self.dumps = 0
        self.in_contact = False
        self.create_subscription(Bool, '/ego_racecar/collision', self.collision_callback, 1)
        # When the car is teleported, forget where we were on the line.
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.reset_callback, 10)
        self._marker_divisor = 0
        self._line_marker = None

    def reset_state(self):
        self.idx = None                # nearest line point to the rear axle (Pure Pursuit)
        self.idx_cog = None            # nearest line point to the centre of gravity (MPC)
        self.steer_cmd = 0.0           # last steering command sent
        self.steer_est = 0.0           # our estimate of the actual wheel angle
        self.cmd_hist = [(-1e9, 0.0)]  # (time sent, steering command), oldest first
        self.spd_hist = [(-1e9, 0.0)]  # (time sent, speed command), oldest first
        self.v_hist = collections.deque(maxlen=80)   # (time, measured speed), one per step
        self.last_cmd = (0.0, 0.0)     # last (steering, speed) sent
        self.last_pub = None           # race time of the last command sent
        self.publish_now = False       # did plan() produce a new command to send?
        self.skipped = 0               # duplicate or stale messages ignored
        self._steps = 0
        self.diag = dict(t0=None, n=0, ms=0.0, worst=0.0, slip=0.0, vmax=0.0)
        self._wall_prev = None
        self.last_time = None
        self.last_solve = None
        self.start_time = None
        if getattr(self, 'mpc', None) is not None:
            self.mpc.reset()

    # Values outside these ranges are what make the car crash or the line leave
    # the track, so they are pulled back in (with a warning) instead of used.
    SAFE = {
        'margin': (0.28, 0.80), 'line_blend': (0.0, 1.0),
        'v_max': (1.0, 20.0), 'v_min': (0.5, 4.0), 'steer_use': (0.3, 1.0),
        'a_lat': (1.0, 80.0), 'a_brake': (1.0, 9.5), 'a_accel': (0.5, 9.5),
        'speed_preview': (0.0, 1.0), 'understeer': (0.0, 2.0), 'trail': (0.0, 1.5),
        'ld_base': (0.25, 2.0), 'ld_gain': (0.0, 0.5),
        'max_steer': (0.1, 0.4189), 'stop_dist': (0.0, 2.0), 'soft_start': (0.0, 5.0),
        'esc_slip': (0.1, 3.0), 'esc_brake': (0.0, 9.5),
        'mpc_period': (0.01, 0.1), 'mpc_samples': (16, 2048), 'mpc_horizon': (0.3, 2.0),
        'mpc_dt': (0.01, 0.06), 'mpc_knots': (3, 20), 'mpc_noise': (0.005, 0.4),
        'mpc_temperature': (0.02, 5.0), 'mpc_latency': (0.0, 0.1),
        'mpc_wall_soft': (0.18, 0.8), 'pp_a_lat': (1.0, 9.0), 'pp_v_max': (1.0, 12.0),
    }

    def safe_param(self, name):
        value = self._raw(name)
        if name in self.SAFE:
            lo, hi = self.SAFE[name]
            if not lo <= value <= hi:
                clamped = min(hi, max(lo, value))
                self.get_logger().warn(f'{name}={value} is outside [{lo}, {hi}]; using {clamped}')
                value = clamped
        return value

    def make_line(self, g):
        """Racing line from the map; the CSV if one is given; the shipped
        centreline if the map step fails for any reason."""
        try:
            self.grid, _, _ = raceline.load_grid(g('tracks_yaml'))
        except Exception as e:
            self.get_logger().error(f'Could not load the map ({e})')
        if g('line_csv'):
            self.get_logger().info(f"Using line from {g('line_csv')}")
            return raceline.load_csv(g('line_csv'))
        try:
            car = dict(v_max=g('v_max'), a_lat=g('a_lat'), a_brake=g('a_brake'),
                       a_accel=g('a_accel'), steer_use=g('steer_use'))
            _, race = raceline.build_raceline(
                g('tracks_yaml'), margin=g('margin'), blend=g('line_blend'),
                optimise_time=bool(self._raw('optimise_time')), car=car,
                log=self.get_logger().info, grid=self.grid)
            return race
        except Exception as e:  # never fail to drive because of the planner
            self.get_logger().error(f'Racing line failed ({e}); using the centreline CSV')
            return raceline.load_csv('/hackathon/maps/icra26_centerline.csv')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def reset_callback(self, msg):
        self.reset_state()
        self.get_logger().info('Car was moved - searching for the line again.')

    def odom_callback(self, msg):
        """The control loop. It runs on odometry, because the simulator
        publishes the scan first and the odometry just after it: acting on
        odometry means acting on the newest state. A command is sent only
        when plan() says a new one is due."""
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        self.speed = math.hypot(vx, vy)
        # The simulator reports the velocity of the centre of gravity in the
        # car's own frame, so its angle is the slip angle.
        self.beta = math.atan2(vy, vx) if self.speed > 0.3 else 0.0
        self.yaw_rate = msg.twist.twist.angular.z
        self.sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        steering, speed = self.plan(self.ranges, self.angles)
        if self.publish_now:
            self.publish(steering, speed)

        # Drawing is only for RViz: skip it entirely when nobody is watching.
        self._marker_divisor = (self._marker_divisor + 1) % 500
        if self._marker_divisor % 25 == 0 and self.marker_pub.get_subscription_count() > 0:
            self.publish_markers(steering, with_line=self._marker_divisor == 0)

    def scan_callback(self, scan):
        """Keep the latest LiDAR scan for the safety net."""
        self.ranges, self.angles = self.preprocess(scan)

    def preprocess(self, scan):
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    # ==================================================================
    # Decide steering and speed
    # ==================================================================
    def plan(self, ranges, angles):
        """Called for every odometry message. Returns (steering, speed) and sets
        self.publish_now: a NEW command is produced only every `mpc_period`."""
        if self.position is None:
            self.publish_now = False
            return 0.0, 0.0
        now = self.sim_time if self.sim_time is not None else \
            self.get_clock().now().nanoseconds * 1e-9
        if self.last_time is not None:
            if now < self.last_time - 0.5:
                self.reset_state()          # race time jumped back: the car was reset
            elif now <= self.last_time + 1e-9:
                self.skipped += 1           # duplicate or out-of-date message: never
                self.publish_now = False    # answer it with another command
                return self.last_cmd
        dt = 0.0 if self.last_time is None else min(now - self.last_time, 0.05)
        self.last_time = now
        if self.start_time is None:
            self.start_time = now

        # The wheel angle is not published, so track it ourselves with the
        # simulator's steering actuator, following the commands as they arrive.
        if dt > 0.0:
            self.steer_est = self.advance_steer(self.steer_est, now - dt, now)
        self.v_hist.append((now, self.speed))
        self._steps += 1
        if self.delay_auto and self._steps % 25 == 0:
            self.measure_delay()

        if self.last_pub is not None and now - self.last_pub < self.mpc_period - 1e-6:
            self.publish_now = False        # not time for a new command yet
            if self.record:
                self.remember(now, self.last_cmd[0], self.last_cmd[1], ranges, angles)
            return self.last_cmd

        if self.controller == 'mpc' and self.mpc is not None:
            try:
                steering, speed = self.plan_mpc(now)
            except Exception as e:
                self.get_logger().error(f'MPC failed ({e}); switching to Pure Pursuit')
                self.controller = 'pure_pursuit'
                steering, speed = self.plan_pure_pursuit()
        else:
            steering, speed = self.plan_pure_pursuit()

        # Stability control: braking hard unloads the rear wheels, which is
        # what turns a slide into a spin. While sliding a lot, brake gently.
        if abs(self.beta) > self.esc_slip and speed < self.speed:
            speed = max(speed, self.speed - self.esc_brake / mpc.KP_BRAKE)
        # Safety net: something right in front of the bumper -> crawl.
        if self.stop_dist > 0.0:
            ahead = ranges[np.abs(angles) < math.radians(8)]
            if ahead.size and ahead.min() < self.stop_dist:
                speed = min(speed, 0.5)
        if now - self.start_time < self.soft_start:
            speed = min(speed, 3.0)
        if self.record:
            self.remember(now, steering, speed, ranges, angles)
        self.steer_cmd = steering
        self.last_cmd = (steering, speed)
        self.last_pub = now
        self.publish_now = True
        self.cmd_hist.append((now, steering))
        self.spd_hist.append((now, speed))
        if len(self.cmd_hist) > 64:
            del self.cmd_hist[:32]
            del self.spd_hist[:32]
        return steering, speed

    def measure_delay(self):
        """How long after we send a command does the car act on it? The
        simulator's speed controller is known exactly, so for each candidate
        delay we predict the next speed from the speed command that would have
        been active, and keep the delay whose predictions fit best."""
        hist = list(self.v_hist)
        if len(hist) < 40:
            return
        errs = [0.0] * 12
        change, pairs = 0.0, 0
        for (t0, v0), (t1, v1) in zip(hist[:-1], hist[1:]):
            h = t1 - t0
            if abs(h - 0.01) > 0.002 or v0 < 0.5:
                continue
            pairs += 1
            change += abs(v1 - v0)
            upper = mpc.A_MAX * mpc.V_SWITCH / v0 if v0 > mpc.V_SWITCH else mpc.A_MAX
            for k in range(12):
                t_sent = t0 - (k + 1) * 0.01 + 1e-6
                cmd = self.spd_hist[0][1]
                for when, c in reversed(self.spd_hist):
                    if when <= t_sent:
                        cmd = c
                        break
                dv = cmd - v0
                acc = mpc.KP_ACCEL * dv if dv > 0.0 else mpc.KP_BRAKE * dv
                acc = max(-mpc.A_MAX, min(upper, acc))
                errs[k] += min((v0 + acc * h - v1) ** 2, 0.01)
        if pairs < 30 or change < 0.3:      # not enough speeding up / braking to tell
            return
        self.delay_err = [0.8 * e0 + 0.2 * e / pairs for e0, e in zip(self.delay_err, errs)]
        best = min(range(12), key=lambda k: self.delay_err[k])
        self.delay = (best + 1) * 0.01

    # ------------------------------------------------------------------
    # Black box recorder
    # ------------------------------------------------------------------
    REC_COLUMNS = ('t,x,y,yaw,speed,yaw_rate,slip,steer_cmd,steer_est,speed_cmd,line_idx,'
                   'line_error,wall_clearance,line_speed,lidar_ahead,solve_ms,cmd_delay,rtf')

    def remember(self, now, steering, speed, ranges, angles):
        x, y = self.position
        cx, cy = x + mpc.LR * math.cos(self.yaw), y + mpc.LR * math.sin(self.yaw)
        idx = self.idx_cog if self.idx_cog is not None else (self.idx or 0)
        lx, ly = self.line[idx]
        nxt = self.line[(idx + 1) % len(self.line)]
        tx, ty = nxt[0] - lx, nxt[1] - ly
        norm = math.hypot(tx, ty) or 1.0
        err = (-(cx - lx) * ty + (cy - ly) * tx) / norm          # + = left of the line
        clear = float(self.grid.clearance_at(np.array([cx]), np.array([cy]))[0]) \
            if self.grid is not None else -1.0
        ahead = ranges[np.abs(angles) < math.radians(8)] if len(ranges) else ranges
        self.rec.append((now, x, y, self.yaw, self.speed, self.yaw_rate, self.beta, steering,
                         self.steer_est, speed, idx, err, clear, float(self.line_speed[idx]),
                         float(ahead.min()) if len(ahead) else -1.0, self.solve_ema * 1e3,
                         self.delay, self.rtf))

    def collision_callback(self, msg):
        hit = bool(msg.data)
        if hit and not self.in_contact and self.record:
            self.dump('crash')
        self.in_contact = hit

    def dump(self, label):
        """Save the black box. The file is written by a background thread, so
        the control loop never waits for the disk."""
        self.dumps += 1
        if self.dumps > 6 or not self.rec:
            return
        data = np.array(self.rec)
        path = os.path.join(self.record_dir,
                            f'{label}_{time.strftime("%H%M%S")}_{self.dumps}.csv')
        last = self.rec[-1]
        self.get_logger().warn(
            f'Black box: {path} ({label} at x={last[1]:.2f} y={last[2]:.2f}, '
            f'{last[4]:.1f} m/s, slide {math.degrees(last[6]):.0f} deg, '
            f'{last[11]:+.2f} m from the line)')

        def write():
            try:
                os.makedirs(self.record_dir, exist_ok=True)
                np.savetxt(path, data, delimiter=',', fmt='%.5f',
                           header=self.REC_COLUMNS, comments='')
            except Exception:
                pass
        threading.Thread(target=write, daemon=True).start()

    def command_active_at(self, t):
        """The steering command the car is acting on at time t (sent earlier)."""
        t_sent = t - self.delay
        for when, cmd in reversed(self.cmd_hist):
            if when <= t_sent + 1e-9:
                return cmd
        return self.cmd_hist[0][1]

    def advance_steer(self, delta, t0, t1):
        """Move the wheel-angle estimate from t0 to t1 with the simulator's
        steering actuator, following the commands that were active then."""
        t = t0
        while t < t1 - 1e-9:
            h = min(0.005, t1 - t)
            rate = mpc.STEER_KP * (self.command_active_at(t) - delta)
            rate = max(-mpc.SV_MAX, min(mpc.SV_MAX, rate))
            delta = max(-mpc.S_MAX, min(mpc.S_MAX, delta + rate * h))
            t += h
        return delta

    def plan_mpc(self, now):
        # centre of gravity = rear axle + LR along the car
        x = self.position[0] + mpc.LR * math.cos(self.yaw)
        y = self.position[1] + mpc.LR * math.sin(self.yaw)
        yaw, delta, v = self.yaw, self.steer_est, self.speed
        self.idx_cog = self.mpc.nearest(x, y, self.idx_cog)
        elapsed = 0.0 if self.last_solve is None else now - self.last_solve
        self.last_solve = now
        t_wall = time.perf_counter()
        # how fast the simulator runs on this machine (for the log)
        if self._wall_prev is not None and elapsed > 0.0 and t_wall > self._wall_prev:
            ratio = min(max(elapsed / (t_wall - self._wall_prev), 0.1), 10.0)
            self.rtf = 0.95 * self.rtf + 0.05 * ratio
        self._wall_prev = t_wall
        # Plan from where the car will be when this command takes effect:
        # until then it keeps following the commands already on their way.
        tau = self.delay
        if tau > 0.0:
            course = yaw + self.beta + 0.5 * self.yaw_rate * tau
            x += v * math.cos(course) * tau
            y += v * math.sin(course) * tau
            yaw += self.yaw_rate * tau
            delta = self.advance_steer(delta, now, now + tau)
        self.mpc.shift(elapsed)
        self._mpc_steer, _ = self.mpc.solve(
            x, y, delta, v, yaw, self.yaw_rate, self.beta, self.idx_cog)
        took = time.perf_counter() - t_wall
        self.solve_ema = took if self.solve_ema == 0.0 else 0.9 * self.solve_ema + 0.1 * took
        self.report(now, took)
        ahead_pts = int(round(self.speed * self.speed_preview / self.spacing))
        speed = float(self.line_speed[(self.idx_cog + ahead_pts) % len(self.line)])
        return self._mpc_steer, speed

    def report(self, now, took):
        """Every 5 s of race time, log how the MPC is doing on this machine,
        and lighten it if solving takes too long to keep up."""
        dg = self.diag
        if dg['t0'] is None:
            dg['t0'] = now
        dg['n'] += 1
        dg['ms'] += took * 1e3
        dg['worst'] = max(dg['worst'], took * 1e3)
        dg['slip'] = max(dg['slip'], abs(self.beta))
        dg['vmax'] = max(dg['vmax'], self.speed)
        if now - dg['t0'] >= 5.0:
            self.get_logger().info(
                f"MPC: {dg['n'] / (now - dg['t0']):.0f} solves/s, {dg['ms'] / dg['n']:.1f} ms each "
                f"(worst {dg['worst']:.0f} ms), biggest slide {math.degrees(dg['slip']):.0f} deg, "
                f"top speed {dg['vmax']:.1f} m/s, command delay {self.delay * 1e3:.0f} ms, "
                f"{self.skipped} repeat messages ignored, sim at {self.rtf:.1f}x real time")
            if dg['ms'] / dg['n'] * self.rtf > 600.0 * self.mpc_period and self.mpc.K > 64:
                self.mpc.K = max(64, int(self.mpc.K * 0.75))
                self.get_logger().warn(
                    f'MPC is slow on this machine; using {self.mpc.K} plans per solve')
            self.diag = dict(t0=now, n=0, ms=0.0, worst=0.0, slip=0.0, vmax=0.0)

    def plan_pure_pursuit(self):
        x, y = self.position
        xy, n = self.line, len(self.line)
        # 1. nearest line point (search just around the last one)
        if self.idx is None:
            self.idx = int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))
        else:
            w = (self.idx + np.arange(-5, 40)) % n
            self.idx = int(w[np.argmin(np.hypot(xy[w, 0] - x, xy[w, 1] - y))])
        if math.hypot(xy[self.idx, 0] - x, xy[self.idx, 1] - y) > 1.0:
            self.idx = int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))
        # 2. walk forward along the line to the lookahead point
        ld = self.ld_base + self.ld_gain * self.speed
        j = self.idx
        while math.hypot(xy[j, 0] - x, xy[j, 1] - y) < ld:
            j = (j + 1) % n
            if j == self.idx:
                break
        # 3. steer along the arc through that point, plus understeer compensation
        dx, dy = xy[j, 0] - x, xy[j, 1] - y
        lateral = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy
        steering = math.atan(2.0 * self.wheelbase * lateral / max(dx * dx + dy * dy, 1e-6))
        steering *= 1.0 + self.understeer * raceline.KC * self.speed ** 2
        steering = max(-self.max_steer, min(self.max_steer, steering))
        return steering, float(self.pp_speed[self.idx])

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def publish(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def publish_markers(self, target_angle, with_line=False):
        """Arrow = where the car is steering. Line = the racing line, coloured
        by target speed (blue slow -> red fast). Shown by 'Driver markers'."""
        array = MarkerArray()
        arrow = Marker()
        arrow.header.frame_id = 'ego_racecar/base_link'
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns, arrow.id = 'team_driver', 0
        arrow.type, arrow.action = Marker.ARROW, Marker.ADD
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 1.5, 0.15, 0.15
        arrow.color.g, arrow.color.b, arrow.color.a = 0.8, 1.0, 0.9
        arrow.pose.orientation.z = math.sin(target_angle / 2.0)
        arrow.pose.orientation.w = math.cos(target_angle / 2.0)
        array.markers.append(arrow)

        if with_line and self._line_marker is not None:
            array.markers.append(self._line_marker)
        elif with_line:
            line = Marker()
            line.header.frame_id = 'map'
            line.ns, line.id = 'raceline', 1
            line.type, line.action = Marker.LINE_STRIP, Marker.ADD
            line.scale.x = 0.05
            line.pose.orientation.w = 1.0
            vmin, vmax = float(self.line_speed.min()), float(self.line_speed.max())
            for k in list(range(len(self.line))) + [0]:
                line.points.append(Point(x=float(self.line[k, 0]), y=float(self.line[k, 1]), z=0.02))
                f = (self.line_speed[k] - vmin) / max(vmax - vmin, 1e-6)
                line.colors.append(ColorRGBA(r=float(f), g=0.2, b=float(1.0 - f), a=1.0))
            self._line_marker = line          # built once, reused afterwards
            array.markers.append(line)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = Driver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()