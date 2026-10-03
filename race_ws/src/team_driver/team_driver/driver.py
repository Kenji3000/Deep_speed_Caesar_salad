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
  4. Fallback: `controller:=pure_pursuit` (or any MPC error) switches to Pure
     Pursuit with understeer compensation on a slower, safer speed plan.

Defaults were tuned in a replica of the judged simulator's physics: about
9.6-10 s per lap and 0 contacts over 10 laps, at command delays of 0-40 ms.
The main speed knob is `a_lat`: lower it (e.g. 16) if the car ever slides
wide or spins, raise it (up to ~21) if it is always clean.

Tune without editing code, e.g.:
    ros2 run team_driver driver --ros-args -p a_lat:=18.0 -p margin:=0.40
"""

import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
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
        self.declare_parameter('a_lat', 20.0)          # [m/s^2] cap on sideways acceleration
        self.declare_parameter('steer_use', 0.95)      # share of full steering a corner may need
        self.declare_parameter('a_brake', 8.0)         # [m/s^2] braking before corners
        self.declare_parameter('a_accel', 9.5)         # [m/s^2] acceleration out of corners
        self.declare_parameter('speed_preview', 0.0)   # [s] ask for the speed this far ahead
        # MPC
        self.declare_parameter('mpc_period', 0.02)     # [s] time between MPC solves
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
        self.declare_parameter('mpc_latency', 0.03)    # [s] delay between deciding and the car acting
        self.declare_parameter('mpc_seed', 0)          # random seed of the plan sampler
        # Pure Pursuit (fallback)
        self.declare_parameter('pp_a_lat', 5.0)        # [m/s^2] safer speed plan for the fallback
        self.declare_parameter('pp_v_max', 9.0)
        self.declare_parameter('ld_base', 0.45)        # [m] lookahead = ld_base + ld_gain*speed
        self.declare_parameter('ld_gain', 0.15)
        self.declare_parameter('max_steer', 0.4189)    # [rad] car limit
        self.declare_parameter('understeer', 1.0)      # 0 = off, 1 = full understeer compensation
        # Safety
        self.declare_parameter('stop_dist', 0.35)      # [m] wall this close dead ahead -> crawl
        self.declare_parameter('soft_start', 1.0)      # [s] limit speed to 3 m/s this long

        self._raw = lambda n: self.get_parameter(n).value
        g = self.safe_param
        self.max_range = g('max_range')
        self.ld_base, self.ld_gain = g('ld_base'), g('ld_gain')
        self.max_steer, self.stop_dist = g('max_steer'), g('stop_dist')
        self.speed_preview = g('speed_preview')
        self.understeer = g('understeer')
        self.soft_start = g('soft_start')
        self.wheelbase = 0.33
        self.controller = str(self._raw('controller'))

        # ---- the line and its speeds (computed once) ----
        self.grid = None
        self.line = self.make_line(g)
        self.line_speed = raceline.speed_profile(
            self.line, g('v_max'), g('v_min'), g('a_lat'), g('a_brake'), g('a_accel'),
            g('steer_use'))
        self.pp_speed = raceline.speed_profile(
            self.line, min(g('pp_v_max'), g('v_max')), g('v_min'), min(g('pp_a_lat'), g('a_lat')),
            g('a_brake'), g('a_accel'), 1.0)
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
                    slip_ok=g('mpc_slip_ok'), seed=int(self._raw('mpc_seed')))
        self.mpc_period, self.mpc_latency = g('mpc_period'), g('mpc_latency')
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

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, g('drive_topic'), 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)
        self.create_subscription(LaserScan, g('scan_topic'), self.scan_callback, 10)
        self.create_subscription(Odometry, g('odom_topic'), self.odom_callback, 10)
        # When the car is teleported, forget where we were on the line.
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.reset_callback, 10)
        self._marker_divisor = 0

    def reset_state(self):
        self.idx = None                # nearest line point to the rear axle (Pure Pursuit)
        self.idx_cog = None            # nearest line point to the centre of gravity (MPC)
        self.steer_cmd = 0.0           # last steering command sent
        self.steer_est = 0.0           # our estimate of the actual wheel angle
        self.cmd_hist = [(-1e9, 0.0)]  # (time sent, steering command), oldest first
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
        'speed_preview': (0.0, 1.0), 'understeer': (0.0, 2.0),
        'ld_base': (0.25, 2.0), 'ld_gain': (0.0, 0.5),
        'max_steer': (0.1, 0.4189), 'stop_dist': (0.0, 2.0), 'soft_start': (0.0, 5.0),
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
        """The control loop. It runs on every odometry message, because the
        simulator publishes the scan first and the odometry just after it:
        acting on odometry means acting on the newest state."""
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
        self.publish(steering, speed)

        self._marker_divisor = (self._marker_divisor + 1) % 200
        if self._marker_divisor % 25 == 0:
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
        if self.position is None:
            return 0.0, 0.0
        now = self.sim_time if self.sim_time is not None else \
            self.get_clock().now().nanoseconds * 1e-9
        # Simulated time jumps backwards when the car is reset: start afresh.
        if self.last_time is not None and now < self.last_time - 1e-6:
            self.reset_state()
        dt = 0.0 if self.last_time is None else min(now - self.last_time, 0.05)
        self.last_time = now
        if self.start_time is None:
            self.start_time = now

        # The wheel angle is not published, so track it ourselves with the
        # simulator's steering actuator: P control, limited to 3.2 rad/s.
        #  A command only reaches the wheels `mpc_latency` after it was decided.
        if dt > 0.0:
            self.steer_est = self.advance_steer(self.steer_est, now - dt, now)

        if self.controller == 'mpc' and self.mpc is not None:
            try:
                steering, speed = self.plan_mpc(now)
            except Exception as e:
                self.get_logger().error(f'MPC failed ({e}); switching to Pure Pursuit')
                self.controller = 'pure_pursuit'
                steering, speed = self.plan_pure_pursuit()
        else:
            steering, speed = self.plan_pure_pursuit()

        # Safety net: something right in front of the bumper -> crawl.
        if self.stop_dist > 0.0:
            ahead = ranges[np.abs(angles) < math.radians(8)]
            if ahead.size and ahead.min() < self.stop_dist:
                speed = min(speed, 0.5)
        if now - self.start_time < self.soft_start:
            speed = min(speed, 3.0)
        self.steer_cmd = steering
        if steering != self.cmd_hist[-1][1]:
            self.cmd_hist.append((now, steering))
            if len(self.cmd_hist) > 64:
                del self.cmd_hist[:32]
        return steering, speed

    def command_active_at(self, t):
        """The steering command the car is acting on at time t (sent earlier)."""
        t_sent = t - self.mpc_latency
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
        if self.last_solve is None or now - self.last_solve >= self.mpc_period - 1e-6:
            elapsed = 0.0 if self.last_solve is None else now - self.last_solve
            self.last_solve = now
            # plan from where the car will be when the command takes effect
            tau = self.mpc_latency
            if tau > 0.0:
                course = yaw + self.beta + 0.5 * self.yaw_rate * tau
                x += v * math.cos(course) * tau
                y += v * math.sin(course) * tau
                yaw += self.yaw_rate * tau
                # the wheels keep following commands already on their way
                delta = self.advance_steer(delta, now, now + tau)
            self.mpc.shift(elapsed)
            self._mpc_steer, _ = self.mpc.solve(
                x, y, delta, v, yaw, self.yaw_rate, self.beta, self.idx_cog)
        ahead_pts = int(round(self.speed * self.speed_preview / self.spacing))
        speed = float(self.line_speed[(self.idx_cog + ahead_pts) % len(self.line)])
        return self._mpc_steer, speed

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

        if with_line:
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