#!/usr/bin/env python3
"""Team driver: Pure Pursuit along a racing line generated from the map.

How it works
  1. At start-up, raceline.py reads the track map, seals the cone rows, traces
     a lap, and bends it into a minimum-curvature racing line (wide in, apex,
     wide out) that stays `margin` metres from every wall. It then gives every
     point a target speed from grip, braking and acceleration limits.
  2. Every LiDAR scan (~40 Hz), take the car's exact position from
     /ego_racecar/odom, find the nearest point on the line, look a little way
     ahead along it, and steer along the arc that reaches that point.
  3. Safety net: if the LiDAR sees something very close straight ahead, slow
     right down.

Tune without editing code, e.g.:
    ros2 run team_driver driver --ros-args -p a_lat:=5.0 -p margin:=0.45
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

from team_driver import raceline


class Driver(Node):

    def __init__(self):
        super().__init__('driver')

        # ---- parameters (all tunable with -p name:=value or a params YAML) ----
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('max_range', 8.0)       # [m] clip the scan here
        # Line
        self.declare_parameter('tracks_yaml', '/hackathon/maps/tracks.yaml')
        self.declare_parameter('line_csv', '')         # set to a CSV to use that instead
        self.declare_parameter('margin', 0.4)         # [m] line centre to wall, minimum
        self.declare_parameter('line_blend', 1.0)      # 0 = centreline, 1 = racing line
        # Speed profile
        self.declare_parameter('v_max', 9.2)           # [m/s] top speed
        self.declare_parameter('v_min', 3.8)           # [m/s] slowest target speed
        self.declare_parameter('a_lat', 5.0)           # [m/s^2] cornering grip
        self.declare_parameter('a_brake', 5.4)         # [m/s^2] braking before corners
        self.declare_parameter('a_accel', 4.5)         # [m/s^2] acceleration out of corners
        # Steering
        self.declare_parameter('ld_base', 0.45)         # [m] lookahead = ld_base + ld_gain*speed
        self.declare_parameter('ld_gain', 0.17)
        self.declare_parameter('max_steer', 0.4)    # [rad] car limit
        self.declare_parameter('stop_dist', 0.4)       # [m] obstacle this close ahead -> crawl

        g = lambda n: self.get_parameter(n).value
        self.max_range = g('max_range')
        self.ld_base, self.ld_gain = g('ld_base'), g('ld_gain')
        self.max_steer, self.stop_dist = g('max_steer'), g('stop_dist')
        self.wheelbase = 0.33

        # ---- the line and its speeds (computed once) ----
        self.line = self.make_line(g)
        self.line_speed = raceline.speed_profile(
            self.line, g('v_max'), g('v_min'), g('a_lat'), g('a_brake'), g('a_accel'))
        self.idx = None                # index of the nearest line point
        self.get_logger().info(
            f'{len(self.line)} line points, {raceline.lap_length(self.line):.1f} m, '
            f'speeds {self.line_speed.min():.1f}-{self.line_speed.max():.1f} m/s')

        # Latest ground-truth pose and speed (allowed by the rules)
        self.position = None
        self.yaw = 0.0
        self.speed = 0.0

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, g('drive_topic'), 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/driver/markers', 1)
        self.create_subscription(LaserScan, g('scan_topic'), self.scan_callback, 10)
        self.create_subscription(Odometry, g('odom_topic'), self.odom_callback, 10)
        # When the car is teleported, forget where we were on the line.
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.reset_callback, 10)
        self._marker_divisor = 0

    def make_line(self, g):
        """Racing line from the map; the CSV if one is given; the shipped
        centreline if the map step fails for any reason."""
        if g('line_csv'):
            self.get_logger().info(f"Using line from {g('line_csv')}")
            return raceline.load_csv(g('line_csv'))
        try:
            _, race = raceline.build_raceline(
                g('tracks_yaml'), margin=g('margin'), blend=g('line_blend'),
                log=self.get_logger().info)
            return race
        except Exception as e:  # never fail to drive because of the planner
            self.get_logger().error(f'Racing line failed ({e}); using the centreline CSV')
            return raceline.load_csv('/hackathon/maps/icra26_centerline.csv')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def reset_callback(self, msg):
        self.idx = None
        self.get_logger().info('Car was moved - searching for the line again.')

    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def scan_callback(self, scan):
        ranges, angles = self.preprocess(scan)
        steering, speed = self.plan(ranges, angles)
        self.publish(steering, speed)

        self._marker_divisor = (self._marker_divisor + 1) % 40
        if self._marker_divisor % 10 == 0:
            self.publish_markers(steering, with_line=self._marker_divisor == 0)

    def preprocess(self, scan):
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=self.max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.max_range)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        return ranges, angles

    # ==================================================================
    # Pure Pursuit
    # ==================================================================
    def plan(self, ranges, angles):
        if self.position is None:
            return 0.0, 0.0
        x, y = self.position
        xy, n = self.line, len(self.line)

        # 1. nearest line point (search just around the last one)
        if self.idx is None:
            self.idx = int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))
        else:
            w = (self.idx + np.arange(-5, 40)) % n
            self.idx = int(w[np.argmin(np.hypot(xy[w, 0] - x, xy[w, 1] - y))])
        # lost the line (e.g. the car was moved)? search the whole lap again
        if math.hypot(xy[self.idx, 0] - x, xy[self.idx, 1] - y) > 1.0:
            self.idx = int(np.argmin(np.hypot(xy[:, 0] - x, xy[:, 1] - y)))

        # 2. walk forward along the line to the lookahead point
        ld = self.ld_base + self.ld_gain * self.speed
        j = self.idx
        while math.hypot(xy[j, 0] - x, xy[j, 1] - y) < ld:
            j = (j + 1) % n
            if j == self.idx:
                break

        # 3. steer along the arc through that point
        dx, dy = xy[j, 0] - x, xy[j, 1] - y
        lateral = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy
        steering = math.atan(2.0 * self.wheelbase * lateral / max(dx * dx + dy * dy, 1e-6))
        steering = max(-self.max_steer, min(self.max_steer, steering))

        # 4. speed from the profile, plus a LiDAR safety net
        speed = float(self.line_speed[self.idx])
        ahead = ranges[np.abs(angles) < math.radians(10)]
        if ahead.size and ahead.min() < self.stop_dist:
            speed = min(speed, 0.5)
        return steering, speed

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
