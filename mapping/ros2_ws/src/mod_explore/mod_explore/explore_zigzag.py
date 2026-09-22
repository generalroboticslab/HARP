#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


def yaw_from_quat(x, y, z, w):
	s1 = 2.0 * (w * z + x * y)
	s2 = 1.0 - 2.0 * (y * y + z * z)
	return math.atan2(s1, s2)


def quat_from_yaw(yaw):
	h = 0.5 * yaw
	return (0.0, 0.0, math.sin(h), math.cos(h))


def body_to_world(dx, dy, yaw):
	c, s = math.cos(yaw), math.sin(yaw)
	return dx * c - dy * s, dx * s + dy * c


class OffboardZigzag(Node):
	"""BODY-frame zigzag path anchored at the vehicle's first local pose."""

	def __init__(self):
		super().__init__("x500_zigzag_node")

		# -------- Parameters --------
		self.declare_parameter('ns', 'mavros')
		self.declare_parameter('rate_hz', 20.0)
		self.declare_parameter('takeoff_alt', 4.0)
		self.declare_parameter('xy_speed', 1.5)
		self.declare_parameter('warmup_sec', 2.0)
		self.declare_parameter('alt_tol', 0.3)
		self.declare_parameter('lock_yaw', True)
		self.declare_parameter('yaw_offset_deg', 0.0)
		self.declare_parameter('reverse_x', False)
		self.declare_parameter('reverse_y', False)
		self.declare_parameter('zigzag.length_m', 20.0)
		self.declare_parameter('zigzag.width_m', 10.0)
		self.declare_parameter('zigzag.resolution_m', 2.0)

		ns = self.get_parameter('ns').get_parameter_value().string_value
		self.ns = ns if ns.startswith('/') else '/' + ns

		self.rate = float(self.get_parameter('rate_hz').value)
		self.dt = 1.0 / self.rate
		self.takeoff_alt = float(self.get_parameter('takeoff_alt').value)
		self.xy_speed = float(self.get_parameter('xy_speed').value)
		self.dt_speed = self.xy_speed * self.dt
		self.warmup_ticks = int(float(self.get_parameter('warmup_sec').value) * self.rate)
		self.alt_tol = float(self.get_parameter('alt_tol').value)
		self.lock_yaw = bool(self.get_parameter('lock_yaw').value)
		self.yaw_bias = math.radians(float(self.get_parameter('yaw_offset_deg').value))
		self.reverse_x = bool(self.get_parameter('reverse_x').value)
		self.reverse_y = bool(self.get_parameter('reverse_y').value)

		self.zigzag_length = max(0.1, abs(float(self.get_parameter('zigzag.length_m').value)))
		self.zigzag_width = max(0.0, abs(float(self.get_parameter('zigzag.width_m').value)))
		self.zigzag_resolution = max(0.1, abs(float(self.get_parameter('zigzag.resolution_m').value)))

		# -------- ROS I/O --------
		qos = QoSProfile(depth=10)
		qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
		self.state = State()
		self.state_sub = self.create_subscription(State, f'{self.ns}/state', self._on_state, qos)
		self.pose_sub = self.create_subscription(PoseStamped, f'{self.ns}/local_position/pose', self._on_pose, qos)
		self.sp_pub = self.create_publisher(PoseStamped, f'{self.ns}/setpoint_position/local', 10)
		self.srv_mode = self.create_client(SetMode, f'{self.ns}/set_mode')
		self.srv_arm = self.create_client(CommandBool, f'{self.ns}/cmd/arming')

		# -------- Internal state --------
		self.has_home = False
		self.home_x = self.home_y = self.home_z = 0.0
		self.init_yaw = 0.0
		self.cmd_x = self.cmd_y = None
		self.last_pose = None

		self.zigzag_waypoints = self._build_zigzag_waypoints()
		self.current_waypoint = 1
		self.current_x = 0.0
		self.current_y = 0.0
		self.waypoint_pause_start = 0.0

		self.stage = 'WARMUP'
		self.ticks = 0
		self.timer = self.create_timer(self.dt, self._tick)

		reverse_info = f"mirror_x={'on' if self.reverse_x else 'off'}, reverse_y_dir={'neg' if self.reverse_y else 'pos'}"
		self.get_logger().info(
			f"[zigzag] v={self.xy_speed} m/s, length={self.zigzag_length} m, "
			f"width={self.zigzag_width} m, resolution={self.zigzag_resolution} m, {reverse_info}"
		)

	# ------------ Callbacks ------------
	def _on_state(self, msg: State):
		self.state = msg

	def _on_pose(self, msg: PoseStamped):
		self.last_pose = msg
		if not self.has_home:
			p = msg.pose.position
			o = msg.pose.orientation
			self.home_x, self.home_y, self.home_z = p.x, p.y, p.z
			self.init_yaw = yaw_from_quat(o.x, o.y, o.z, o.w) + self.yaw_bias
			self.cmd_x, self.cmd_y = p.x, p.y
			self.has_home = True
			self.get_logger().info(
				f"home=({p.x:.2f},{p.y:.2f},{p.z:.2f}), yaw0={math.degrees(self.init_yaw):.1f} deg"
			)

	# ------------ Helpers ------------
	def _try_set_mode(self, mode: str):
		if not self.srv_mode.service_is_ready():
			return
		req = SetMode.Request()
		req.custom_mode = mode
		self.srv_mode.call_async(req)

	def _try_arm(self):
		if not self.srv_arm.service_is_ready():
			return
		req = CommandBool.Request()
		req.value = True
		self.srv_arm.call_async(req)

	def _build_zigzag_waypoints(self):
		"""Build a rectangular body-frame zigzag that advances along the x-axis."""
		length = -self.zigzag_length if self.reverse_x else self.zigzag_length
		x_step = self.zigzag_resolution if length >= 0.0 else -self.zigzag_resolution
		y_sign = -1.0 if self.reverse_y else 1.0
		y_amp = 0.5 * self.zigzag_width
		num_columns = int(math.ceil(abs(length) / self.zigzag_resolution))

		waypoints = [(0.0, 0.0)]
		for column in range(num_columns + 1):
			x = x_step * column
			if abs(x) > abs(length):
				x = length

			if column % 2 == 0:
				waypoints.append((x, y_sign * y_amp))
				waypoints.append((x, -y_sign * y_amp))
			else:
				waypoints.append((x, -y_sign * y_amp))
				waypoints.append((x, y_sign * y_amp))

		return waypoints

	def _zigzag_setpoint(self):
		if self.current_waypoint >= len(self.zigzag_waypoints):
			gx, gy = body_to_world(self.current_x, self.current_y, self.init_yaw)
			return self.home_x + gx, self.home_y + gy

		target_x, target_y = self.zigzag_waypoints[self.current_waypoint]
		dx = target_x - self.current_x
		dy = target_y - self.current_y
		d2 = dx * dx + dy * dy

		if d2 < 0.25:
			t = self.get_clock().now().nanoseconds / 1e9
			if self.waypoint_pause_start == 0.0:
				self.waypoint_pause_start = t
				self.get_logger().info(f"WP {self.current_waypoint} pause")
			elif t - self.waypoint_pause_start >= 1.0:
				self.current_waypoint += 1
				self.waypoint_pause_start = 0.0
		elif d2 > 1e-12:
			d = math.sqrt(d2)
			step = min(self.dt_speed, d)
			self.current_x += dx / d * step
			self.current_y += dy / d * step

		gx, gy = body_to_world(self.current_x, self.current_y, self.init_yaw)
		return self.home_x + gx, self.home_y + gy

	# ------------ Main ------------
	def _tick(self):
		if not self.state.connected or not self.has_home:
			return

		if self.stage == 'WARMUP':
			self.ticks += 1
			if self.ticks >= self.warmup_ticks:
				self._try_set_mode("OFFBOARD")
				self.stage = 'OFFBOARD'
				self.get_logger().info("-> OFFBOARD (request)")

		elif self.stage == 'OFFBOARD':
			if self.state.mode == "OFFBOARD":
				self._try_arm()
				self.stage = 'ARM'
				self.get_logger().info("OFFBOARD set; -> ARM (request)")

		elif self.stage == 'ARM':
			if self.state.armed:
				self.stage = 'ASCEND'
				self.get_logger().info("armed; -> ASCEND")

		elif self.stage == 'ASCEND':
			self.cmd_x = self.home_x
			self.cmd_y = self.home_y
			target_z = self.home_z + self.takeoff_alt
			if self.last_pose is not None:
				z = self.last_pose.pose.position.z
				if abs(z - target_z) <= self.alt_tol:
					self.stage = 'TRACK'
					self.get_logger().info("Reached takeoff altitude; -> TRACK")

		elif self.stage == 'TRACK':
			self.cmd_x, self.cmd_y = self._zigzag_setpoint()
			if self.current_waypoint >= len(self.zigzag_waypoints):
				self.stage = 'HOLD'
				self.get_logger().info("[zigzag] completed trajectory -> HOLD")

		elif self.stage == 'HOLD':
			pass

		sp = PoseStamped()
		sp.header.stamp = self.get_clock().now().to_msg()
		sp.header.frame_id = "map"
		sp.pose.position.x = self.cmd_x if self.cmd_x is not None else self.home_x
		sp.pose.position.y = self.cmd_y if self.cmd_y is not None else self.home_y
		sp.pose.position.z = self.home_z + self.takeoff_alt

		if self.lock_yaw:
			current_yaw = self.init_yaw
		elif self.last_pose is not None:
			o = self.last_pose.pose.orientation
			current_yaw = yaw_from_quat(o.x, o.y, o.z, o.w)
		else:
			current_yaw = self.init_yaw

		qx, qy, qz, qw = quat_from_yaw(current_yaw)
		sp.pose.orientation.x = qx
		sp.pose.orientation.y = qy
		sp.pose.orientation.z = qz
		sp.pose.orientation.w = qw
		self.sp_pub.publish(sp)


def main():
	rclpy.init()
	node = OffboardZigzag()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
	main()
