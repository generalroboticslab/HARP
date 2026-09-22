"""Save the latest colored 2.5D RViz map and its offline editor."""

from datetime import datetime
from pathlib import Path
import time

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

from mod_explore.map_export import pointcloud_xyzrgb, write_pcd, generate_editor


class MapRecorder(Node):
    def __init__(self):
        super().__init__('map_recorder')
        output_dir = self.declare_parameter('output_dir', 'maps').value
        self.topic = self.declare_parameter('cloud_topic', '/explore_map_color').value
        self.cell_size = float(self.declare_parameter('cell_size', 0.2).value)
        self.idle_seconds = float(self.declare_parameter('idle_seconds', 3.0).value)
        if self.idle_seconds < 0:
            raise ValueError('idle_seconds must be nonnegative (0 disables idle export)')
        if self.cell_size <= 0:
            raise ValueError('cell_size must be positive')
        session = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.directory = Path(output_dir).expanduser().resolve() / session
        self.pcd_path = self.directory / 'map.pcd'
        self.html_path = self.directory / 'map_editor.html'
        self.points = None
        self.revision = 0
        self.saved_revision = -1
        self.frame_id = None
        self.last_cloud = None
        self.last_attempt = 0.0
        # Request the last complete map too if joining after bag playback.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subscription = self.create_subscription(
            PointCloud2, self.topic, self.receive_cloud, qos)
        self.service = self.create_service(Trigger, 'save_map_editor', self.save_service)
        # Idle export must also work while a bag's /clock is paused.
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(1.0, self.check_idle, clock=self.steady_clock)
        self.get_logger().info(
            f'Saving the latest colored 2.5D map from {self.topic}. '
            f'PCD and HTML output: {self.directory}. '
            'Export on input idle, shutdown, or /save_map_editor.')

    def receive_cloud(self, msg):
        if self.frame_id is not None and self.frame_id != msg.header.frame_id:
            self.get_logger().error('Cloud frame changed; refusing to mix coordinate frames')
            return
        try:
            points = pointcloud_xyzrgb(msg)
            if not len(points):
                return
            # Each message already contains the entire map. Appending would
            # duplicate cells and keep stale heights and colors.
            self.points = points
            self.frame_id = msg.header.frame_id
            self.revision += 1
            self.last_cloud = time.monotonic()
        except ValueError as error:
            self.get_logger().error(f'Could not read the colored 2.5D map: {error}')

    def check_idle(self):
        now = time.monotonic()
        if (self.idle_seconds > 0 and self.last_cloud is not None
                and self.revision != self.saved_revision
                and now - self.last_cloud >= self.idle_seconds
                and now - self.last_attempt >= max(5.0, self.idle_seconds)):
            self.save()

    def save(self):
        if self.points is None:
            return False, f'No colored 2.5D map received on {self.topic} yet.'
        if self.saved_revision == self.revision:
            return True, f'Already saved: {self.html_path}'
        self.last_attempt = time.monotonic()
        try:
            write_pcd(self.pcd_path, self.points, self.frame_id, color_field='rgb',
                      cell_size=self.cell_size)
            generate_editor(self.pcd_path, self.html_path, self.frame_id,
                            cell_size=self.cell_size, source_topic=self.topic)
            self.saved_revision = self.revision
            message = (f'Saved {len(self.points):,} colored 2.5D cells: {self.pcd_path}\n'
                       f'Open the editor: {self.html_path}')
            if self.context.ok():
                self.get_logger().info(message)
            else:
                print(message, flush=True)
            return True, message
        except (OSError, ValueError) as error:
            message = f'Map export failed: {error}'
            if self.context.ok():
                self.get_logger().error(message)
            else:
                print(message, flush=True)
            return False, message

    def save_service(self, request, response):
        response.success, response.message = self.save()
        return response

    def finish(self):
        self.save()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MapRecorder()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.finish()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
