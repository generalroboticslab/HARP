#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace
{
constexpr const char * kExploreFrame = "camera_init";
constexpr float kMinimumAutoColorSpan = 0.10F;

struct GridKey
{
  std::int64_t x;
  std::int64_t y;

  bool operator==(const GridKey & other) const
  {
    return x == other.x && y == other.y;
  }
};

struct GridKeyHash
{
  std::size_t operator()(const GridKey & key) const
  {
    const auto hx = std::hash<std::int64_t>{}(key.x);
    const auto hy = std::hash<std::int64_t>{}(key.y);
    return hx ^ (hy + 0x9e3779b97f4a7c15ULL + (hx << 6U) + (hx >> 2U));
  }
};

struct CellStats
{
  float min_z = std::numeric_limits<float>::infinity();
  float max_z = -std::numeric_limits<float>::infinity();
  std::uint32_t point_count = 0;

  void add(float z)
  {
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
    ++point_count;
  }

  bool valid() const
  {
    return point_count > 0 && std::isfinite(min_z) && std::isfinite(max_z);
  }

  float verticalRange() const
  {
    return valid() ? max_z - min_z : 0.0F;
  }
};

geometry_msgs::msg::Quaternion multiply(
  const geometry_msgs::msg::Quaternion & lhs,
  const geometry_msgs::msg::Quaternion & rhs)
{
  geometry_msgs::msg::Quaternion out;
  out.w = lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z;
  out.x = lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y;
  out.y = lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x;
  out.z = lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w;

  const auto norm = std::sqrt(out.w * out.w + out.x * out.x + out.y * out.y + out.z * out.z);
  if (norm > 0.0) {
    out.w /= norm;
    out.x /= norm;
    out.y /= norm;
    out.z /= norm;
  }
  return out;
}

geometry_msgs::msg::Quaternion pitchRotation(double radians)
{
  geometry_msgs::msg::Quaternion q;
  q.w = std::cos(radians * 0.5);
  q.x = 0.0;
  q.y = std::sin(radians * 0.5);
  q.z = 0.0;
  return q;
}

geometry_msgs::msg::Quaternion yawRotation(double radians)
{
  geometry_msgs::msg::Quaternion q;
  q.w = std::cos(radians * 0.5);
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(radians * 0.5);
  return q;
}

double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

void rotatePitch180(geometry_msgs::msg::Vector3 & vector)
{
  vector.x = -vector.x;
  vector.z = -vector.z;
}

std::int8_t heightToOccupancy(float z, float z_min, float z_max)
{
  const float span = std::max(z_max - z_min, 1.0e-3F);
  const float t = std::clamp((z - z_min) / span, 0.0F, 1.0F);
  return static_cast<std::int8_t>(std::lround(100.0F * t));
}

std::int8_t rangeToOccupancy(float range, float range_max)
{
  const float normalized = std::clamp(range / std::max(range_max, 1.0e-3F), 0.0F, 1.0F);
  return static_cast<std::int8_t>(std::lround(100.0F * normalized));
}

float packedRgb(std::uint8_t r, std::uint8_t g, std::uint8_t b)
{
  const std::uint32_t rgb =
    (static_cast<std::uint32_t>(r) << 16U) |
    (static_cast<std::uint32_t>(g) << 8U) |
    static_cast<std::uint32_t>(b);
  float packed = 0.0F;
  std::memcpy(&packed, &rgb, sizeof(packed));
  return packed;
}

float heightToRgb(float z, float z_min, float z_max)
{
  const float span = std::max(z_max - z_min, 1.0e-3F);
  const float t = std::clamp((z - z_min) / span, 0.0F, 1.0F);
  const auto red = static_cast<std::uint8_t>(255.0F * (1.0F - t));
  const auto green = static_cast<std::uint8_t>(255.0F * (1.0F - std::abs(2.0F * t - 1.0F)));
  const auto blue = static_cast<std::uint8_t>(255.0F * t);
  return packedRgb(red, green, blue);
}
}  // namespace

class LioPostNode : public rclcpp::Node
{
public:
  LioPostNode()
  : Node("lio_post"),
    map_resolution_(declare_parameter<double>("map_resolution", 0.20)),
    map_source_topic_(declare_parameter<std::string>("map_source_topic", "/cloud_registered")),
    use_mavros_reference_(declare_parameter<bool>("use_mavros_reference", true)),
    mavros_pose_topic_(
      declare_parameter<std::string>("mavros_pose_topic", "/explore/mavros/local_position/pose")),
    vegetation_range_max_(declare_parameter<double>("vegetation_range_max", 0.50)),
    update_range_xy_(declare_parameter<double>("update_range_xy", 6.0)),
    pitch_180_(pitchRotation(M_PI)),
    mavros_ref_yaw_quat_(yawRotation(0.0)),
    mavros_ref_x_(0.0),
    mavros_ref_y_(0.0),
    mavros_ref_z_(0.0),
    mavros_ref_yaw_(0.0),
    latest_odom_x_(0.0),
    latest_odom_y_(0.0),
    height_reference_z_(0.0),
    have_mavros_ref_(!use_mavros_reference_),
    have_odom_(false),
    have_height_reference_(false),
    published_first_map_(false)
  {
    if (map_resolution_ <= 0.0) {
      RCLCPP_WARN(
        get_logger(),
        "map_resolution must be positive; using 0.20 m instead of %.3f",
        map_resolution_);
      map_resolution_ = 0.20;
    }
    if (vegetation_range_max_ <= 0.0) {
      RCLCPP_WARN(
        get_logger(),
        "vegetation_range_max must be positive; using 0.50 m instead of %.3f",
        vegetation_range_max_);
      vegetation_range_max_ = 0.50;
    }
    if (update_range_xy_ <= 0.0) {
      RCLCPP_WARN(
        get_logger(),
        "update_range_xy must be positive; using 6.0 m instead of %.3f",
        update_range_xy_);
      update_range_xy_ = 6.0;
    }

    map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      "explore_map",
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    color_map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "explore_map_color",
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("explore_odom", 20);
    traj_marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      "explore_traj_marker",
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    mavros_pose_viz_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "explore_mavros_pose_viz",
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable());

    map_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      map_source_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&LioPostNode::mapCallback, this, std::placeholders::_1));

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/Odometry",
      20,
      std::bind(&LioPostNode::odomCallback, this, std::placeholders::_1));

    if (use_mavros_reference_) {
      mavros_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        mavros_pose_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(&LioPostNode::mavrosPoseCallback, this, std::placeholders::_1));
    }

    RCLCPP_INFO(
      get_logger(),
      "Subscribing to %s and %s; updating map cells within %.2f m XY range",
      map_source_topic_.c_str(),
      odom_sub_->get_topic_name(),
      update_range_xy_);
    RCLCPP_INFO(
      get_logger(),
      "Publishing explore_map OccupancyGrid, explore_map_color PointCloud2, and explore_odom in frame %s",
      kExploreFrame);
    if (use_mavros_reference_) {
      RCLCPP_INFO(
        get_logger(),
        "Waiting for first %s pose to set fixed translation and yaw transform",
        mavros_pose_topic_.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "Using corrected LIO coordinates without a MAVROS reference");
    }
  }

private:
  void mavrosPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    msg->header.frame_id = "camera_init";
    mavros_pose_viz_pub_->publish(*msg);

    if (have_mavros_ref_) {
      return;
    }

    mavros_ref_x_ = msg->pose.position.x;
    mavros_ref_y_ = msg->pose.position.y;
    mavros_ref_z_ = msg->pose.position.z;
    mavros_ref_yaw_ = yawFromQuaternion(msg->pose.orientation);
    mavros_ref_yaw_quat_ = yawRotation(mavros_ref_yaw_);
    have_mavros_ref_ = true;

    RCLCPP_INFO(
      get_logger(),
      "Captured first MAVROS pose reference: xyz=(%.3f, %.3f, %.3f), yaw=%.3f rad",
      mavros_ref_x_,
      mavros_ref_y_,
      mavros_ref_z_,
      mavros_ref_yaw_);
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!have_mavros_ref_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Waiting for first %s before publishing explore_odom",
        mavros_pose_topic_.c_str());
      return;
    }

    auto out = *msg;
    out.header.frame_id = kExploreFrame;
    if (out.child_frame_id.empty()) {
      out.child_frame_id = "body";
    }

    double x = -msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    double z = -msg->pose.pose.position.z;
    applyMavrosReferenceTransform(x, y, z);
    out.pose.pose.position.x = x;
    out.pose.pose.position.y = y;
    out.pose.pose.position.z = z;
    // Left multiply corrects the LIO/world frame. Right multiply corrects the mounted
    // sensor body frame so the reported heading follows the drone body, not the upside-down lidar.
    out.pose.pose.orientation =
      multiply(mavros_ref_yaw_quat_, multiply(multiply(pitch_180_, msg->pose.pose.orientation), pitch_180_));

    rotatePitch180(out.twist.twist.linear);
    rotatePitch180(out.twist.twist.angular);
    rotateYaw(out.twist.twist.linear);
    rotateYaw(out.twist.twist.angular);

    latest_odom_x_ = out.pose.pose.position.x;
    latest_odom_y_ = out.pose.pose.position.y;
    have_odom_ = true;
    if (!have_height_reference_) {
      height_reference_z_ = out.pose.pose.position.z;
      have_height_reference_ = true;
      RCLCPP_INFO(
        get_logger(),
        "Height color reference set to initial corrected odometry z = %.3f m",
        height_reference_z_);
    }

    odom_pub_->publish(out);
    publishTrajectoryMarker(out);
  }

  void mapCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!have_mavros_ref_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Waiting for first %s before updating explore_map",
        mavros_pose_topic_.c_str());
      return;
    }
    if (!have_height_reference_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Waiting for %s before updating explore_map so heights are relative",
        odom_sub_->get_topic_name());
      return;
    }

    cell_map_.reserve(cell_map_.size() + msg->width * msg->height);

    try {
      sensor_msgs::PointCloud2ConstIterator<float> in_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> in_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> in_z(*msg, "z");

      for (; in_x != in_x.end(); ++in_x, ++in_y, ++in_z) {
        double x = -*in_x;
        double y = *in_y;
        double z = -*in_z;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
          continue;
        }
        applyMavrosReferenceTransform(x, y, z);

        const double range_origin_x = have_odom_ ? latest_odom_x_ : 0.0;
        const double range_origin_y = have_odom_ ? latest_odom_y_ : 0.0;
        const double dx = static_cast<double>(x) - range_origin_x;
        const double dy = static_cast<double>(y) - range_origin_y;
        if ((dx * dx + dy * dy) > update_range_xy_ * update_range_xy_) {
          continue;
        }

        const GridKey key{
          static_cast<std::int64_t>(std::floor(x / map_resolution_)),
          static_cast<std::int64_t>(std::floor(y / map_resolution_))};

        cell_map_[key].add(static_cast<float>(z));
      }
    } catch (const std::runtime_error & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Skipping %s message without float x/y/z fields: %s",
        map_source_topic_.c_str(),
        error.what());
      return;
    }

    if (cell_map_.empty()) {
      return;
    }

    float relative_min_z = std::numeric_limits<float>::infinity();
    float relative_max_z = -std::numeric_limits<float>::infinity();
    auto min_x = std::numeric_limits<std::int64_t>::max();
    auto min_y = std::numeric_limits<std::int64_t>::max();
    auto max_x = std::numeric_limits<std::int64_t>::min();
    auto max_y = std::numeric_limits<std::int64_t>::min();
    for (const auto & entry : cell_map_) {
      if (entry.second.valid()) {
        const float relative_height = relativeZ(entry.second.max_z);
        relative_min_z = std::min(relative_min_z, relative_height);
        relative_max_z = std::max(relative_max_z, relative_height);
      }
      min_x = std::min(min_x, entry.first.x);
      min_y = std::min(min_y, entry.first.y);
      max_x = std::max(max_x, entry.first.x);
      max_y = std::max(max_y, entry.first.y);
    }
    expandTinyHeightRange(relative_min_z, relative_max_z);

    const auto width = static_cast<std::uint32_t>(max_x - min_x + 1);
    const auto height = static_cast<std::uint32_t>(max_y - min_y + 1);

    nav_msgs::msg::OccupancyGrid out;
    out.header.stamp = msg->header.stamp;
    out.header.frame_id = kExploreFrame;
    out.info.map_load_time = msg->header.stamp;
    out.info.resolution = static_cast<float>(map_resolution_);
    out.info.width = width;
    out.info.height = height;
    out.info.origin.position.x = static_cast<double>(min_x) * map_resolution_;
    out.info.origin.position.y = static_cast<double>(min_y) * map_resolution_;
    out.info.origin.position.z = 0.0;
    out.info.origin.orientation.w = 1.0;
    out.data.assign(static_cast<std::size_t>(width) * height, -1);

    for (const auto & entry : cell_map_) {
      if (!entry.second.valid()) {
        continue;
      }

      const auto x = static_cast<std::uint32_t>(entry.first.x - min_x);
      const auto y = static_cast<std::uint32_t>(entry.first.y - min_y);
      const auto index = static_cast<std::size_t>(y) * width + x;

      const float relative_height = relativeZ(entry.second.max_z);
      const auto height_score = heightToOccupancy(
        relative_height,
        relative_min_z,
        relative_max_z);
      const auto roughness_score = rangeToOccupancy(
        entry.second.verticalRange(),
        static_cast<float>(vegetation_range_max_));
      out.data[index] = std::max(height_score, roughness_score);
    }

    map_pub_->publish(out);
    publishColorMap(msg->header.stamp, relative_min_z, relative_max_z);
    if (!published_first_map_) {
      published_first_map_ = true;
      RCLCPP_INFO(
        get_logger(),
        "Published first explore_map OccupancyGrid with %u x %u cells from %zu 2.5D stat cells in frame %s",
        out.info.width,
        out.info.height,
        cell_map_.size(),
        out.header.frame_id.c_str());
    }
  }

  void publishColorMap(
    const builtin_interfaces::msg::Time & stamp,
    float relative_min_z,
    float relative_max_z)
  {
    sensor_msgs::msg::PointCloud2 out;
    out.header.stamp = stamp;
    out.header.frame_id = kExploreFrame;
    out.height = 1;
    out.width = static_cast<std::uint32_t>(cell_map_.size());
    out.is_dense = true;

    sensor_msgs::PointCloud2Modifier modifier(out);
    modifier.setPointCloud2Fields(
      4,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "rgb", 1, sensor_msgs::msg::PointField::FLOAT32);
    modifier.resize(cell_map_.size());

    sensor_msgs::PointCloud2Iterator<float> out_x(out, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(out, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(out, "z");
    sensor_msgs::PointCloud2Iterator<float> out_rgb(out, "rgb");

    for (const auto & entry : cell_map_) {
      const auto & stats = entry.second;
      if (!stats.valid()) {
        continue;
      }

      *out_x = static_cast<float>((entry.first.x + 0.5) * map_resolution_);
      *out_y = static_cast<float>((entry.first.y + 0.5) * map_resolution_);
      *out_z = relativeZ(stats.max_z);
      *out_rgb = heightToRgb(
        relativeZ(stats.max_z),
        relative_min_z,
        relative_max_z);
      ++out_x;
      ++out_y;
      ++out_z;
      ++out_rgb;
    }

    color_map_pub_->publish(out);
  }

  void publishTrajectoryMarker(const nav_msgs::msg::Odometry & odom)
  {
    geometry_msgs::msg::Point point;
    point.x = odom.pose.pose.position.x;
    point.y = odom.pose.pose.position.y;
    point.z = odom.pose.pose.position.z;
    trajectory_points_.push_back(point);

    visualization_msgs::msg::Marker marker;
    marker.header = odom.header;
    marker.ns = "lio_post";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05;
    marker.color.r = 0.0F;
    marker.color.g = 1.0F;
    marker.color.b = 1.0F;
    marker.color.a = 1.0F;
    marker.points = trajectory_points_;
    traj_marker_pub_->publish(marker);
  }

  float relativeZ(float z) const
  {
    return z - static_cast<float>(height_reference_z_);
  }

  void applyMavrosReferenceTransform(double & x, double & y, double & z) const
  {
    const double cos_yaw = std::cos(mavros_ref_yaw_);
    const double sin_yaw = std::sin(mavros_ref_yaw_);
    const double rotated_x = cos_yaw * x - sin_yaw * y;
    const double rotated_y = sin_yaw * x + cos_yaw * y;
    x = rotated_x + mavros_ref_x_;
    y = rotated_y + mavros_ref_y_;
    z += mavros_ref_z_;
  }

  void rotateYaw(geometry_msgs::msg::Vector3 & vector) const
  {
    const double cos_yaw = std::cos(mavros_ref_yaw_);
    const double sin_yaw = std::sin(mavros_ref_yaw_);
    const double rotated_x = cos_yaw * vector.x - sin_yaw * vector.y;
    const double rotated_y = sin_yaw * vector.x + cos_yaw * vector.y;
    vector.x = rotated_x;
    vector.y = rotated_y;
  }

  void expandTinyHeightRange(float & min_z, float & max_z) const
  {
    if (!std::isfinite(min_z) || !std::isfinite(max_z)) {
      min_z = -kMinimumAutoColorSpan * 0.5F;
      max_z = kMinimumAutoColorSpan * 0.5F;
      return;
    }

    const float span = max_z - min_z;
    if (span >= kMinimumAutoColorSpan) {
      return;
    }

    const float center = 0.5F * (min_z + max_z);
    min_z = center - kMinimumAutoColorSpan * 0.5F;
    max_z = center + kMinimumAutoColorSpan * 0.5F;
  }

  double map_resolution_;
  std::string map_source_topic_;
  bool use_mavros_reference_;
  std::string mavros_pose_topic_;
  double vegetation_range_max_;
  double update_range_xy_;
  geometry_msgs::msg::Quaternion pitch_180_;
  geometry_msgs::msg::Quaternion mavros_ref_yaw_quat_;
  double mavros_ref_x_;
  double mavros_ref_y_;
  double mavros_ref_z_;
  double mavros_ref_yaw_;
  double latest_odom_x_;
  double latest_odom_y_;
  double height_reference_z_;
  bool have_mavros_ref_;
  bool have_odom_;
  bool have_height_reference_;
  std::unordered_map<GridKey, CellStats, GridKeyHash> cell_map_;
  std::vector<geometry_msgs::msg::Point> trajectory_points_;
  bool published_first_map_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr mavros_pose_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr color_map_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr traj_marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mavros_pose_viz_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LioPostNode>());
  rclcpp::shutdown();
  return 0;
}
