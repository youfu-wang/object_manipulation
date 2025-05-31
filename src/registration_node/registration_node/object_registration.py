import copy
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from tf2_ros import (
    Buffer, TransformListener, TransformBroadcaster,
    LookupException, ConnectivityException, TimeoutException,
)
import tf2_geometry_msgs

from geometry_msgs.msg import Pose, TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from pose_publisher_py2.test_reg import PointCloudRegistration
import open3d as o3d
from transforms3d.quaternions import mat2quat


class PointCloudRegistrationNode(Node):
    def __init__(self):
        super().__init__('pointcloud_registration_node')
        self.get_logger().info("Starting pointcloud_registration_node…")

        # ---------- TF2 ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ---------- Frames ----------
        self.camera_frame = 'mechmind_camera/textured_point_cloud'
        self.tool_frame   = 'base_link'
        self.object_frame = 'object_pred'

        # ---------- Publishers ----------
        self.pose_pub = self.create_publisher(
            Pose, '/pose_in_base_link', 10)

        self.cloud_pub = self.create_publisher(
            PointCloud2, '/registered_object_cloud', 1)

        # ---------- Subscribed target cloud ----------
        self.target_topic = "/processed_scene"
        self.target_pcd: o3d.geometry.PointCloud | None = None
        self.subscriber = self.create_subscription(
            PointCloud2,
            self.target_topic,
            self.target_cloud_callback,
            10
        )

        # ---------- Source PCD file ----------
        self.source_file = "source_transformed_2205_2.pcd"

        # ---------- Runtime buffers ----------
        self.pose_in_camera: Pose | None = None
        self.cloud_msg: PointCloud2 | None = None

        # ---------- State ----------
        self.registration_done = False
        self.publish_timer = None

    # ---------- TF helper ----------
    def get_transform(self, target_frame, source_frame, timeout_sec: float = 0.2):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, Time(),
                timeout=Duration(seconds=timeout_sec))
        except (LookupException, ConnectivityException, TimeoutException):
            return None

    # ---------- util ----------
    @staticmethod
    def matrix_to_pose(T):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (
            float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        w, x, y, z = mat2quat(T[:3, :3])
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = x, y, z, w
        return pose

    # ---------- subscriber callback ----------
    def target_cloud_callback(self, msg: PointCloud2):
        """
        Receive the target scene cloud once, convert to Open3D, then run registration.
        Subsequent messages are ignored.
        """
        if self.registration_done:
            return

        self.target_pcd = PointCloudRegistrationNode.convert_r2o(msg)
        if self.target_pcd.is_empty():
            self.get_logger().warn("Received empty target cloud — waiting for next message.")
            return

        self.get_logger().info(f"Target cloud received from '{self.target_topic}', running registration…")
        self.run_registration_once()

    # ---------- one-off registration ----------
    def run_registration_once(self):
        if self.registration_done:
            return

        # --- read source PCD ---
        try:
            src_pcd = o3d.io.read_point_cloud(self.source_file)
        except Exception as e:
            self.get_logger().error(f"Failed to read source PCD file: {e}")
            return
        if src_pcd.is_empty():
            self.get_logger().error("Source PCD is empty!")
            return

        # --- target PCD came from subscriber ---
        if self.target_pcd is None:
            self.get_logger().warn("Target PCD not yet available — aborting registration.")
            return
        tgt_pcd = self.target_pcd

        # --- full registration ---
        reg = PointCloudRegistration(voxel_size=0.001)
        result = reg.register(src_pcd, tgt_pcd)
        T_cam_obj = result.transformation
        if T_cam_obj is None:
            self.get_logger().error("Registration failed: no transformation.")
            return

        # ---------- Pose in camera frame ----------
        self.pose_in_camera = self.matrix_to_pose(T_cam_obj)

        # ---------- Build registered cloud ----------
        aligned_pcd = copy.deepcopy(src_pcd)
        aligned_pcd.transform(T_cam_obj)
        self.cloud_msg = self._o3d_to_pointcloud2(aligned_pcd)

        self.registration_done = True
        self.get_logger().info("Registration done — start periodic publishing.")
        self.publish_timer = self.create_timer(0.2, self.publish_loop)

    # ---------- periodic publish ----------
    def publish_loop(self):
        if self.pose_in_camera is None:
            return

        tf_cam_to_tool = self.get_transform(self.tool_frame, self.camera_frame)
        if tf_cam_to_tool is None:
            self.get_logger().warn("TF base_link←camera unavailable — skip this cycle.")
            return

        # Pose in tool0
        pose_in_base_link = tf2_geometry_msgs.do_transform_pose(
            self.pose_in_camera, tf_cam_to_tool)

        # ---------- publish Pose msg ----------
        self.pose_pub.publish(pose_in_base_link)

        # ---------- broadcast TF (tool0 → object_pred) ----------
        ts = TransformStamped()
        ts.header.stamp    = self.get_clock().now().to_msg()
        ts.header.frame_id = self.tool_frame
        ts.child_frame_id  = self.object_frame
        ts.transform.translation.x = pose_in_base_link.position.x
        ts.transform.translation.y = pose_in_base_link.position.y
        ts.transform.translation.z = pose_in_base_link.position.z
        ts.transform.rotation      = pose_in_base_link.orientation
        self.tf_broadcaster.sendTransform(ts)

        # ---------- publish registered cloud ----------
        if self.cloud_msg is not None:
            self.cloud_msg.header.stamp = self.get_clock().now().to_msg()
            self.cloud_pub.publish(self.cloud_msg)

        # ---------- log ----------
        p, q = pose_in_base_link.position, pose_in_base_link.orientation
        self.get_logger().info(
            f"Object pose in '{self.tool_frame}': "
            f"P=({p.x:.3f},{p.y:.3f},{p.z:.3f}) "
            f"Q=({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f})"
        )

    # ---------- Open3D → PointCloud2 ----------
    def _o3d_to_pointcloud2(self, cloud: o3d.geometry.PointCloud) -> PointCloud2:
        pts = cloud.points
        if len(pts) == 0:
            self.get_logger().warn("Aligned point cloud is empty.")
            return PointCloud2()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.camera_frame
        pc2_msg = point_cloud2.create_cloud_xyz32(header, [tuple(p) for p in pts])
        
        return pc2_msg

    @staticmethod
    def convert_r2o(ros_cloud: PointCloud2, skip_nans: bool = True) -> o3d.geometry.PointCloud:
        """
        Turn ROS2 Pointcloud2 msg into open3d data.
        """
        field_names = [field.name for field in ros_cloud.fields]

        cloud_data = list(
            point_cloud2.read_points(ros_cloud,
                            field_names=field_names,
                            skip_nans=skip_nans)
        )

        # No color and only x, y, z
        points = [(x, y, z) for (x, y, z, *_) in cloud_data]
        cloud_arr = np.array(points, dtype=np.float32)

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud_arr)

        return o3d_cloud


# ---------- entry ----------

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudRegistrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
