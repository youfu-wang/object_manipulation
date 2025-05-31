import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from mecheye_ros_interface.srv import CapturePointCloud
import open3d as o3d
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header  

class PointCloudSaver(Node):
    @staticmethod
    def convert_r2o(ros_cloud: PointCloud2, skip_nans: bool = True) -> o3d.geometry.PointCloud:
        """
        Turn ROS2 Pointcloud2 msg into open3d data.
        """
        field_names = [field.name for field in ros_cloud.fields]

        cloud_data = list(
            pc2.read_points(ros_cloud,
                            field_names=field_names,
                            skip_nans=skip_nans)
        )

        # No color and only x, y, z
        points = [(x, y, z) for (x, y, z, *_) in cloud_data]
        cloud_arr = np.array(points, dtype=np.float32)

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud_arr)

        return o3d_cloud

    @staticmethod
    def scene_cut_aabb(pcd):
        x_range = (-0.06, 0.6)
        y_range = (-0.4, 0.22)
        z_range = (1, 1.35)

        # create a box area and keep the points in this area
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(x_range[0], y_range[0], z_range[0]),
            max_bound=(x_range[1], y_range[1], z_range[1])
        )
        cropped_pcd = pcd.crop(bbox)

        return cropped_pcd

    def _o3d_to_pointcloud2(self, cloud: o3d.geometry.PointCloud) -> PointCloud2:
        """
        Turn open3d data into ROS2 Pointcloud2 msg.
        """
        pts = cloud.points
        if len(pts) == 0:
            self.get_logger().warn("Aligned point cloud is empty.")
            return PointCloud2()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "mechmind_camera/textured_point_cloud"
        pc2_msg = pc2.create_cloud_xyz32(header, [tuple(p) for p in pts])
        
        return pc2_msg

    def __init__(self):
        super().__init__('point_cloud_saver')
        self.point_cloud_received = False
        self.ros_pc2 = None

        # create Server Client
        self.client = self.create_client(CapturePointCloud, '/capture_point_cloud')
        while not self.client.wait_for_service(timeout_sec=0.5):
            self.get_logger().info('Wait for /capture_point_cloud service available...')

        # create topic subscriber
        self.subscription = self.create_subscription(
            PointCloud2,
            '/mechmind/point_cloud',
            self.point_cloud_callback,
            10)

        self.publisher = self.create_publisher(PointCloud2, '/processed_scene', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

        # send request to Mech Eye Server
        self.call_capture_service()

    def call_capture_service(self):
        request = CapturePointCloud.Request()
        future = self.client.call_async(request)
        future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        try:
            future.result()
            self.get_logger().info('succeed to ask /capture_point_cloud server')
        except Exception as e:
            self.get_logger().error(f'fail to connect server: {e}')

    def timer_callback(self):
        if self.ros_pc2 is not None:
            self.publisher.publish(self.ros_pc2)

    def point_cloud_callback(self, msg: PointCloud2):
        if self.point_cloud_received:
            return
        self.point_cloud_received = True

        # ---------- ROS2 → Open3D ----------
        o3d_pc = PointCloudSaver.convert_r2o(msg, skip_nans=True)
        if len(o3d_pc.points) == 0:
            self.get_logger().warn("No points in incoming cloud!")
            return

        # point clouds processing
        cropped = PointCloudSaver.scene_cut_aabb(o3d_pc)
        if len(cropped.points) == 0:
            self.get_logger().warn("No points in cropped cloud!")
            return

        # save as pcd.file
        o3d.io.write_point_cloud("21052025.pcd", cropped)

        #self.saved_cloud = self._o3d_to_pointcloud2(cropped)

        # ---------- Open3D → ROS2 ----------
        self.ros_pc2 = self._o3d_to_pointcloud2(cropped)
        #self.publisher.publish(self.ros_pc2)
        self.get_logger().info(
            f'Received {len(o3d_pc.points)} pts, '
            f'cropped {len(cropped.points)} pts, published scene.')



def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
