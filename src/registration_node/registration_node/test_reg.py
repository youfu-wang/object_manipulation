"""
This
"""
import open3d as o3d
import numpy as np

class PointCloudRegistration:

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], 
                             [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def preprocess_point_cloud(self, pcd):
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        radius_normal = self.voxel_size * 3.653
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        
        radius_feature = self.voxel_size * 14.46
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return pcd_down, pcd_fpfh

    def prepare_dataset(self, source, target):
        source.transform(self.trans_init)
        source_down, source_fpfh = self.preprocess_point_cloud(source)
        target_down, target_fpfh = self.preprocess_point_cloud(target)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        distance_threshold = self.voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, 
            target_down, 
            source_fpfh, 
            target_fpfh, 
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], 
            o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
        )
        return result

    def refine_registration(self, source, target, initial_transform):
        distance_threshold = self.voxel_size * 0.859

        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        return result_icp

    def compute_inlier_metrics(self, source, target, transformation, max_distance):
        """
        caculate "inlier Ratio" and "RMSE"
        """
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, 
            target, 
            max_distance,
            transformation
        )
        return evaluation.fitness, evaluation.inlier_rmse

    def register(self, source, target):
        # Data needed 
        source_full, target_full, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(source, target)

        # use RANSAC to get original transformation
        result_ransac = self.execute_global_registration(source_down, target_down, source_fpfh, target_fpfh)
        #print("Global registration result:")
        #print(result_ransac.transformation)

        # use ICP
        result_icp = self.refine_registration(source_full, target_full, result_ransac.transformation)
        print("Refined registration result:")
        print(result_icp.transformation)

        fitness_fine, rmse_fine = self.compute_inlier_metrics(source, target, result_icp.transformation, self.voxel_size * 2)
        print(f"number of target's point after voxel_down: {len(target_down.points)}")
        print(f"Correspondences of ICP: {len(np.asarray(result_icp.correspondence_set))}")
        print(f"HOW IS THE RESULT?: Inlier Ratio={fitness_fine*100:.2f}%, RMSE={rmse_fine:.6f}")

        return result_icp
