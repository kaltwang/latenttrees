from menpo.shape import PointCloud

def pc_subset(point_cloud, idx_include):
    return PointCloud(point_cloud.points[idx_include,:])