import open3d as o3d
import numpy as np
import torch

def compute_centroid(point_cloud):
    points = np.asarray(point_cloud.points)
    centroid = np.mean(points, axis=0)
    return centroid

# Load point cloud
point_cloud = o3d.io.read_point_cloud("/home/efedele/Datasets/replica/room0/room0_mesh.ply")

# Load instance annotations (assuming annotations are available in a suitable format)
masks = torch.load("/home/efedele/Programming/original_repos/openmask3d/output/2024-05-03-12-18-28-experiment/room0_mesh_masks_db_0.5.pt")  # Modify as per the annotation format

# Process each segment
for idx, mask in enumerate(masks):
    # Extract points belonging to the current segment
    segment_points = np.asarray(point_cloud.points)[mask == True]
    
    # Create a new point cloud for the segment
    segment_cloud = o3d.geometry.PointCloud()
    segment_cloud.points = o3d.utility.Vector3dVector(segment_points)
    
    # Compute the centroid of the segment
    centroid = compute_centroid(segment_cloud)
    
    # Visualize and save the segment
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(segment_cloud)
    
    # Set up camera view to center on the object's centroid
    ctr = vis.get_view_control()
    ctr.set_lookat(centroid)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)
    
    # Capture and save image
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"segment_{idx}.png")
    vis.destroy_window()
