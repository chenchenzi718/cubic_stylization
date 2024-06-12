import open3d as o3d


# 使用 Open3D 可视化结果
def visualize_mesh(vertices, faces, title="Mesh"):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(title)
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()
