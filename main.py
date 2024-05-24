from CubicStylization import *
from Visualization import *


if __name__ == '__main__':
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    file = "./example/bunny_10k.obj"
    cube = CubeStylization(mesh_file=file)
    visualize_mesh(cube.V, cube.F, title="initial mesh")

    cube.Iteration()
    visualize_mesh(cube.new_v, cube.F, title="cubic stylization result")
