import time
import numpy as np
from matplotlib import pyplot as plt
from doper.world import Scene, Rasterizer, Polygon


if __name__ == "__main__":

    a = Polygon(
        [[[5, 10], [20, 10]], [[20, 10], [20, 20]], [[5, 20], [20, 20]], [[5, 20], [5, 10]]]
    )
    scene = Scene([a] * 10)
    rasterizer = Rasterizer()
    plt.imshow(rasterizer.rasterize(scene, (10, 8)))
    plt.show()
    times = []
    for i in range(20):
        start = time.time()
        rasterizer.rasterize(scene, (10, 10))
        times.append(time.time() - start)
    print(np.median(times))
