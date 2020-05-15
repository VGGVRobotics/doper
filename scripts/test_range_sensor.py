import time
import numpy as np
from matplotlib import pyplot as plt
from doper.world import Scene, Polygon, UndirectedRangeSensor


if __name__ == "__main__":

    a = Polygon(
        [[[20, 10], [5, 10]], [[20, 10], [20, 20]], [[5, 20], [20, 20]], [[5, 10], [5, 20]]]
    )
    scene = Scene([a])
    sensor = UndirectedRangeSensor(10, 1)
    for i in range(6):
        ranges, points = sensor.get_observation((i * 5, 25), scene, return_intersection_points=True)
        plt.scatter(i * 5, 25, s=100, c="black"),
        for s in a.segments:
            plt.scatter(s[:, 0], s[:, 1], c="red")
        points = points
        plt.scatter(points[:, 0], points[:, 1], c="green")
        plt.show()
    times = []
    scene = Scene([a] * 100)
    for i in range(100):
        start = time.time()
        ranges, points = sensor.get_observation((i * 5, 8), scene, return_intersection_points=True)
        times.append(time.time() - start)
    print(1 / np.median(times))
