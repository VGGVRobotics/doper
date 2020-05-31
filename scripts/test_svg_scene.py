import numpy as np
from doper.world.assets import get_svg_scene
from doper.world.visualizer import TaichiRenderer
from doper.world.observations import UndirectedRangeSensor
from time import time


if __name__ == "__main__":
    scene = get_svg_scene("../assets/simple_level.svg", px_per_meter=50)
    gui = TaichiRenderer(scene=scene, px_per_meter=50, window_size_meters=(12, 9))
    sensor = UndirectedRangeSensor(distance_range=3.5, angle_step=1)
    sensor_pos = np.array((1.0, 1.0))
    while True:
        gui.ti_gui.get_event()
        if gui.ti_gui.event is not None:
            if gui.ti_gui.event.key == "a":
                sensor_pos[0] -= 0.01
            elif gui.ti_gui.event.key == "d":
                sensor_pos[0] += 0.01
            elif gui.ti_gui.event.key == "w":
                sensor_pos[1] += 0.01
            elif gui.ti_gui.event.key == "s":
                sensor_pos[1] -= 0.01
        ranges, points = sensor.get_observation(sensor_pos, scene, True)
        start = time()
        result = scene.get_closest_geometry(sensor_pos)
        print(time() - start)
        print(scene.get_polygons_in_radius(sensor_pos, 3.5))
        print(scene.get_closest_geometry(sensor_pos))
        gui.render(sensor_pos=sensor_pos, ray_intersection_points=points)
