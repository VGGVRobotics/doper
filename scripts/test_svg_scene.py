import numpy as np
from doper.utils.assets import get_svg_scene
from doper.world.visualizer import TaichiRenderer
from doper.agents.observations import UndirectedRangeSensor
from time import time


if __name__ == "__main__":
    scene = get_svg_scene("assets/simple_level.svg", px_per_meter=50)
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
        start = time()
        ranges, points = sensor.get_observation(sensor_pos, scene, True)
        print(time() - start)
        gui.render(sensor_pos=sensor_pos, ray_intersection_points=points)
