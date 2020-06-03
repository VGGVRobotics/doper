from typing import Tuple, Union
import taichi as ti
import numpy as np

from .scene import Scene


class TaichiRenderer:
    def __init__(
        self,
        scene: Scene,
        window_size_meters: Tuple[int, int] = (12, 9),
        frustrum_left_corner: Tuple[float, float] = (0, 0),
        px_per_meter: float = 50,
        name: str = "Taichi",
    ) -> None:
        """Class for scene visualisation

        Args:
            scene (Scene): current scene representation
            window_size_meters (Tuple[int, int], optional): size of the view frustrum in meters.
                Defaults to (12, 9).
            frustrum_left_corner (Tuple[float, float], optional): left lower world coordinates of
                view frustrum. Defaults to (0, 0).
            px_per_meter (float, optional): pixels per meter. Defaults to 50.
            name (str, optional): window name. Defaults to "Taichi".
        """
        self._scene = scene
        self._gui = ti.GUI(name, res=[s * px_per_meter for s in window_size_meters])
        self._window_size_meters = np.array(window_size_meters)
        self._frustrum_left_corner = frustrum_left_corner
        self._px_per_meter = px_per_meter

    def _render_scene(self) -> None:
        """Renders scene geometry
        """
        for poly in self._scene.get_polygons_in_area(
            self._frustrum_left_corner, self._frustrum_left_corner + self._window_size_meters
        ):
            for i in range(len(poly.segments)):
                segment = poly.segments[i].copy()
                normal = poly.normals[i].copy()
                segment = segment / self._window_size_meters
                normal = normal / self._window_size_meters
                midpoint = (segment[1] + segment[0]) / 2
                self._gui.line(segment[1], segment[0], color=0xFF00FF)
                self._gui.line(midpoint, midpoint + normal)

    def _render_sensor(
        self,
        sensor_pos: Union[Tuple[int, int], np.ndarray],
        ray_intersection_points: np.ndarray,
        sensor_heading: Union[Tuple[float, float], np.ndarray, None],
    ):
        """Renders sensor position, direction and observation.

        Args:
            sensor_pos (Union[Tuple[int, int], np.ndarray]): world coordinates of the sensor.
            ray_intersection_points (np.ndarray): [n_rays, 2] sensor ray intersections with geometry
            sensor_heading (Union[Tuple[float, float], np.ndarray, None]): heading vector
        """
        sensor_pos = np.array(sensor_pos)
        sensor_pos = sensor_pos / self._window_size_meters
        ray_intersection_points = ray_intersection_points / self._window_size_meters
        self._gui.circle(sensor_pos, color=0xFFFF00, radius=self._px_per_meter * 0.2)
        self._gui.circles(ray_intersection_points, color=0x008888, radius=self._px_per_meter * 0.05)
        if sensor_heading is not None:
            sensor_heading = sensor_heading / self._window_size_meters
            sensor_heading = np.array(sensor_heading)
            self._gui.line(sensor_pos, sensor_pos + sensor_heading, color=0xFF0000, radius=1)

    def render(
        self,
        sensor_pos: Union[Tuple[int, int], np.ndarray],
        ray_intersection_points: np.ndarray,
        sensor_heading: Union[Tuple[float, float], np.ndarray, None] = None,
    ) -> None:
        """Renders one frame of simulation

        Args:
            sensor_pos (Union[Tuple[int, int], np.ndarray]):  world coordinates of the sensor.
            ray_intersection_points (np.ndarray): [n_rays, 2] sensor ray intersections with geometry
            sensor_heading (Union[Tuple[float, float], np.ndarray, None], optional): heading vector.
                Defaults to None.
        """

        self._render_scene()
        self._render_sensor(sensor_pos, ray_intersection_points, sensor_heading)
        self._gui.show()

    @property
    def ti_gui(self) -> ti.GUI:
        """ti.GUI: handle to taichi gui backend
        """
        return self._gui
