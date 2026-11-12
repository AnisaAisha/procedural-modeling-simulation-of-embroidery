"""Orbit camera around CAM_TARGET."""
import math

import taichi as ti

import constants as C

cam_pos = ti.Vector.field(3, dtype=ti.f32, shape=())


class OrbitCamera:
    def __init__(self):
        self.radius = C.CAM_RADIUS
        self.reset()

    def reset(self):
        # angled top-side view
        self.yaw = C.CAM_YAW
        self.pitch = C.CAM_PITCH
        self.update()

    def orbit(self, dx, dy):
        self.yaw -= dx * C.ORBIT_SPEED
        self.pitch = max(-C.PITCH_LIMIT, min(C.PITCH_LIMIT, self.pitch + dy * C.ORBIT_SPEED))
        self.update()

    def zoom(self, delta):
        self.radius = max(C.MIN_CAM_RADIUS, min(C.MAX_CAM_RADIUS, self.radius + delta))
        self.update()

    def update(self):
        cp = math.cos(self.pitch)
        offset = (self.radius * cp * math.sin(self.yaw),
                  self.radius * math.sin(self.pitch),
                  self.radius * cp * math.cos(self.yaw))
        cam_pos[None] = [C.CAM_TARGET[i] + offset[i] for i in range(3)]
