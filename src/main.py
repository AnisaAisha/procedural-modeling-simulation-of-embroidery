"""Satin stitch simulation. 

Controls: drag = orbit, W / S = zoom, T = reset view, Esc = quit.
"""
import taichi as ti

ti.init(arch=ti.gpu, default_ip=ti.i32)  # must run before any field is created

import constants as C  # noqa: E402
from camera import OrbitCamera  # noqa: E402
from render import pixels, render  # noqa: E402
from stitches import build_curves  # noqa: E402


def main():
    gui = ti.GUI(f"Satin stitch ({C.CURVE_TYPE})", res=(C.WIDTH, C.HEIGHT))
    cam = OrbitCamera()
    dragging = False
    last = (0.0, 0.0)
    frame = 0

    while gui.running:
        while gui.get_event():
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif gui.event.key == ti.GUI.LMB:
                dragging = gui.event.type == ti.GUI.PRESS
                last = gui.get_cursor_pos()

        if dragging:
            mx, my = gui.get_cursor_pos()
            cam.orbit(mx - last[0], my - last[1])
            last = (mx, my)
        if gui.is_pressed('w', '='):
            cam.zoom(-C.ZOOM_STEP)
        if gui.is_pressed('s', '-'):
            cam.zoom(C.ZOOM_STEP)
        if gui.is_pressed('t'):
            cam.reset()

        build_curves(frame * C.TIME_STEP)
        render()
        gui.set_image(pixels.to_numpy())
        gui.show()
        frame += 1


if __name__ == "__main__":
    main()
