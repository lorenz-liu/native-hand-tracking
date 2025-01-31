import math
import time

import cv2
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from pygame.locals import *

from native_hand_tracker import NativeHandTracker


class Cube3D:
    def __init__(self):
        self.vertices = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]

        self.edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        self.faces = [
            (0, 1, 2, 3),
            (5, 4, 7, 6),
            (4, 0, 3, 7),
            (1, 5, 6, 2),
            (4, 5, 1, 0),
            (3, 2, 6, 7),
        ]

        self.colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

        self.rotation = [0, 0, 0]

    def draw(self):
        glPushMatrix()
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)

        # Draw faces
        glBegin(GL_QUADS)
        for i, face in enumerate(self.faces):
            glColor3fv(self.colors[i])
            for vertex in face:
                glVertex3fv(self.vertices[vertex])
        glEnd()

        # Draw edges
        glColor3f(0, 0, 0)
        glBegin(GL_LINES)
        for edge in self.edges:
            for vertex in edge:
                glVertex3fv(self.vertices[vertex])
        glEnd()

        glPopMatrix()


def main():
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Set up the 3D perspective
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    # Initialize hand tracker and camera
    cap = cv2.VideoCapture(0)
    tracker = NativeHandTracker()
    cube = Cube3D()

    # Previous state variables
    prev_pinch_center = None
    prev_angle = None
    is_grabbed = False

    clock = pygame.time.Clock()

    while True:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Process hand tracking
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.process_frame(frame)
        landmark_list = tracker.get_landmark_positions(frame)

        if landmark_list:
            # Get pinch information
            pinch_distance, pinch_center = tracker.get_pinch_details(landmark_list)

            # Detect pinch gesture
            if tracker.is_pinching(landmark_list):
                if not is_grabbed:
                    is_grabbed = True
                    prev_pinch_center = pinch_center
                    prev_angle = tracker.get_hand_rotation(landmark_list)
                else:
                    if prev_pinch_center:
                        # Calculate rotation based on hand movement
                        dx = pinch_center[0] - prev_pinch_center[0]
                        dy = pinch_center[1] - prev_pinch_center[1]

                        # Update cube rotation
                        sensitivity = 0.5
                        cube.rotation[1] += dx * sensitivity
                        cube.rotation[0] += dy * sensitivity

                        # Normalize angles
                        cube.rotation = [angle % 360 for angle in cube.rotation]

                    prev_pinch_center = pinch_center
            else:
                is_grabbed = False
                prev_pinch_center = None

        # Clear the screen and redraw the cube
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        cube.draw()
        pygame.display.flip()

        # Show the camera feed in a separate window
        cv2.imshow("Hand Tracking", frame)

        # Handle exit condition
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        clock.tick(60)

    # Clean up
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
