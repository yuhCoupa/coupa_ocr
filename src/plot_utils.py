import numpy as np
import cv2
from .misc_utils import save_image_opencv
from .CONSTS import MIN_NUM_VOTES, NUM_HOUGH_LINES


def plot_hough_lines(edges, origin_image):
    '''
    get and plot hough lines
    '''
    lines = cv2.HoughLines(edges, 1, np.pi / 180, MIN_NUM_VOTES)
    for rho, theta in np.squeeze(lines)[:NUM_HOUGH_LINES, ]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(origin_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    save_image_opencv(origin_image, 'hough_lined')
