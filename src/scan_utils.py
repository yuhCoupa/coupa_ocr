import numpy as np
import cv2
from multiprocessing import Pool
from functools import partial
from copy import deepcopy
from math import atan, acos
from itertools import combinations
from .CONSTS import MIN_NUM_VOTES, MIN_AREA_RATIO, NON_MAX_SUP_RANGE, NUM_CPUS, NUM_HOUGH_LINES, EXPAND_PIXEL
from .misc_utils import save_image_opencv


def skeletonize(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        open_ = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def get_intersection_vertices(edges, min_votes=MIN_NUM_VOTES):
    '''
    get intersection vertices between hough lines
    '''
    img_shape = edges.shape
    lines = cv2.HoughLines(edges, 1, np.pi / 180, min_votes)
    lines = np.squeeze(lines)[:NUM_HOUGH_LINES, :]
    vertices = []
    group_lines = combinations(range(len(lines)), 2)

    def x_in_range(x):
        return 0 < x < img_shape[1] - EXPAND_PIXEL

    def y_in_range(y):
        return 0 < y < img_shape[0] - EXPAND_PIXEL

    for line_i_idx, line_j_idx in group_lines:
        line_i, line_j = lines[line_i_idx], lines[line_j_idx]

        if 80.0 < get_angle_between_lines(line_i, line_j) < 100.0:
            vertice = get_intersection(line_i, line_j)

            if x_in_range(vertice[0]) and y_in_range(vertice[1]):
                vertices.append(vertice)
    return np.array(vertices)


def get_angle_between_lines(line_1, line_2):
    '''
    get angle between hough lines
    '''
    _, theta1 = line_1
    _, theta2 = line_2
    theta1 += 1e-10
    theta2 += 1e-10
    # x * cos(theta) + y * sin(theta) = rho
    # y * sin(theta) = x * (- cos(theta)) + rho
    # y = x * (-cos(theta) / sin(theta)) + rho
    m1 = -(np.cos(theta1) / np.sin(theta1))
    m2 = -(np.cos(theta2) / np.sin(theta2))
    return abs(atan(abs(m2 - m1) / (1 + m2 * m1))) * (180 / np.pi)


def get_intersection(line1, line2):
    '''
    get the intersection of two lines given in Hesse normal form.
    see https://stackoverflow.com/a/383527/5087436
    '''
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return x0, y0


def _parallel_non_max_sup_vertices_fcn(p, sup_range, vertices, hed_image):
    '''
    stateless function for suppressing non-max intensity pixel
    '''
    H, W = hed_image.shape
    x, y = p
    x_start = max(x - sup_range, 0)
    x_end = min(x + sup_range + 1, W)
    y_start = max(y - sup_range, 0)
    y_end = min(y + sup_range + 1, H)
    contained_vertices = vertices[(vertices[:, 0] >= x_start) &
                                  (vertices[:, 0] <= x_end) &
                                  (vertices[:, 1] >= y_start) &
                                  (vertices[:, 1] <= y_end)]
    ctr_pixel_intensity = hed_image[y, x]
    region_max_pixel_intensity = np.max(hed_image[contained_vertices[:, 1],
                                                  contained_vertices[:, 0]])
    if ctr_pixel_intensity >= region_max_pixel_intensity:
        return (x, y)


def non_max_suppression_vertices(vertices, hed_image, sup_range=NON_MAX_SUP_RANGE):
    '''
    suppress a vertice in a local region if it is not the brightest
    '''
    vertices = np.array(vertices)
    _parallel_process = partial(_parallel_non_max_sup_vertices_fcn,
                                sup_range=sup_range,
                                vertices=vertices, hed_image=hed_image)
    with Pool(NUM_CPUS) as pool:
        _filtered_vertices = pool.map(_parallel_process, vertices)
    filtered_vertices_ = list(filter(None, _filtered_vertices))
    filtered_vertices = np.unique(filtered_vertices_, axis=0)
    return filtered_vertices


def order_quadrilateral(quadrilateral):
    '''
    order vertices of a quadrilateral in tl,tr,br,bl order
    '''
    quadrilateral = np.array(quadrilateral)
    ctr = np.mean(quadrilateral, axis=0)
    rel_pos = quadrilateral - ctr
    top = quadrilateral[rel_pos[:, 1] <= 0]
    top = top[top[:, 0].argsort()]
    bot = quadrilateral[rel_pos[:, 1] > 0]
    bot = bot[bot[:, 0].argsort()[::-1]]
    return np.vstack((top, bot)).astype(int)


def is_quadrilateral_convex(vertices):
    '''
    check if a quadrilater formed by four vertices is convex
    '''
    vertice_triplets = [[0, 1, 2],
                        [1, 2, 3],
                        [2, 3, 0],
                        [3, 0, 1]]

    def get_cross_product(a, b, c):
        return (a[0] - b[0]) * (b[1] - c[1]) - (a[1] - b[1]) * (b[0] - c[0])

    def get_angle_between_vectors(a, b, c):
        ab = (b - a) + 1e-15
        bc = (c - b) + 1e-15
        cos_ = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        cos_ = np.clip(cos_, -0.9999, 0.9999)
        theta = acos(cos_) * (180 / np.pi)
        return theta

    signs = []
    for triplet in vertice_triplets:
        a, b, c = vertices[triplet]
        theta = get_angle_between_vectors(a, b, c)
        if theta < 80 or theta > 100:
            return False
        signs.append(get_cross_product(a, b, c) > 0)
    return all(signs) or not any(signs)


def get_quadrilateral_area(quadrilateral):
    '''
    calcuate the area of a quadrilateral
    '''
    quadrilateral = np.expand_dims(quadrilateral, axis=1)
    area = cv2.contourArea(quadrilateral)
    return area


def score_segement(p1, p2, hed):
    '''
    accumulate the pixel intensity lying on a line segement
    of a quadrilateral
    '''
    H = hed.shape[0]
    W = hed.shape[1]
    p1X = p1[0]
    p1Y = p1[1]
    p2X = p2[0]
    p2Y = p2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = p2X - p1X
    if dX < 0:
        dX += -1
    else:
        dX += 1

    dY = p2Y - p1Y
    if dY < 0:
        dY += -1
    else:
        dY += 1
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itcache = np.empty(shape=(np.maximum(dYa, dXa), 3),
                       dtype=np.float32)
    itcache.fill(np.nan)

    # Obtain coordinates along the line
    negY = p1Y > p2Y
    negX = p1X > p2X
    if p1X == p2X:  # vertical line segment
        itcache[:, 0] = p1X
        if negY:
            itcache[:, 1] = np.arange(p1Y, p2Y - 1, -1)
        else:
            itcache[:, 1] = np.arange(p1Y, p2Y + 1)
    elif p1Y == p2Y:  # horizontal line segment
        itcache[:, 1] = p1Y
        if negX:
            itcache[:, 0] = np.arange(p1X, p2X - 1, -1)
        else:
            itcache[:, 0] = np.arange(p1X, p2X + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itcache[:, 1] = np.arange(p1Y, p2Y - 1, -1)
            else:
                itcache[:, 1] = np.arange(p1Y, p2Y + 1)
            itcache[:, 0] = (slope * (itcache[:, 1] - p1Y)).astype(np.int) + p1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itcache[:, 0] = np.arange(p1X, p2X - 1, -1)
            else:
                itcache[:, 0] = np.arange(p1X, p2X + 1)
            itcache[:, 1] = (slope * (itcache[:, 0] - p1X)).astype(np.int) + p1Y

    # Remove points outside of image
    colX = itcache[:, 0]
    colY = itcache[:, 1]
    itcache = itcache[(colX >= 0) & (colY >= 0) & (colX < W) & (colY < H)]

    # Get intensities from hed ndarray
    itcache[:, 2] = hed[itcache[:, 1].astype(np.uint), itcache[:, 0].astype(np.uint)]
    return itcache


def _parallel_get_quadrilateral_fcn(quadrilateral_idx, filtered_vertices, hed):
    '''
    stateless function to get the accumulated intensity of all lines of a
    quadrilateral
    '''
    pt_i, pt_j, pt_k, pt_l = quadrilateral_idx
    image_area = hed.shape[0] * hed.shape[1]
    line_segement_idx = [[0, 1],
                         [1, 2],
                         [2, 3],
                         [3, 0]]
    quadrilateral = [filtered_vertices[pt_i],
                     filtered_vertices[pt_j],
                     filtered_vertices[pt_k],
                     filtered_vertices[pt_l]]
    quadrilateral = order_quadrilateral(quadrilateral)
    is_valid = is_quadrilateral_convex(quadrilateral)

    _score = 0
    if is_valid:
        area = get_quadrilateral_area(quadrilateral)
        area_ratio = area / image_area
        if area_ratio >= MIN_AREA_RATIO:
            for seg_idx in line_segement_idx:
                p1, p2 = quadrilateral[seg_idx]
                itcache = score_segement(p1, p2, hed)
                _score += np.sum(itcache[:, -1])
            return (quadrilateral, _score)


def get_bounding_quadrilateral(filtered_vertices, hed, test_image):
    '''
    get the most fitting bounding quadrilateral of a document
    '''
    quadrilateral_idx = combinations(range(len(filtered_vertices)), 4)
    _parallel_process = partial(_parallel_get_quadrilateral_fcn,
                                filtered_vertices=filtered_vertices,
                                hed=hed)
    with Pool(NUM_CPUS) as pool:
        _valid_quadrilaterals = pool.map(_parallel_process, quadrilateral_idx)
    valid_quadrilaterals_ = list(filter(None, _valid_quadrilaterals))
    valid_quadrilaterals, score = zip(*valid_quadrilaterals_)
    bounding_quadrilateral = valid_quadrilaterals[np.argmax(score)]
    tl = bounding_quadrilateral[0]
    tr = bounding_quadrilateral[1]
    br = bounding_quadrilateral[2]
    bl = bounding_quadrilateral[3]
    if (tl[0] - EXPAND_PIXEL > 0) and (tl[1] - EXPAND_PIXEL > 0):
        bounding_quadrilateral[0, :] = np.array([tl[0] - EXPAND_PIXEL,
                                                 tl[1] - EXPAND_PIXEL])
    if (tr[0] + EXPAND_PIXEL < hed.shape[1]) and (tr[1] - EXPAND_PIXEL > 0):
        bounding_quadrilateral[1, :] = np.array([tr[0] + EXPAND_PIXEL,
                                                 tr[1] - EXPAND_PIXEL])
    if (br[0] + EXPAND_PIXEL < hed.shape[1]) and (br[1] + EXPAND_PIXEL < hed.shape[0]):
        bounding_quadrilateral[2, :] = np.array([br[0] + EXPAND_PIXEL,
                                                 br[1] + EXPAND_PIXEL])
    if (bl[0] - EXPAND_PIXEL > 0) and (bl[1] + EXPAND_PIXEL < hed.shape[0]):
        bounding_quadrilateral[3, :] = np.array([bl[0] - EXPAND_PIXEL,
                                                 bl[1] + EXPAND_PIXEL])
    origin_image = deepcopy(test_image)
    poly_lines = np.array(bounding_quadrilateral, np.int32).reshape((-1, 1, 2))
    cv2.polylines(origin_image, [poly_lines],
                  isClosed=True,
                  color=(255, 0, 0), thickness=2)
    save_image_opencv(origin_image, 'polygon')
    return bounding_quadrilateral


def get_max_width_height(bounding_quadrilateral):
    tl, tr, br, bl = bounding_quadrilateral
    # Calculate width
    width_a = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2)
    width_b = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2)
    max_width = max(int(width_a), int(width_b))

    # Calculate height
    height_a = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2)
    height_b = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2)
    max_height = max(int(height_a), int(height_b))
    return max_width, max_height
