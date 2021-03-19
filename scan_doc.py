import numpy as np
import cv2
from copy import deepcopy
from multiprocessing import freeze_support
from src.misc_utils import save_image_opencv
from src.plot_utils import plot_hough_lines
from src.scan_utils import (skeletonize, non_max_suppression_vertices,
                            get_max_width_height, get_intersection_vertices,
                            get_bounding_quadrilateral)
from src.CONSTS import ORIGIN_HEIGHT, NUM_CANDIDATE_VERTICES, MIN_NUM_VOTES


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


class ScanDoc:
    def __init__(self, input_image):
        self.net = self._load_hed_model()
        self.original_img = deepcopy(input_image)
        self._resize_img(input_image)

    def _resize_img(self, input_image):
        input_height = input_image.shape[0]
        input_width = input_image.shape[1]
        aspect_ratio = input_height / input_width
        self.HEIGHT = input_height
        self.WIDTH = input_width
        if input_height > ORIGIN_HEIGHT:
            self.WIDTH = int(ORIGIN_HEIGHT / aspect_ratio)
            self.HEIGHT = ORIGIN_HEIGHT
        self.input_image = cv2.resize(input_image,
                                      (self.WIDTH, self.HEIGHT))
        self.scale = input_height / ORIGIN_HEIGHT

    def _load_hed_model(self):
        net = cv2.dnn.readNetFromCaffe('model/deploy.prototxt',
                                       'model/hed_pretrained_bsds.caffemodel')
        cv2.dnn_registerLayer('Crop', CropLayer)
        return net

    def detect_edges_canny(self):
        # convert the image to gray scale
        gray_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        # blur the image to remove the high frequency noise
        gray_image_blured = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # perform Canny edge detection
        edged_image = cv2.Canny(gray_image_blured, 75, 200)
        save_image_opencv(edged_image, 'canny')
        return edged_image

    def detect_edges_hed(self):
        """
        Function to return an edged image using HED alogorithm
        """
        inp = cv2.dnn.blobFromImage(self.input_image, scalefactor=1.0,
                                    size=(self.WIDTH, self.HEIGHT),
                                    mean=(104.00698793, 116.66876762, 122.67891434),
                                    swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()
        hed = cv2.resize(out[0, 0], (self.WIDTH, self.HEIGHT))
        hed = (255 * hed).astype("uint8")
        save_image_opencv(hed, 'hed')
        hed_skel = skeletonize(hed).astype("uint8")
        save_image_opencv(hed_skel, 'hed_skel')
        thresheld_hed = cv2.threshold(hed_skel, 0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        save_image_opencv(thresheld_hed, 'thresheld_hed')
        return hed, thresheld_hed

    def detect_doc_corner(self, hed, threshold_hed, min_votes):
        plot_hough_lines(threshold_hed, deepcopy(self.input_image))
        vertices = get_intersection_vertices(threshold_hed, min_votes)
        vertices = np.unique(vertices, axis=0)
        full_point_image = deepcopy(self.input_image)
        new_image = deepcopy(self.input_image)
        for x, y in vertices:
            cv2.circle(full_point_image, (x, y),
                       radius=1, color=(0, 0, 255), thickness=5)

        save_image_opencv(full_point_image, 'hed_point_full')
        if vertices.shape[0] <= NUM_CANDIDATE_VERTICES:
            filtered_vertices = non_max_suppression_vertices(vertices,
                                                             hed, 1)
            for x, y in filtered_vertices:
                cv2.circle(new_image, (x, y),
                           radius=1, color=(0, 0, 255), thickness=5)
            save_image_opencv(new_image, 'hed_point')
            return filtered_vertices
        filtered_vertices = non_max_suppression_vertices(vertices,
                                                         hed)
        filtered_vertices = np.array(filtered_vertices)
        vertice_pixel_intensity = hed[filtered_vertices[:, 1], filtered_vertices[:, 0]]
        filtered_vertices = filtered_vertices[vertice_pixel_intensity.argsort()[::-1]]

        if filtered_vertices.shape[0] > NUM_CANDIDATE_VERTICES:
            filtered_vertices = filtered_vertices[:NUM_CANDIDATE_VERTICES]

        for x, y in filtered_vertices:
            cv2.circle(new_image, (x, y),
                       radius=1, color=(0, 0, 255), thickness=5)
        save_image_opencv(new_image, 'hed_point')
        return filtered_vertices

    def detect_doc_boundary(self, hed, edged_image):
        '''
        get document boundary, if failed, reduce min_votes for the hough line
        algorithm
        '''
        min_votes = MIN_NUM_VOTES
        for _ in range((MIN_NUM_VOTES - 50) // 10):
            try:
                filtered_vertices = self.detect_doc_corner(hed, edged_image, min_votes)
                quadrilateral = get_bounding_quadrilateral(filtered_vertices,
                                                           hed,
                                                           deepcopy(self.input_image))
                break
            except ValueError:
                min_votes -= 10
        max_width, max_height = get_max_width_height(quadrilateral * self.scale)
        quadrilateral = np.array(quadrilateral, dtype="float32") * self.scale
        destination_corner = np.array([[0, 0],
                                       [max_width - 1, 0],
                                       [max_width - 1, max_height - 1],
                                       [0, max_height - 1]], dtype="float32")

        return quadrilateral, destination_corner, max_width, max_height

    def get_scanned_document(self):
        '''
        function to get scanning quality document
        '''
        hed, edged_image = self.detect_edges_hed()
        quadrilateral, destination_corner, \
            max_width, max_height\
            = self.detect_doc_boundary(hed, edged_image)
        transformation_matrix = cv2.getPerspectiveTransform(quadrilateral,
                                                            destination_corner)
        scanned_doc = cv2.warpPerspective(self.original_img,
                                          transformation_matrix,
                                          (max_width, max_height))
        save_image_opencv(scanned_doc, 'scanned')
        return scanned_doc

    def get_deskewed_document(self, scanned_doc):
        '''
        Function to deskew document text
        '''
        gray_img = cv2.cvtColor(scanned_doc, cv2.COLOR_BGR2GRAY)
        # Make white foreground, black background
        gray_img = cv2.bitwise_not(gray_img)
        # Threshold the image, setting all foreground pixels to
        # 255 and all background pixels to 0
        binary_img = cv2.threshold(gray_img, 0, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Erode the image to remove pixelated noise
        kernel = np.ones((5, 5), np.uint8)
        eroded_img = cv2.erode(binary_img, kernel, iterations=1)
        text_block_coords = np.column_stack(np.where(eroded_img > 0))
        hgt_rot_angle = cv2.minAreaRect(text_block_coords)[-1]
        com_rot_angle = hgt_rot_angle + 90 if hgt_rot_angle < -45 else hgt_rot_angle

        (h, w) = scanned_doc.shape[0:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, com_rot_angle, 1.0)
        deskewed_doc = cv2.warpAffine(scanned_doc, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
        save_image_opencv(deskewed_doc, 'desckewed')
        return deskewed_doc

    def get_greyscale_scan(self):
        scanned_doc = self.get_scanned_document()
        de_skewed_doc = self.get_deskewed_document(scanned_doc)
        bin_deskewed_doc = cv2.threshold(cv2.cvtColor(de_skewed_doc, cv2.COLOR_BGR2GRAY), 0, 255,
                                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        pre_processed_doc = cv2.cvtColor(bin_deskewed_doc, cv2.COLOR_GRAY2BGR)
        save_image_opencv(pre_processed_doc, 'scan_grayscale')
        return scanned_doc


if __name__ == '__main__':
    freeze_support()
    input_img = cv2.imread('test_img/img_7.jpg')
    sd = ScanDoc(input_img)
    scanned_doc = sd.get_greyscale_scan()
