import re
import cv2
import pandas as pd
from PIL import Image
import tesserocr
from tesserocr import (PyTessBaseAPI, RIL,
                       iterate_level, PSM)
from scan_doc import ScanDoc
from src.misc_utils import save_image_opencv
Image.MAX_IMAGE_PIXELS = 1000000000
pd.set_option('display.max_rows', 1000)
pd.options.mode.chained_assignment = None


class CoupaOCR:
    def __init__(self):
        self.tess_api = PyTessBaseAPI(oem=tesserocr.OEM.LSTM_ONLY,
                                      psm=PSM.SPARSE_TEXT)
        self.text_detect_level = RIL.TEXTLINE

    @staticmethod
    def get_scanned_doc(input_img):
        sd = ScanDoc(input_img)
        scanned_doc = sd.get_greyscale_scan()
        return scanned_doc

    def extract_text_to_df(self, input_image):
        scanned_doc = self.get_scanned_doc(input_image)
        pil_img = Image.fromarray(scanned_doc)
        self.tess_api.SetImage(pil_img)
        self.tess_api.Recognize()
        text_iterator = self.tess_api.GetIterator()
        text_data = []
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for text_attr in iterate_level(text_iterator, self.text_detect_level):
            _text = text_attr.GetUTF8Text(self.text_detect_level)
            text_data.append(_text)
            _x1, _y1, _x2, _y2 = text_attr.BoundingBox(self.text_detect_level)
            x1.append(_x1)
            y1.append(_y1)
            x2.append(_x2)
            y2.append(_y2)
            cv2.rectangle(scanned_doc, (_x1, _y1), (_x2, _y2), (255, 0, 0), 2)
        save_image_opencv(scanned_doc, 'text_rec')
        page_height = scanned_doc.shape[0]
        page_width = scanned_doc.shape[1]
        df_out = pd.DataFrame()
        df_out.loc[:, 'Data'] = text_data
        df_out.loc[:, 'X1'] = x1
        df_out.loc[:, 'Y1'] = y1
        df_out.loc[:, 'X2'] = x2
        df_out.loc[:, 'Y2'] = y2
        df_out.loc[:, 'PageHeight'] = page_height
        df_out.loc[:, 'PageWidth'] = page_width
        return df_out


def remove_new_line_symbol(data):
    return re.sub("\n", "", data)


if __name__ == '__main__':
    input_img = cv2.imread('test_img/img_7.jpg')
    coupa_ocr = CoupaOCR()
    df_out = coupa_ocr.extract_text_to_df(input_img)
    df_out.loc[:, 'Data'] = df_out.Data.map(remove_new_line_symbol)
    import pdb; pdb.set_trace()
