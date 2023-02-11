import fitz
from PIL import Image
from paddleocr import PaddleOCR,draw_ocr
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='japan') # need to run only once to download and load model into memory

def pdf2opencv_image(pdf_filename):
    """pdfファイルを開き、全ページをレンダリングして、opencvの画像データの配列にして返す。

    Args:
        pdf_filename (string): pdfファイル名
    """
    
    pages_opencv_image = []
    with fitz.open(pdf_filename) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix = fitz.Matrix(4,4))
            pillow_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
            opencv_image = np.array(pillow_image, dtype=np.uint8)
            pages_opencv_image.append(opencv_image)
            
    return pages_opencv_image

def apply_ocr(images):
    """OCRを実行し、テキストの配列を返す。

    Args:
        images (list(np.ndarray)): opencvの画像データのリスト
    """
    buffer = []
    for image in images:
        result = ocr.ocr(image, cls=False)
        for x in result:
            for line in x:
                print(line)
                buffer.append(line[-1][0])
        break
    return buffer
        

if __name__ == "__main__":
    images = pdf2opencv_image("data/146064.pdf")
    text = apply_ocr(images)
    