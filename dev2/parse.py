import os
import pandas as pd
import docx2txt
import openpyxl
import fitz
from PIL import Image
from paddleocr import PaddleOCR,draw_ocr
import numpy as np

# OCRモデルの読み込み
ocr = PaddleOCR(use_angle_cls=True, lang='japan') # need to run only once to download and load model into memory

def pdf2ndarray(pdf_filename):
    """pdfファイルを開き全ページをレンダリングして、ndarrayデータの配列にして返す。

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
                # print(line)
                buffer.append(line[-1][0])
    return buffer

rows = []

for root, dirs, files in os.walk(top='./data'):
    for file in files:
        file = os.path.join(root, file)
        
        # DOCXファイル
        if file.lower().endswith(('.docx', '.DOCX')):
            print(file)
            text = docx2txt.process(file)
            if text is None:
                continue
            elif len(text) == 1:
                f.write(file +","+ text.replace("\n", ""))
            elif len(text) > 1:
                text2 = [t.replace("\n", "") for t in text]
                text2 = ''.join(text2).replace(",","")
                # f.write( + "\n")
                rows.append([file, text2])

        # XLSXファイル
        if file.lower().endswith(('.xlsx', '.XLSX')):
            print(file)
            buffer = []
            wb = openpyxl.load_workbook(file, data_only=True)
            for ws in wb.worksheets:
                for cells in tuple(ws.rows):
                    for cell in cells:
                        if cell.value is not None:
                            buffer.append(str(cell.value).replace("\n", "").replace(",",""))
            buffer = ' '.join(buffer)
            # f.write(file + "," + buffer + "\n")
            rows.append([file, buffer])

        # PDFファイル
        if file.lower().endswith(('.pdf')):
            print(file)
            buffer = []
            images = pdf2ndarray(file)
            text = apply_ocr(images)
            # f.write(file + "," + " ".join(text).replace(",","") + "\n")
            rows.append([file, text])

df = pd.DataFrame(rows, columns=["filepath", "sentence"])
df.to_csv("parsed.csv", encoding='utf_8_sig')
