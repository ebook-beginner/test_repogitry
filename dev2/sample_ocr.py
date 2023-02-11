from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='japan') # need to run only once to download and load model into memory
img_path = './japan_2.jpg'
result = ocr.ocr(img_path, cls=True)
for x in result:
    for line in x:
        print(line[-1][0])
    