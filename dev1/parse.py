import os
import pandas as pd
import docx2txt
import openpyxl

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


df = pd.DataFrame(rows, columns=["filepath", "sentence"])
df.to_csv("parsed.csv", encoding='utf_8_sig')
    

# with open("parsed.csv", 'w', encoding='utf_8_sig') as f:
