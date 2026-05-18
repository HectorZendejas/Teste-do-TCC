import pandas as pd
from pathlib import Path
base = Path('c:/Users/hecto/Downloads/TCCteste')
for name in ['Data_Train.xlsx', 'Test_set.xlsx']:
    path = base / name
    xl = pd.ExcelFile(path)
    print('FILE:', name)
    print('SHEETS:', xl.sheet_names)
    df = pd.read_excel(path)
    print('SHAPE:', df.shape)
    print('COLUMNS:', list(df.columns))
    print(df.head(3).to_string(index=False))
    print('-' * 80)
