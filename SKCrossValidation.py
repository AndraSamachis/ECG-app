import pandas as pd
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold


def save_arff(file_path, df):
    with open('fisiere/meta.txt', 'r') as txt_file:
        text_content = txt_file.read()

    with open(file_path, 'w') as f:
        f.write(text_content)
        f.write("\n@DATA\n")
        for row in df.values.tolist():
            row_str = ''
            for item in row:
                if isinstance(item, bytes):
                    row_str += item.decode("utf-8")
                else:
                    row_str += str(item)
                row_str += ','
            row_str = row_str.rstrip(',')
            f.write(row_str + '\n')

def perform_skf(df, class_column, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    y = df[class_column].astype(str)  #convertim in siruri de caractere

    for i, (train_idx, test_idx) in enumerate(skf.split(df, y)):

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        save_arff(f'train_fold_{i + 1}.arff', train_df)
        save_arff(f'test_fold_{i + 1}.arff', test_df)

def skf():
    data, meta = arff.loadarff('fisiere/instante_eliminate.arff')
    df = pd.DataFrame(data)
    perform_skf(df, 'class')

