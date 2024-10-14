from scipy.io import arff
import pandas as pd
from ID3 import calculate_information_gain
from DatasetProcessor import DatasetProcessor
class_names = {
    b'1': 'Normal',
    b'2': 'Ischemic changes (Coronary Artery Disease)',
    b'3': 'Old Anterior Myocardial Infarction',
    b'4': 'Old Inferior Myocardial Infarction',
    b'5': 'Sinus tachycardia',
    b'6': 'Sinus bradycardia',
    b'9': 'Left bundle branch block',
    b'10': 'Right bundle branch block'
}
#cheile - siruri de caractere in format byte
def parse_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    attribute_types = {}
    all_attributes_list = meta.names()
    for attribute in all_attributes_list:
        attribute_types[attribute] = meta[attribute][0]
    return df, attribute_types


def get_attributes_sorted_by_information_gain(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    attribute_types = {attribute: meta[attribute][0] for attribute in df.columns}

    dp = DatasetProcessor(df, attribute_types)
    information_gains = []

    for attribute in df.columns[:-1]:
        information_gain, _ = calculate_information_gain(dp, attribute)
        information_gains.append((attribute, information_gain))

    information_gains.sort(key=lambda x: x[1], reverse=True)

    for attribute, gain in information_gains:
        loc = df.columns.get_loc(attribute)
        print(f"{gain:.5f}\t{loc}\t{attribute}")

