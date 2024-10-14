from DatasetProcessor import DatasetProcessor
from RandomForestDataProcessor import RandomForestDataProcessor
from predict import predict_forest, predict_example, predict_set
from save_load_trained_tree import save_tree_to_file, load_tree_from_file
from ID3 import ID3
from utils import parse_arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


def train_tree(data_processor, attributes):
    rootNode = ID3('', attributes, data_processor)
    return rootNode

def predict_with_model(test_df, model):
    if isinstance(model, list):
        test_features = test_df.drop(columns=['class'])
        predictions = []
        for _, example in test_features.iterrows():
            prediction = predict_forest(example, model)
            predictions.append(prediction)
    else:
        test_features = test_df.drop(columns=['class'])
        predictions = predict_set(test_features, model)
    return predictions

def save_model(model, filename):
    if isinstance(model, list):
        for idx, tree in enumerate(model):
            tree_filename = f"{filename}_tree_{idx + 1}.txt"
            save_tree_to_file(tree, tree_filename)
    else:
        save_tree_to_file(model, filename)

def load_model(filename, use_random_forest=False, n_trees=10):
    if use_random_forest:
        trees = []
        for idx in range(n_trees):
            tree_filename = f"{filename}_tree_{idx + 1}.txt"
            tree = load_tree_from_file(tree_filename)
            trees.append(tree)
        return trees
    else:
        return load_tree_from_file(filename)



def train_and_visualize_decision_tree(X_train, Y_train):
    Y_train_str = Y_train.astype(str)
    label_encoder = LabelEncoder()
    Y_train_encoded = label_encoder.fit_transform(Y_train_str)

    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, Y_train_encoded)

    plt.figure(figsize=(30,20))
    plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=label_encoder.classes_, max_depth = 3)
    plt.show()


def TreeProcessing(use_random_forest=False, n_trees=10, train_new_tree=True, num_folds=5):
    selected_columns = ['heartrate', 'chAVR_TwaveAmp', 'chV6_TwaveAmp', 'chDI_TwaveAmp', 'chV1_RPwaveAmp',
                        'chV1_RPwave', 'chV5_TwaveAmp', 'chV3_QwaveAmp', 'chV3_Qwave', 'chV1_QRSA', 'chDII_TwaveAmp',
                        'chV6_QRSTA', 'chV1_intrinsicReflecttions', 'chV3_Rwave', 'chV3_intrinsicReflecttions',
                        'chV3_RwaveAmp', 'chV5_QRSTA', 'Tinterval', 'chV3_Swave', 'chV3_SwaveAmp', 'chV2_QRSA',
                        'QRSduration', 'class']

    accuracies = []

    if use_random_forest:
        a = 1
    else:
        a = num_folds

    b = num_folds + 1

    for fold in range(a, b):
        train_file = f"train/train_fold_{fold}.arff"
        test_file = f"test/test_fold_{fold}.arff"
        model_filename = f'decision_tree{fold}.txt'

        if train_new_tree:
            df, attribute_types = parse_arff(train_file)
            df = df[selected_columns]

            filtered_attribute_types = {k: v for k, v in attribute_types.items() if k in selected_columns}

            if use_random_forest:
                data_processor = RandomForestDataProcessor(df, filtered_attribute_types)
                trees = []
                for _ in range(n_trees):
                    subset = data_processor.generateRandomSubset()
                    attrib_list = data_processor.getSelectedAttribList()
                    subset_dp = RandomForestDataProcessor(subset, filtered_attribute_types)
                    tree = train_tree(subset_dp, attrib_list)
                    trees.append(tree)
                model = trees
            else:
                X_train = df.drop(columns=['class'])
                Y_train = df['class']
                #train_and_visualize_decision_tree(X_train, Y_train)
                data_processor = DatasetProcessor(df, filtered_attribute_types)
                attrib_list = df.columns.tolist()[:-1]
                rootNode = train_tree(data_processor, attrib_list)
                model = rootNode

            save_model(model, model_filename)
        else:
            model = load_model(model_filename, use_random_forest, n_trees)

        test_data, _ = parse_arff(test_file)
        test_df = pd.DataFrame(test_data)
        test_df = test_df[selected_columns]

        y_pred = predict_with_model(test_df, model)
        y_test = test_df['class'].values

        if len(y_test) != 0:
            if train_new_tree:
                accuracy = 100 * sum(y_pred == y_test) / len(y_test)
            else:
                correct_predictions = 0
                for actual, predicted in zip(y_test, y_pred):
                    if str(actual) == predicted:
                        #print(f'{actual} -> {predicted}')
                        correct_predictions += 1
                accuracy = 100 * correct_predictions / len(y_test)

            print(f"Acuratetea pe setul de testare pentru fold {fold}: {accuracy:.2f}%")
            accuracies.append(accuracy)
            #return accuracy
        for example, predicted_class in zip(test_df.values.tolist(), y_pred):
            print("Attributes:", example, "Predicted class:", predicted_class)

    if accuracies:
        mean_accuracy = np.mean(accuracies)
        print(f"Acuratetea medie pe cele {num_folds} folduri: {mean_accuracy:.2f}%")

    return f"Acuratetea pe setul de testare: {accuracy: .2f}%"

if __name__ == '__main__':
    TreeProcessing(use_random_forest=True, n_trees=10, train_new_tree=False, num_folds=1)

