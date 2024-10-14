def predict_example(example, node):
    #pana cand ajungem la un nod frunza
    while node.children:
        #daca nodul are punct de impartire
        if node.split_point is not None:
            try:
                example_value = float(example[node.name])
                split_point_value = float(node.split_point)
            except ValueError:
                #if conversia esueaza
                #convertim val bytes la string folosind codificarea utf 8
                example_value = example[node.name].decode('utf-8') if isinstance(example[node.name], bytes) else str(example[node.name])
                split_point_value = node.split_point.decode('utf-8') if isinstance(node.split_point, bytes) else str(node.split_point)

            #comparam valoarea cu punctul de impartire
            if example_value <= split_point_value:
                node = [child for child in node.children if '<=' in child.branchName][0]
            else:
                node = [child for child in node.children if '>' in child.branchName][0]
        else:
            attr_value = example[node.name].decode('utf-8') if isinstance(example[node.name], bytes) else example[node.name]
            #cautam ramura in care valoarea atributului este egala cu branchname
            node = next(child for child in node.children if (child.branchName.encode('utf-8') if isinstance(attr_value, bytes) else attr_value) == child.branchName)


    return node.name


def predict_set(data, node):
    predictions = []
    for index, example in data.iterrows():
        predictions.append(predict_example(example, node))
    return predictions

def predict_forest(example, trees):
    predictions = []

    for tree in trees:
        prediction = predict_example(example, tree)
        predictions.append(prediction)

    final_prediction = max(set(predictions), key=predictions.count)

    return final_prediction
