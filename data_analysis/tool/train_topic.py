import os
import json
import fasttext
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



train_csv_path = '/data/wangyichuan/process/analysis/train_data/valid.csv'
test_csv_path = '/data/wangyichuan/process/analysis/train_data/test.csv'
train_data_path = '/data/wangyichuan/process/analysis/train_data/data_train.txt'
model_path = '/data/wangyichuan/process/analysis/model/topic.bin'


def make_data(data_path=train_csv_path):
    texts, labels = [], []
    with open(data_path, 'r') as f:
        for line in f:
            temp = line.replace('\n', '').split(',')
            text, label = ','.join(temp[:-1]), temp[-1]
            texts.append(text)
            labels.append(label)
    # train_X, test_X, train_Y, test_Y = train_test_split(texts, labels, test_size=0.3, random_state=42, shuffle=True)
    return texts[1:], labels[1:]


def save_train_data(texts, labels, train_path=train_data_path):
    train = [text+'\t__label__'+label+'\n' for text, label in zip(texts, labels)]
    with open(train_path, 'w') as f:
        f.writelines(train)


def predict(text, model):
    return model.predict(text)[0][0][9:]



if __name__ == '__main__':
    texts, labels = make_data(train_csv_path)
    print(len(texts), len(labels))
    save_train_data(texts, labels)
    # train
    model = fasttext.train_supervised(train_data_path, epoch=10)
    model.save_model(model_path)
    # dev
    pred_label = []
    for text in texts:
        pred_label.append(predict(text, model))
    print(labels[:5])
    print(pred_label[:5])
    print(accuracy_score(labels, pred_label))
    # test
    texts_test, labels_test = make_data(test_csv_path)
    print(len(texts_test), len(labels_test))
    pred_label = []
    for text in texts_test:
        pred_label.append(predict(text, model))
    print(labels_test[:5])
    print(pred_label[:5])
    print(accuracy_score(labels_test, pred_label))
    # all train
    save_train_data(texts+texts_test, labels+labels_test)
    model = fasttext.train_supervised(train_data_path, epoch=10)
    model.save_model(model_path)
    # test
    pred_label = []
    for text in texts:
        pred_label.append(predict(text, model))
    print(labels[:5])
    print(pred_label[:5])
    print(accuracy_score(labels, pred_label))

    pred_label = []
    for text in texts_test:
        pred_label.append(predict(text, model))
    print(labels_test[:5])
    print(pred_label[:5])
    print(accuracy_score(labels_test, pred_label))