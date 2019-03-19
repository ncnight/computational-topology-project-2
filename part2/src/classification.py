from sklearn import svm
import sklearn as sk
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

"""
Constants
"""
PNG_DIR = "../../part1/data/original_png/"


def generate_test_and_training_set_raw(img_dir, test_num=7):
    pngs = [img_dir +  f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    pngs = sorted(pngs, key=lambda s: s.casefold())
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    label = 0
    for counter, png in enumerate(pngs):
        if counter % 10 == 0 and counter != 0:
            label += 1
        if counter % 10 == test_num:
            x_test.append(png)
            y_test.append(label)
        else:
            x_train.append(png)
            y_train.append(label)

    return (x_train, y_train, x_test, y_test)

def generate_256x256_img(img_files):
    img_vecs = []
    for counter, pic in enumerate(img_files):
        im = Image.open(pic)
        im = im.resize((256,256), Image.ANTIALIAS)
        mpeg = np.asarray(im)
        img_vecs.append(mpeg.flatten())
    return np.asarray(img_vecs)

def svm_validate(x_test, y_test, model):
    return model.score(x_test, y_test)

def fit_svm(x_train, y_train, untrainged_model):
    untrainged_model.fit(x_train, y_train)

def print_training_output(description, accuracies):
    best_test_acc, best_train_acc, worst_test_acc, worst_train_acc = accuracies
    print("--------------------------" + description + "--------------------------")
    print("Best testing accuracy: " + str(best_test_acc) + " with training accuracy of: " + str(best_train_acc))
    print("Worst testing accuracy: " + str(worst_test_acc) + " with training accuracy of: " + str(worst_train_acc))
    print("--------------------------------------------------------------------------")

def cross_validate(model, data_dir):
    best_test_acc = 0
    best_train_acc = 0
    worst_test_acc = 2 # larger than 100%
    worst_train_acc = 2
    for y_loc in range(8):
        x_train, y_train, x_test, y_test = generate_test_and_training_set_raw(data_dir, test_num=y_loc)
        x_train = generate_256x256_img(x_train)
        x_test = generate_256x256_img(x_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        fit_svm(x_train, y_train, model)
        test_acc = svm_validate(x_test, y_test, model)
        train_acc = svm_validate(x_train, y_train, model)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
        if test_acc < worst_test_acc:
            worst_test_acc = test_acc
            worst_train_acc = train_acc
        model = sk.base.clone(model)
    return (best_test_acc, best_train_acc, worst_test_acc, worst_train_acc)


def main():
    naive_model = svm.SVC(C=1.5, gamma="scale")
    accuracies = cross_validate(naive_model, PNG_DIR)
    print_training_output("Naive version", accuracies)

if __name__ == '__main__':
    main()