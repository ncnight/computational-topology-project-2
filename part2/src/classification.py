from sklearn import svm
import sklearn as sk
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import sklearn_tda as tda

"""
Constants
"""
PNG_DIR = "../../part1/data/original_png/"
DIM0_WASSERSTEIN = "../../part1/data/hera_output/wasserstein-dim0.npy"
DIM1_WASSERSTEIN = "../../part1/data/hera_output/wasserstein-dim1.npy"
DIM0_BOTTLENECK = "../../part1/data/hera_output/geom_bottleneck-dim0.npy"
DIM1_BOTTLENECK = "../../part1/data/hera_output/geom_bottleneck-dim1.npy"
RIPSER_OUTPUT = "../../part1/data/ripser_outputs/"
NP_CHECKPOINTS = "../np_checkpoints/"


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

def print_training_output(description, accuracies, C=None):
    best_test_acc, best_train_acc, worst_test_acc, worst_train_acc = accuracies
    print("--------------------------" + description + "--------------------------")
    if C is not None:
        print("Optimal C: " + str(C))
    print("Best testing accuracy: " + str(best_test_acc) + " with training accuracy of: " + str(best_train_acc))
    print("Worst testing accuracy: " + str(worst_test_acc) + " with training accuracy of: " + str(worst_train_acc))
    print("--------------------------------------------------------------------------")

def split_distance_matrix(distance_matrix, test_num=7):
    test_matrix = distance_matrix[test_num, :]
    slice = test_num
    slices = [slice]
    for _ in range(7):
        slice += 10
        slices.append(slice)
        test_matrix = np.vstack((test_matrix, distance_matrix[slice, :]))
    train_matrix = np.delete(distance_matrix, slices, axis=0)
    train_matrix = np.delete(train_matrix, slices, axis=1)
    test_matrix = np.delete(test_matrix, slices, axis=1)
    return train_matrix, test_matrix

def persistence_scale_kernel(X, Y, PSS_kernel=None):
    if PSS_kernel is None:
        PSS_kernel = tda.PersistenceScaleSpaceKernel(bandwidth=1.)
        X_fit = PSS_kernel.fit([np.asarray(X)])
    Y_fit = PSS_kernel.transform([np.asarray(Y)])
    return Y_fit[0][0], PSS_kernel

def cross_validate_10_fold(model, data_dir, distance_matrix=None):
    best_test_acc = 0
    best_train_acc = 0
    worst_test_acc = 2 # larger than 100%
    worst_train_acc = 2
    for y_loc in range(8):
        x_train, y_train, x_test, y_test = generate_test_and_training_set_raw(data_dir, test_num=y_loc)
        if distance_matrix is not None:
            x_train, x_test = split_distance_matrix(distance_matrix, test_num=y_loc)
        else:
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

def convert_ripser_to_np(dir, dim='dim0'):
    feature_matrix = []
    output = [dir + f for f in listdir(dir) if isfile(join(dir, f)) and f.find(dim) != -1]
    for count, out in enumerate(output):
        feature_matrix.append([])
        with open(out) as fp:
            for line in fp:
                parsed = line.strip().split(" ")
                feature_matrix[count].append([float(parsed[0]), float(parsed[1])])
    return np.asarray(feature_matrix)

def compute_PSS_gram_matrix(dim="dim0", np_checkpoints=None):
    if np_checkpoints is not None:
        try:
            gram_matrix = np.load(np_checkpoints + "pss_kernel_" + dim + ".npy")
            print("Found saved matrix for PSS kernel in " + dim)
            return gram_matrix
        except IOError:
            print("Couldn't find matrix for PSS kernel in " + dim + ". Generating...")
    pds = convert_ripser_to_np(RIPSER_OUTPUT, dim=dim)
    gram_matrix = np.ones((80, 80))
    for count1, pd1 in enumerate(pds):
        print("Starting PD " + str(count1))
        PSS_kernel = None
        for count2, pd2 in enumerate(pds):
            gram_matrix[count1, count2], PSS_kernel = persistence_scale_kernel(pd1, pd2, PSS_kernel)
    np.save(NP_CHECKPOINTS + "pss_kernel_" + dim, gram_matrix)
    return gram_matrix

def optimize_C(png_dir, kernel=None, gram_matrix=None, description="SVM optimization"):
    c_values = [c / 10 for c in range(2, 18, 5)]
    best_c = -1
    best_acc = [0]
    for c in c_values:
        if kernel is None:
            model = svm.SVC(C=c, gamma='scale')
        elif kernel == 'precomputed' and gram_matrix is not None:
            model = svm.SVC(C=c, kernel=kernel)
        accuracies = cross_validate_10_fold(model=model, data_dir=png_dir, distance_matrix=gram_matrix)
        if accuracies[0] > best_acc[0]:
            best_c = c
            best_acc = accuracies
    print_training_output(description, accuracies, C=c)
    return best_c


def main():
    ###Naive with raw images
    optimize_C(PNG_DIR, description="Raw image classification - RBF kernel with gamma=\'scale\'")

    ###Dim0 persistence scale kernel
    gram_matrix_pss_dim0 = compute_PSS_gram_matrix(dim="dim0", np_checkpoints=NP_CHECKPOINTS)
    optimize_C(PNG_DIR, kernel='precomputed', gram_matrix=gram_matrix_pss_dim0, description="PSS kernel SVM in dim0")

    ###Dim1 persistence scale kernel
    gram_matrix_pss_dim1 = compute_PSS_gram_matrix(dim="dim1", np_checkpoints=NP_CHECKPOINTS)
    optimize_C(PNG_DIR, kernel='precomputed', gram_matrix=gram_matrix_pss_dim1, description="PSS kernel SVM in dim1")

    ###combined pss kernel
    gram_matrix_pss = gram_matrix_pss_dim0 + gram_matrix_pss_dim1
    optimize_C(PNG_DIR, kernel='precomputed', gram_matrix=gram_matrix_pss, description="PSS kernel SVM with dim0 and dim1")


if __name__ == '__main__':
    main()