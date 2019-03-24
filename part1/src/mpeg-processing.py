"""
CS 6170 - Spring 2019
Nithin Chalapathi
"""
import numpy as np
import imageio as imio
import random
from os import listdir
from os.path import isfile, join
from PIL import Image
import sys
import subprocess as sp
import re
from matplotlib import pyplot as plt
from sklearn import manifold

def is_white(pixel):
    return pixel != 0
def is_black(pixel):
    return pixel == 0

def has_black_neighbors(x, y, img_np):
    for i in range(-1,2):
        for j in range(-1,2):
            if is_black(img_np[x+i, y+j]):
                return True
    return False

"""
Takes a relative file path to a MPEG-7 png file. Returns a list of the location of white pixels that have a black neighbor.
"""
def sample(png_loc):
    mpeg = imio.imread(png_loc)
    point_locations = []
    dims = mpeg.shape
    for x in range(1, dims[0]-1):
        for y in range(1, dims[1]-1):
            pixel = mpeg[x,y]
            if is_white(pixel) and has_black_neighbors(x,y,mpeg):
                point_locations.append((y,x))
    return point_locations

def generate_png(dir_gif):
    print("Creating pngs...")
    gifs = [dir_gif + f for f in listdir(dir_gif) if isfile(join(dir_gif, f))]
    for gif in gifs:
        print(gif)
        img = Image.open(gif)
        img.save(gif.replace('gif', 'png').replace('originals', 'original_png'), 'png', optimize=True, quality=True)

def create_point_cloud_files(img_dir):
    print("Creating point clouds...")
    pictures = [img_dir + f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    for img in pictures:
        print(img)
        points = sample(img)
        points_file = open(img.replace('original_png', 'point_clouds').replace('png', 'txt'), 'w')
        while len(points) > 1000: #sample boundary points randomly to prevent overly large boundaries
            del points[random.randint(0, len(points)-1)]
        for pt in points:
            points_file.write(str(pt[0]) + " " + str(pt[1]) + "\n")
        points_file.close()

def run_ripser(ripser_loc, point_cloud_dir, output_loc):
    print('Running ripser...')
    clouds = [point_cloud_dir + f for f in listdir(point_cloud_dir) if isfile(join(point_cloud_dir, f))]
    ripser_cmd = [ripser_loc, '--format', 'point-cloud']
    for cloud in clouds:
        ripser_cmd.append(cloud)
        print(ripser_cmd)
        finished_ripser = sp.run(ripser_cmd, stdout=sp.PIPE, check=True, encoding='utf-8')
        dims = finished_ripser.stdout.split('persistence intervals in dim 1:\n')
        with open(cloud.replace(point_cloud_dir, output_loc).replace('.txt', '-dim0.txt'), 'w') as f:
            for line in dims[0].split('\n'):
                if line[0:2] == ' [':
                    line_split = re.split("[\[,)]", line)
                    f.write(line_split[1])
                    f.write(" ")
                    if line_split[2] == " ":
                        line_split[2] = "100" #high persistence to indicate the feature doesn't die
                    f.write(line_split[2])
                    f.write("\n")
        with open(cloud.replace(point_cloud_dir, output_loc).replace('.txt', '-dim1.txt'), 'w') as f:
            for line in dims[1].split('\n'):
                if line[0:2] == ' [':
                    line_split = re.split("[\[,)]", line)
                    f.write(line_split[1])
                    f.write(" ")
                    f.write(line_split[2])
                    f.write("\n")
        del ripser_cmd[-1]


def run_hera(PH_barcode_loc="../data/ripser_outputs/", point_cloud_loc="../data/point_clouds/", distance="geom_bottleneck", dim="dim0"):
    print("Running hera with " + distance + " on " + dim + "...")
    barcodes = [PH_barcode_loc + f.replace(".txt", "-" + dim + ".txt") for f in listdir(point_cloud_loc) if
                isfile(join(point_cloud_loc, f))]
    barcodes = sorted(barcodes, key=lambda s: s.casefold())
    try:
        distance_matrix = np.load("../data/hera_output/" + distance + "-" + dim + ".npy")
        print("Found saved matrix for " + distance + " " + dim)
        return distance_matrix
    except IOError:
        print("Couldn't find matrix for " + distance + " " + dim + ". Generating...")
    distance_matrix = np.ones((80,80), dtype=np.float64)
    if distance == "geom_bottleneck":
        executable = distance + "/build/" + "bottleneck_dist"
    else:
        executable = "geom_matching/wasserstein/build/wasserstein_dist"
    hera_loc = "../hera/" + executable
    for counter1, b1 in enumerate(barcodes):
        for counter2, b2 in enumerate(barcodes):
            hera_cmd = [hera_loc, b1, b2]
            finished_hera = sp.run(hera_cmd, stdout=sp.PIPE, check=True,  encoding="utf-8")
            distance_matrix[counter1, counter2] = float(finished_hera.stdout.strip())
    #Correcting float precision errors
    for i in range(80):
        for j in range(80):
            if distance_matrix[i, j] != distance_matrix[j, i]:
                if distance_matrix[i,j] - distance_matrix[j,i] <=1e-4:
                    distance_matrix[i, j] = distance_matrix[j, i]
                else:
                    print("Float precision error")
                    print(barcodes[i] + " " + barcodes[j])
                    print(distance_matrix[i,j])
                    print(distance_matrix[j,i])
    np.save("../data/hera_output/" + distance + "-" + dim, distance_matrix)
    return distance_matrix

def plot_mds(distance_matrix, colors8, description, fig, pos):
    fig.add_subplot(pos)
    embedding = manifold.MDS(n_components=2, dissimilarity='precomputed')
    transformed = embedding.fit_transform(distance_matrix)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=colors8)
    plt.title(description)

def plot_tsne(distance_matrix, colors8, description, fig, pos):
    fig.add_subplot(pos)
    embedding = manifold.TSNE(n_components=2, metric='precomputed')
    transformed = embedding.fit_transform(distance_matrix)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=colors8)
    plt.title(description)

"""
apple = blue 
bird = green
bone = red 
children = cyan
face = magenta
fork = yellow
horseshoe = black
pencil = grey
spoon = pink
watch = orange
"""
def get_color_vector():
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'grey', 'pink', 'orange']
    colors8 = []
    for color in colors:
        for _ in range(8):
            colors8.append(color)
    return colors8

def visualize_original_mpeg_png(mpeg_loc):
    print("Visualizing 80 mpeg images using t-SNE and MDS")
    pictures = [mpeg_loc + f for f in listdir(mpeg_loc) if isfile(join(mpeg_loc, f))]
    pictures = sorted(pictures, key=lambda s: s.casefold())
    dataset = []
    for counter, pic in enumerate(pictures):
        im = Image.open(pic)
        im = im.resize((256,256), Image.ANTIALIAS)
        mpeg = np.asarray(im)
        dataset.append(mpeg.flatten())

    colors8 = get_color_vector()
    #TSNE
    embedding = manifold.TSNE(n_components=2)
    transformed = embedding.fit_transform(dataset)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=colors8)
    plt.title("Raw images scaled to 256x256 - t-SNE")
    plt.show()

    #MDS
    embedding = manifold.MDS(n_components=2)
    transformed = embedding.fit_transform(dataset)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=colors8)
    plt.title("Raw images scaled to 256x256 - MDS")
    plt.show()

def main():
    if len(sys.argv) == 1:
        print("Select a routine to run!")
        exit(1)
    if sys.argv[1] == '-all':
        sys.argv.append('-gen-png')
        sys.argv.append('-gen-point-cloud')
        sys.argv.append('-run-ripser')
        sys.argv.append('-run-hera-and-plot')
        sys.argv.append('-vis-mpeg')
    if '-gen-png' in sys.argv :
        generate_png('../data/originals/')
    if '-gen-point-cloud' in sys.argv:
        create_point_cloud_files('../data/original_png/')
    if '-run-ripser' in sys.argv:
        run_ripser('../ripser/ripser', '../data/point_clouds/', '../data/ripser_outputs/')
    if '-run-hera-and-plot' in sys.argv:
        fig_was = plt.figure(1)
        fig_was.suptitle("Wasserstein Graphs")
        colors8 = get_color_vector()
        distance_matrix = run_hera(distance="wasserstein", dim="dim0")
        plot_mds(distance_matrix, colors8, "Wasserstein Distance in Dim 0 (MDS)", fig_was, 221)
        plot_tsne(distance_matrix, colors8, "Wasserstein Distance in Dim 0 (t-SNE)", fig_was, 222)

        distance_matrix = run_hera(distance="wasserstein", dim="dim1")
        plot_mds(distance_matrix, colors8, "Wasserstein Distance in Dim 1 (MDS)", fig_was, 223)
        plot_tsne(distance_matrix, colors8, "Wasserstein Distance in Dim 1 (t-SNE)", fig_was, 224)
        plt.show()

        plt.close(fig_was)
        fig_bottle = plt.figure(2)
        fig_bottle.suptitle("Bottleneck Graphs")
        distance_matrix = run_hera(distance="geom_bottleneck", dim="dim0")
        plot_mds(distance_matrix, colors8, "Bottleneck Distance in Dim 0 (MDS)", fig_bottle, 221)
        plot_tsne(distance_matrix, colors8, "Bottleneck Distance in Dim 0 (t-SNE)", fig_bottle, 222)

        distance_matrix = run_hera(distance="geom_bottleneck", dim="dim1")
        plot_mds(distance_matrix, colors8, "Bottleneck Distance in Dim 1 (MDS)", fig_bottle, 223)
        plot_tsne(distance_matrix, colors8, "Bottleneck Distance in Dim 1 (t-SNE)", fig_bottle, 224)
        plt.show()
    if '-vis-mpeg' in sys.argv:
        visualize_original_mpeg_png("../data/original_png/")


if __name__ == '__main__':
    main()
