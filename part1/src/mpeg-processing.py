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
    barcodes.sort()
    try:
        distance_matrix = np.load("../data/hera_output/" + distance + "-" + dim + ".npy")
        print("Found saved matrix for " + distance + " " + dim)
        return distance_matrix
    except IOError:
        print("Couldn't find matrix for " + distance + " " + dim + ". Generating...")
    distance_matrix = np.ones((80,80))
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
    np.save("../data/hera_output/" + distance + "-" + dim, distance_matrix)
    return distance_matrix

def main():
    if len(sys.argv) == 1:
        print("Select a routine to run!")
        exit(1)
    if sys.argv[1] == '-all':
        sys.argv.append('-gen-png')
        sys.argv.append('-gen-point-cloud')
        sys.argv.append('-run-ripser')
        sys.argv.append('-run-hera')
    if '-gen-png' in sys.argv :
        generate_png('../data/originals/')
    if '-gen-point-cloud' in sys.argv:
        create_point_cloud_files('../data/original_png/')
    if '-run-ripser' in sys.argv:
        run_ripser('../ripser/ripser', '../data/point_clouds/', '../data/ripser_outputs/')
    if '-run-hera' in sys.argv:
        run_hera(distance="wasserstein", dim="dim0")
        run_hera(distance="wasserstein", dim="dim1")
        run_hera(distance="geom_bottleneck", dim="dim0")
        run_hera(distance="geom_bottleneck", dim="dim1")
if __name__ == '__main__':
    main()
