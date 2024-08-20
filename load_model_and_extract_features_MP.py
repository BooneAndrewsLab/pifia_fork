from model.extract_features_MP import *
from model import models
from PIL import Image
import pandas as pd
import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-path', type=str,
                    help='Path to file containing single-cell information. Required columns are: ORF, Name, Strain ID, '
                          'Center_X, Center_Y and Image_Path.')
parser.add_argument('-n', '--num-classes', type=int, default=4049,
                    help='Number of classes in the training set. Default is 4049.')
parser.add_argument('-c', '--crop-size', type=int, default=64, help="Single-cell crop size. Default is 64.")
parser.add_argument('-o', '--output-dir', type=str, help="Folder where to save single-cell feature profiles.")
parser.add_argument('-m', '--marker', type=str, default='', help="Specify marker name. Optional only.")

args = parser.parse_args()

def main():
    # Load the model
    model = models.pifia_network(args.num_classes,
                                 k=1,
                                 num_features=64,
                                 dense1_size=128,
                                 last_block=True)

    model.load_weights('model/pretrained_weights/pifia_weights_i0')

    # Load single-cell data
    df = pd.read_csv(args.input_path, keep_default_na=False,
                     usecols=['ORF', 'Name', 'Strain ID', 'Center_X', 'Center_Y', 'Image_Path'])
    gene_map = {}
    path_to_cell_array = {}
    for orf, name, strainid, x, y, img_path in df.itertuples(index=False):
        # if not name:
        #     gene = orf
        # else:
        #     gene = name
        # if gene not in gene_map:
        #     gene_map[gene] = []

        if strainid not in gene_map:
            gene_map[strainid] = []

        center_x = int(x)
        center_y = int(y)
        loc_left = center_x - args.crop_size/2
        loc_upper = center_y - args.crop_size/2
        loc_right = center_x + args.crop_size/2
        loc_lower = center_y + args.crop_size/2

        try:
            full_image = Image.open(img_path)
        except FileNotFoundError:
            continue
        cell_crop = full_image.crop((loc_left, loc_upper, loc_right, loc_lower))

        cell_path = f'{img_path}_X_{center_x}_Y_{center_y}'
        gene_map[strainid].append(cell_path)
        path_to_cell_array[cell_path] = np.array(cell_crop)

    if args.marker != 'Hta2':
        seek_ch = 0
    else:
        seek_ch = 1

    for protein_name, cells in gene_map.items():

        protein_features, protein_images = get_features_from_protein(protein_name, gene_map, model, path_to_cell_array,
                                                                     seek_ch, average=False)
        if args.marker:
            output_path = f'{args.output_dir}/{args.marker}_{protein_name}_scFP.npy'
        else:
            output_path = f'{args.output_dir}/{protein_name}_scFP.npy'
        np.save(output_path, protein_features)
        print(f'{args.marker} {protein_name} features: {protein_features.shape}')


if __name__ == "__main__":
    main()
