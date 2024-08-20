import numpy as np
from model import models
from model.extract_features import *

num_classes = 4049
model = models.pifia_network(num_classes,
                             k=1,
                             num_features=64,
                             dense1_size=128,
                             last_block=True)

model.load_weights('model/pretrained_weights/pifia_weights_i0')
# with open('model/pretrained_weights/pifia_weights_i0', 'rb') as f:
#     model.load_weights(f)

labels_dict = np.load('data/protein_to_files_dict_toy_dataset.npy', allow_pickle=True)[()]
protein_name = 'NUP2'
protein_features, protein_images = get_features_from_protein(protein_name, labels_dict, model,
                                                             average=False, subset='test')
print("protein_features", protein_features.mean())
