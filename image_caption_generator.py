from utils import load_mscoco_annotations_val
import torch
from create_vocabulary import Voc
from tqdm import tqdm

# Download dataset is done

# Load dataset is done

val_captions, val_image_names = load_mscoco_annotations_val()

# TODO Tokenize captions
voc = Voc(name="Vocabulary")

for caption in tqdm(val_captions):

    voc.addCaption(caption=caption)

voc.trim(min_count=13)







