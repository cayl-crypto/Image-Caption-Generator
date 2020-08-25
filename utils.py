import os
import sys
from tqdm import tqdm
import requests 
from zipfile import ZipFile


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    print("Downloading... " + url + " to " + save_path)
    with open(save_path, 'wb') as fd:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
            fd.write(chunk)


def extract_and_remove_zip_file(full_path, target_dir):
    with ZipFile(full_path, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(path=target_dir)
    os.remove(full_path)

def download_dataset(Annotation=False, Train=False, Val=False):

    dataset_folder = 'Datasets'
    try:
        os.mkdir(dataset_folder)
        print("Directory ", dataset_folder, " Created ")
    except FileExistsError:
        print("Directory ", dataset_folder, " already exists")

    if Annotation:
        ## Download Annotations
        ann_dataset_url_path, ann_dataset_save_path  = get_mscoco_captioning_2017_annotations_path()
        ann_save_path = dataset_folder + "/" + ann_dataset_save_path
        #download_url(url=ann_dataset_url_path, save_path=ann_save_path)
        extract_and_remove_zip_file(full_path=ann_save_path, target_dir=dataset_folder)

    if Train:
        ## Download Train Images
        train_dataset_url_path, train_dataset_save_path  = get_mscoco_captioning_train_2017_images_path()
        train_save_path = dataset_folder+ "/" + train_dataset_save_path
        download_url(url=train_dataset_url_path, save_path=train_save_path)
        extract_and_remove_zip_file(full_path=train_save_path, target_dir=dataset_folder)

    if Val:
        ## Download Val Images
        val_dataset_url_path, val_dataset_save_path  = get_mscoco_captioning_val_2017_images_path()
        val_save_path = dataset_folder + "/" + val_dataset_save_path
        download_url(url=val_dataset_url_path, save_path=val_save_path)
        extract_and_remove_zip_file(full_path=val_save_path, target_dir=dataset_folder)
    
    

def get_mscoco_captioning_train_2017_images_path():
    # returns download url of image captioning 2017 train images
    return "http://images.cocodataset.org/zips/train2017.zip","train2017.zip"
   

def get_mscoco_captioning_val_2017_images_path():
    # returns download url of image captioning 2017 validation images
    return "http://images.cocodataset.org/zips/val2017.zip","val2017.zip"

def get_mscoco_captioning_2017_annotations_path():
    # returns download url of image captioning 2017 annotations
    return "http://images.cocodataset.org/annotations/annotations_trainval2017.zip","annotations2017.zip"

