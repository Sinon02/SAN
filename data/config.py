import os

DATA_PATH = os.path.dirname(os.path.realpath(__file__))
DATASET_NAME = 'CROHME'
IMAGE_PATH = {'CROHME': 'off_image_train', 'HME100K': 'train_images'}
TRAIN_CAPTION_PATH = {'CROHME': 'train_caption.txt', 'HME100K': 'train_labels.txt'}
TEST_CAPTION_PATH = {'CROHME': 'test_caption.txt', 'HME100K': 'test_labels.txt'}
DATASET_PATH = os.path.join(DATA_PATH, DATASET_NAME)
