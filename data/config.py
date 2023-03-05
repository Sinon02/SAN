import os

DATA_PATH = os.path.dirname(os.path.realpath(__file__))
DATASET_NAME = 'CROHME'
IMAGE_PATH = {
    'train': {
        'CROHME': 'off_image_train',
        'HME100K': 'train_images'
    },
    'test': {
        'CROHME': 'off_image_test',
        'HME100K': 'testimages'
    }
}
CAPTION_PATH = {
    'train': {
        'CROHME': 'train_caption.txt',
        'HME100K': 'train_labels.txt'
    },
    'test': {
        'CROHME': 'test_caption.txt',
        'HME100K': 'test_labels.txt'
    }
}
DATASET_PATH = os.path.join(DATA_PATH, DATASET_NAME)
