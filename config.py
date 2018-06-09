BASE_DIR = '/media/bach4/kylee/sampleCNN-data/'
DATA_DIR = '/media/bach4/dataset/'
MTT_DIR = '/media/bach1/dataset/MaganatagatuePub/mp3/'
AUDIO_DIR = DATA_DIR + 'pubMagnatagatune_mp3s_to_npy/'
ANNOT_FILE = DATA_DIR + 'annotations_final.csv'
LIST_OF_TAGS = BASE_DIR + '50_tags.txt'


# audio params 
SR = 22050
NUM_SAMPLES = 59049
NUM_TAGS = 50


# train params 
BATCH_SIZE = 64 
LR = 0.008
DROPOUT_RATE = 0.5
NUM_EPOCHS = 100
