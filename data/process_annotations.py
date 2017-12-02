import os
import pandas as pd
import numpy as np
np.random.seed(0)

'''
Go through the annotation_final.csv, process labels for the audio data
'''
# testing
annotation_path = 'path_to_annotation_file/annotations_final.csv'

'''
Some tags are considered to be redundant, so it seems reasonable to do some cleanup.
Thanks to https://github.com/keunwoochoi/magnatagatune-list :)
'''
def _merge_redundant_tags(filename):
    synonyms = [['beat', 'beats'],
                ['chant', 'chanting'],
                ['choir', 'choral'],
                ['classic', 'clasical', 'classical'],
                ['drum', 'drums'],
                ['electronic', 'electro', 'electronica', 'electric'],
                ['fast', 'fast beat', 'quick'],
                ['female', 'female singer', 'female singing', 'female vocal', 'female vocals', 'female voice', 'woman', 'woman singing', 'women'],
                ['flute', 'flutes'],
                ['guitar', 'guitars'],
                ['hard', 'hard rock'],
                ['harpsichord', 'harpsicord'],
                ['heavy', 'heavy metal', 'metal'],
                ['horn', 'horns'],
                ['indian', 'india'],
                ['jazz', 'jazzy'],
                ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
                ['no beat', 'no drums'],
                ['no vocal', 'no singing', 'no singer','no vocals', 'no voice', 'no voices', 'instrumental'],
                ['opera', 'operatic'],
                ['orchestra', 'orchestral'],
                ['quiet', 'silence'],
                ['singer', 'singing'],
                ['space', 'spacey'],
                ['string', 'strings'],
                ['synth', 'synthesizer'],
                ['violin', 'violins'],
                ['vocal', 'vocals', 'voice', 'voices'],
                ['weird', 'strange']]
    
    synonyms_correct = [synonyms[i][0] for i in range(len(synonyms))]
    synonyms_redundant = [synonyms[i][1:] for i in range(len(synonyms))]
    
    df = pd.read_csv(filename, delimiter='\t')
    new_df = df.copy()
    
    for i in range(len(synonyms)):
        for j in range(len(synonyms_redundant[i])):
            redundant_df = df[synonyms_redundant[i][j]]
            new_df[synonyms_correct[i]] = (new_df[synonyms_correct[i]] + redundant_df) > 0
            new_df[synonyms_correct[i]] = new_df[synonyms_correct[i]].astype(int)
            new_df.drop(synonyms_redundant[i][j] ,1, inplace=True)
    return new_df 


'''
read csv file, take top n tags, split data samples into train/validation/test
data split : 
    train : 0 - b
    validation : c
    test : d - f
'''
def _load_annotations(filename, n_top=50, n_audios_per_shard=100, merge=True):
    if (merge) : 
        df = _merge_redundant_tags(filename)
    else :
        df = pd.read_csv(filename, delimiter='\t')
    
    top50 = (df.drop(['clip_id','mp3_path'], axis=1)
            .sum()
            .sort_values()
            .tail(50)
            .index
            .tolist())
    
    df = df[top50 + ['clip_id', 'mp3_path']]

    # remove rows with all 0 labels
    df = df.loc[~(df.loc[:, top50] == 0).all(axis=1)]
    
    def split_by_directory(mp3_path):
        # example
        # df['mp3_path'] = "f/american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.mp3"
        directory = mp3_path.split('/')[0]
        part = int(directory, 16)

        if part in range(12):
            return 'train'
        elif part == 12 : 
            return 'val'
        elif part in range(13, 16):
            return 'test'
    
    df['split'] = df['mp3_path'].apply(lambda mp3_path: split_by_directory(mp3_path))
    
    for split in ['train', 'val', 'test']:
        n_audios = sum(df['split'] == split) # count how many audios are in each split
        n_shards = n_audios // n_audios_per_shard # count number of shards in each split
        n_remainders = n_audios % n_audios_per_shard 
        
        '''
        effect of shuffling each of the splitted data
        == assigning which shard/group this friend will be in
        ''' 
        shards = np.tile(np.arange(n_shards), n_audios_per_shard)
        shards = np.concatenate([shards, np.arange(n_remainders)])
        shards = np.random.permutation(shards)
        
        df.loc[df['split'] == split, 'shard'] = shards
        
    df['shard'] = df['shard'].astype(int) 
    
    return df


'''
1. Create entry for each segment in the mp3 file (10 segments/mp3file) ...could have come up with a better/smarter way...
2. Split the dataset into train/val/test and save them in separate csv file

output:
annotations_final_train.csv
annotations_final_val.csv
annotations_final_test.csv
'''

def _create_and_split_annotations_for_segments(filename, n_top=50, n_audios_per_shard=100, merge=True):
    df = _load_annotations(filename, n_top, n_audios_per_shard, merge)

    # for each df entry, copy it into a new row and add column new_df['segment_id'] 
    # do it for each segment (10 per audio file)
    # insert it into the new df
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()

    counter = 0
    for index, row in df.iterrows():
        if counter % 100 == 0:
            print "currently at row {}".format(counter)

        # check if the npy file is existing..could have been not created since the length is less than 29.1
        # mp3_to_npy_path = filename.split("")
        npypath = 'mp3_to_npy/' + row['mp3_path'].split(".")[0] + '.npy'

        if not os.path.exists(npypath):
            print ("npy file not found...pass")
            continue
        
        if row['split'] == 'train':
            for i in range (10):
                row['segment_id'] = i
                train_df = train_df.append(row, ignore_index=True)
        elif row['split'] == 'val':
            for i in range (10):
                row['segment_id'] = i
                val_df = val_df.append(row, ignore_index=True)
        elif row['split'] == 'test':
            for i in range (10):
                row['segment_id'] = i
                test_df = test_df.append(row, ignore_index=True)
        
        counter += 1
    
    print "iterated through {} rows".format(counter)

    print "train data size is {}".format(train_df.shape[0])
    print "val data size is {}".format(val_df.shape[0])
    print "test data size is {}".format(test_df.shape[0])
    
    # convert type from float to int and save the new dataframes to csv file
    # for train_df
    tags = [col for col in train_df.columns if col not in ['mp3_path', 'split']]
    for tag in tags :
        train_df[tag] = train_df[tag].astype(int)

    train_df.to_csv(filename.split("/")[-1].split(".")[0] + '_train.csv', encoding='utf-8')
    
    # for val_df
    tags = [col for col in val_df.columns if col not in ['mp3_path', 'split']]
    for tag in tags :
        val_df[tag] = val_df[tag].astype(int)

    val_df.to_csv(filename.split("/")[-1].split(".")[0] + '_val.csv', encoding='utf-8')
    
    # for test_df 
    tags = [col for col in test_df.columns if col not in ['mp3_path', 'split']]
    for tag in tags :
        test_df[tag] = test_df[tag].astype(int)

    test_df.to_csv(filename.split("/")[-1].split(".")[0] + '_test.csv', encoding='utf-8')


if __name__ == "__main__":
    _create_and_split_annotations_for_segments(annotation_path)

