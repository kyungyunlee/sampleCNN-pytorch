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
Tag organization by https://github.com/keunwoochoi/magnatagatune-list
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

''' There are a lot of tags, so reduce it to top N popular tags '''
def reduce_to_N_tags(filename, base_dir, n_tops=50, merge=True):
    if merge:
        df = _merge_redundant_tags(filename)
    else :
        df = pd.read_csv(filename, delimiter='\t')
    print (df.drop(['clip_id', 'mp3_path'], axis=1).sum(axis=0).sort_values())
    topN = df.drop(['clip_id', 'mp3_path'], axis=1).sum(axis=0).sort_values().tail(n_tops).index.tolist()[::-1]
    print (len(topN), topN)
    taglist_f = open(str(n_tops) + '_tags.txt', 'w')
    for tag in topN:
        taglist_f.write(tag+'\n')
    taglist_f.close()

    # df = df[topN + ['clip_id', 'mp3_path']]
    df = pd.concat([df.loc[:,topN], df.loc[:,'clip_id'], df.loc[:,'mp3_path']], axis=1)

    # remove rows with all 0 labels
    df = df.loc[~(df.loc[:, topN] == 0).all(axis=1)]
    print (df.shape)
    # save new csv file
    new_filename = base_dir + str(n_tops) + '_tags_' + filename.split('/')[-1]
    df.to_csv(new_filename, sep='\t', encoding='utf-8', index=False)
    return new_filename

''' Split into train/val/test '''
def split_data(filename, base_dir, ratio=0.2):
    df = pd.read_csv(filename, delimiter='\t')
    data_len = df.shape[0]
    print ("Data shape {}".format(df.shape))
    
    test_len = int (data_len * ratio)
    train_valid_len = data_len - test_len
    valid_len = int(train_valid_len * ratio)
    train_len = train_valid_len - valid_len
    print ("Train %d, valid %d, test %d"%(train_len, valid_len, test_len))
    
    # add headers to all files
    test_df = df.iloc[train_valid_len:]
    valid_df = df.iloc[train_len : train_valid_len]
    train_df = df.iloc[:train_len]
    
    # save each test, valid, train files
    f = filename.split('/')[-1]
    test_df.to_csv(base_dir + 'test_' + f, sep='\t',index=False)
    valid_df.to_csv(base_dir + 'valid_' + f, sep='\t',index=False)
    train_df.to_csv(base_dir + 'train_' + f, sep='\t', index=False)


if __name__ == "__main__":
    base_dir = '/media/bach4/kylee/sampleCNN-data/'
    data_dir = '/media/bach4/dataset/'
    annot_file = 'annotations_final.csv'
    new_csvfile = reduce_to_N_tags(data_dir + annot_file, base_dir)
    split_data(new_csvfile, base_dir)

