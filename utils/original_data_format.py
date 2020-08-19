import pandas as pd
import os

def load_data(table):
    ext = os.path.splitext(table)

    if ext == '.csv':
        df = pd.read_csv(table)
    elif ext == '.tsv':
        df = pd.read_csv(table, sep='\t')
    elif ext == '.xlsx':
        df = pd.read_excel(table)
    else:
        print('Your file format {} is not supported!'.format(ext))
        raise ValueError()

    return df


def uda_formater(raw_df, content, label, ids, mode='r', save_path=None):
    '''
    '''
    if type(ids) == str:
        raw_df[ids] = raw_df[ids].dtype(str)
        raw_df['id'] = raw_df[label].str.cat(raw_df[ids], sep='_')

    elif type(ids) == list:
        for i in ids:
            raw_df[i] = raw_df[i].dtype(str)
        
        for n, ii in enumerate(ids):
            if n == 0:
                raw_df['id'] = raw_df[label].str.cat(raw_df[ii], sep='_')

            else:
                raw_df['id'] = raw_df['id'].str.cat(raw_df[ii], sep='_')

    save_df = raw_df.rename(columns={content:'content', label:'label'})
    save_df = save_df[['content', 'label', 'id']]
    
    if mode == 'r':
        return save_df

    elif mode == 'o':
        save_df.to_csv(os.path.join(save_path, 'uda_fomat_data.tsv'), sep='\t')


def train_test_split(df, train_ratio=None, num_train=None, test_ratio=None, num_test=None):
    df = df.sample(frac=1)
    df = df.sample(frac=1)

    if train_ratio != None & num_train == None & test_ratio == None & num_test == None:
        l = int(len(df) * train_ratio)
        train = df[:l]
        test = df[l:]

    elif num_train != None & train_ratio == None & test_ratio == None & num_test == None:
        train = df[:num_train]
        test = df[num_train:]

    elif test_ratio != None & num_train == None & train_ratio == None & num_test == None:
        l = int(len(df) * test_ratio)
        test = df[:l]
        train = df[l:]

    elif num_test != None & train_ratio == None & test_ratio == None & num_train == None:
        test = df[:num_test]
        train = df[num_test:]

    