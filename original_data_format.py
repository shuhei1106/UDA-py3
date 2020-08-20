import pandas as pd
import os
from absl import app, flags, logging

FLAGS = flags.FLAGS

'''
General Parameters
'''
flags.DEFINE_string('sup_data', './data/original/v2/raw_data.csv',
                    'Your labeled dataset. The format should be a table file such as csv, tsv, excel, etc., '
                    'with at least a sentence, id, and label in it.')

flags.DEFINE_string('unsup_data', './data/original/unsup_32m/all_repo_32m.csv',
                    'Your labeled dataset. The format should be a table file such as csv, tsv, excel, etc., '
                    'with at least a sentence, id, and label in it.')

flags.DEFINE_string('save_dir', './data/dataset/v2',
                    'Destination of the converted dataset in UDA format.')


'''
Supervised Parameters
'''
flags.DEFINE_string('sup_cont_col', 'text',
                    'A column in a dataframe that contains the text you want to classify'
                    'Input is string only.')

flags.DEFINE_string('label_col', 'label',
                    'The column containing the label in the dataframe'
                    'Input is string only.')

flags.DEFINE_string('sup_id_col', 'facility code,accession number',
                    'The column containing the id in the dataframe'
                    'If there is more than one, please separate them with a comma (,).'
                    'Input is string.')

flags.DEFINE_enum('smethod_arg', 'train_ratio', ['train_ratio', 'num_train', 'test_ratio', 'num_test'],
                    'How to divide the data when you hold out.'
                    'train_ratio : Determine the percentage of train data.'
                    'num_train : Determine the number of train data.'
                    'test_ratio : Determine the percentage of test data.'
                    'num_test : Determine the number of test data.')

flags.DEFINE_float('split_arg', 0.8,
                    'The value for splitting by the method selected by smthod_arg.')


'''
Unsupervised Parameters
'''
flags.DEFINE_string('unsup_cont_col', 'FINDING,DIAGNOSIS',
                    'A column in a dataframe that contains the text you want to classify'
                    'Input is string only.')

flags.DEFINE_string('unsup_id_col', 'FACILITY_CODE,ACCESSION_NUMBER',
                    'The column containing the id in the dataframe'
                    'Input is string or list only.')



def load_data(table):
    ext = os.path.splitext(table)

    if ext[-1] == '.csv':
        df = pd.read_csv(table)
    elif ext[-1] == '.tsv':
        df = pd.read_csv(table, sep='\t')
    elif ext[-1] == '.xlsx':
        df = pd.read_excel(table)
    else:
        print('Your file format {} is not supported!'.format(ext))
        raise ValueError()

    return df


def uda_formater(raw_df, content, label, ids, mode='o', save_dir=None):
    '''
    '''
    if ',' not in ids:
        raw_df[ids] = raw_df[ids].astype(str)
        raw_df['id'] = raw_df[label].str.cat(raw_df[ids], sep='_')

    elif ',' in ids:
        ids = ids.split(',')
        for i in ids:
            raw_df[i] = raw_df[i].astype(str)
        
        for n, ii in enumerate(ids):
            if n == 0:
                raw_df['id'] = raw_df[label].str.cat(raw_df[ii], sep='_')

            else:
                raw_df['id'] = raw_df['id'].str.cat(raw_df[ii], sep='_')

    save_df = raw_df.rename(columns={content:'content', label:'label'})
    save_df = save_df[['content', 'label', 'id']]
    
    if mode == 'o':
        return save_df

    elif mode == 'w':
        save_df.to_csv(os.path.join(save_dir, 'uda_fomat_data.tsv'), sep='\t')


def unsup_uda_formater(unsup_data, content, ids, mode='o', save_dir=None):
    '''
    '''
    unsup_data['label'] = 'unsup'

    if ',' not in ids:
        unsup_data[ids] = unsup_data[ids].astype(str)
        unsup_data['id'] = unsup_data['label'].str.cat(unsup_data[ids], sep='_')

    elif ',' in ids:
        ids = ids.split(',')
        for i in ids:
            unsup_data[i] = unsup_data[i].astype(str)
        
        for n, ii in enumerate(ids):
            if n == 0:
                unsup_data['id'] = unsup_data['label'].str.cat(unsup_data[ii], sep='_')

            else:
                unsup_data['id'] = unsup_data['id'].str.cat(unsup_data[ii], sep='_')

    if ',' in content:
        contents = content.split(',')
        for c in contents:
            unsup_data[c] = unsup_data[c].astype(str)
        
        for n, cc in enumerate(contents):
            if n == 0:
                unsup_data['content'] = unsup_data[cc]

            else:
                unsup_data['content'] = unsup_data['content'].str.cat(unsup_data[cc], sep='')
    else:
        unsup_data = unsup_data.rename(columns={content:'content'})

    save_df = unsup_data[['content', 'label', 'id']]
    
    if mode == 'o':
        return save_df

    elif mode == 'w':
        save_df.to_csv(os.path.join(save_dir, 'unsup_uda_fomat_data.tsv'), sep='\t')


def train_test_split(df, mode, split_value):
    df = df.sample(frac=1)
    df = df.sample(frac=1)

    if mode == 'train_ratio':
        l = int(len(df) * split_value)
        train = df[:l]
        test = df[l:]

    elif mode == 'test_ratio':
        l = int(len(df) * split_value)
        test = df[:l]
        train = df[l:]

    elif mode == 'num_test':
        test = df[:split_value]
        train = df[split_value:]

    elif mode == 'num_train':
        train = df[:split_value]
        test = df[:split_value]
    
    else:
        ValueError()

    return train, test


def add_unsup(train_data, unsup_data, unsup_shaffle=True, mode='o', save_dir=None):
    if unsup_shaffle:
        unsup_data = unsup_data.sample(frac=1)

    df = pd.concat([train_data, unsup_data])

    if mode == 'o':
        return df

    elif mode == 'w':
        df.to_csv(os.path.join(save_dir, 'trainpp_uda_fomat_data.tsv'), sep='\t')


def main(argv):
    sup_df = load_data(FLAGS.sup_data)
    logging.info('sup data loaded')
    unsup_df = load_data(FLAGS.unsup_data)
    logging.info('unsup data loaded')

    sup_df = uda_formater(sup_df, FLAGS.sup_cont_col, FLAGS.label_col, FLAGS.sup_id_col)
    unsup_df = unsup_uda_formater(unsup_df, FLAGS.unsup_cont_col, FLAGS.unsup_id_col)
    logging.info('your dataset converted UDA format')

    train_df, test_df = train_test_split(sup_df, FLAGS.smethod_arg, FLAGS.split_arg)
    logging.info('finished train_test_split')

    test_df.to_csv(os.path.join(FLAGS.save_dir, 'test.tsv'), sep='\t', index=False)

    train_df = add_unsup(train_df, unsup_df)

    train_df.to_csv(os.path.join(FLAGS.save_dir, 'train.tsv'), sep='\t', index=False)
    logging.info('All the steps are done.')



if __name__ == '__main__':
    app.run(main)