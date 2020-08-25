from googletrans import Translator
import pandas as pd
from absl import flags, app, logging
import os
from joblib import Parallel, delayed

FLAGS = flags.FLAGS

'''
Parameters
'''
flags.DEFINE_string('train_data', './data/dataset/v2/train.tsv',
                    'Data for back translation.'
                    'The data must be in tsv format and must be train data.')

flags.DEFINE_string('sample', '-1',
                    'Specifies how much of the total data is used for back translation.')

flags.DEFINE_string('source_lang', 'ja',
                    'The language of the source text.'
                    'See https://py-googletrans.readthedocs.io/en/latest/ for details on supported languages.')

flags.DEFINE_string('target_lang', 'en',
                    'The language to translate the source text into. ')

flags.DEFINE_string('save_dir', './data/dataset/v2/backtranslated',
                    'path to back translated text.')


def paprallel_translate(no, text, s_lang='ja', t_lang='en'):
    print('processing no.', no)
    translator = Translator()
    tl_text = translator.translate(text, dest=t_lang, src=s_lang)
    tl_text = tl_text.text

    bt_text = translator.translate(tl_text, dest=s_lang, src=t_lang)
    bt_text = bt_text.text

    return no, bt_text


def back_translation(train_data, sample, save_dir, s_lang, t_lang):
    '''
    '''
    train_df = pd.read_csv(train_data, sep='\t')

    if sample != -1:

        if type(sample) == float:
            l = int(len(train_df) * sample)
            logging.info('{} of the total data is used for back translation.　¥n The data count is {}.'.format(sample, l))
            bt_df = train_df[:l]
            bt_df = bt_df['content']

        elif type(sample) == int:
            bt_df = train_df[:sample+1]
            bt_df = bt_df['content']
            logging.info('{} out of {} data is used for back translation.'.format(sample, len(train_df)))

    else:
        bt_df = train_df['content']
        logging.info('All data is used for back translation.')

    logging.info('translating...')

    bt_text = dict(Parallel(n_jobs=-1)(delayed(paprallel_translate)(i, t, FLAGS.source_lang, FLAGS.target_lang) for i, t in enumerate(bt_df.values.tolist())))

    bt_text_sorted = sorted(bt_text.items(), key=lambda x:x[0])

    results = [i[1] for i in bt_text_sorted]

    save_results = '\n'.join(results)

    try:
        os.mkdir(save_dir)
    except:
        pass

    with open(os.path.join(save_dir, 'bt_contents{}.txt'.format(sample)), 'w') as f:
        f.writelines(save_results)

    logging.info('All processes are finished.')


def main(argv):
    logging.info('start back translation process')
    
    if '.' in FLAGS.sample:
        sample = float(FLAGS.sample)

    else:
        sample = int(FLAGS.sample)

    back_translation(FLAGS.train_data, sample, FLAGS.save_dir, FLAGS.source_lang, FLAGS.target_lang)


if __name__ == '__main__':
    app.run(main)