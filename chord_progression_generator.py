import os
import sys

from generator import Generator


def create_generator(constants_cfg):
    """
    Creates a generator for chord progressions.

    :param constants_cfg: dict<str:str>
    :return: generator.Generator
    """
    train_cfg = {
        'line_delimited': True,  # set to True if each text has its own line in the source file
        'num_epochs': 1,  # epochs to be trained but it will train until certain threshold
        'gen_epochs': 10,  # generates sample text from model after given number of epochs
        'train_size': 0.8,  # training set to test set ratio
        'dropout': 0.0,  # random ignore proportion of source tokens each epoch
        'validation': False,  # If train__size < 1.0, test on holdout dataset
        'is_csv': False  # set to True if file is a CSV exported from Excel/BigQuery/pandas
    }
    model_cfg = {
        'word_level': True,  # set to True if want to train a word-level model
        'rnn_size': 128,  # number of LSTM cells of each layer
        'rnn_layers': 3,  # number of LSTM layers
        'rnn_bidirectpional': False,  # consider text both forwards and backward
        'max_length': 5,  # number of tokens to consider before predicting the next
        'max_words': 500  # maximum number of words to model; the rest will be ignored
    }

    gen = Generator(
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        constants_cfg=constants_cfg,
        error_threshold=0.7,
        mode='chords'
    )
    return gen


def main():
    """
    Creates a generator and handels terminal input

    parameters: <mode> <genre> [sub_genre]
    modes: train, train_new, generate
    """
    constants_cfg = dict()
    with open('constants.cfg', mode='r') as file:
        for line in file:
            if line[0] != '#':
                tmp = line.rstrip('\n').split('=')
                constants_cfg[tmp[0]] = tmp[1]

    gen = create_generator(constants_cfg)

    if len(sys.argv) in (3, 4):
        source_path = os.path.join(constants_cfg['DATA_PATH'],
                                   constants_cfg['CHORDS_PATH'],
                                   constants_cfg['ANALYSIS_PATH'])
        for arg in sys.argv[2:]:
            source_path = os.path.join(source_path, arg)
        source_path += '.txt'

        if os.path.exists(source_path):
            genre = sys.argv[2]
            if len(sys.argv) == 3:
                sub_genre = ''
            else:
                sub_genre = sys.argv[3]

            if sys.argv[1] == 'train_new':
                gen.train_new_model(genre=genre, sub_genre=sub_genre)
            elif sys.argv[1] == 'generate':
                model = gen.load_model(genre=genre, sub_genre=sub_genre)
                if model is None:
                    print('No existing Model.'
                          'Please make sure to train a new Model first')
                    return
                gen.generate_text_to_file(model)
            elif sys.argv[1] == 'train':
                model = gen.load_model(genre=genre, sub_genre=sub_genre)
                if model is None:
                    print('No existing Model.'
                          'Please make sure to train a new Model first')
                    return
                gen.train_model(model=model, genre=genre, sub_genre=sub_genre)
            else:
                print('Operation arguments invalid.'
                      'Please check the spelling.')
        else:
            print('Genre arguments invalid.'
                  'Please check the spelling and make sure you run chord_progression_processor.py')


if __name__ == '__main__':
    main()
