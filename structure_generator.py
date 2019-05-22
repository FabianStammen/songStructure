import os
import io
from contextlib import redirect_stdout
import re
from datetime import datetime
from textgenrnn import textgenrnn


class StructureGenerator:
    def __init__(self, data_path='data_parsed', relative_path='relative',
                 structure_path='structure', model_path='models', output_path='songs'):
        self.__data_path = data_path
        self.__structure_path = structure_path
        self.__relative_path = relative_path
        self.__model_path = model_path
        self.__output_path = output_path
        self.__train_cfg = {
            'line_delimited': True,  # set to True if each text has its own line in the source file
            'num_epochs': 1,  # epochs to be trained but it will train until certain threshold
            'gen_epochs': 10,  # generates sample text from model after given number of epochs
            'train_size': 0.8,  # training set to test set ratio
            'dropout': 0.0,  # random ignore proportion of source tokens each epoch
            'validation': False,  # If train__size < 1.0, test on holdout dataset
            'is_csv': False  # set to True if file is a CSV exported from Excel/BigQuery/pandas
        }
        self.__model_cfg = {
            'word_level': True,  # set to True if want to train a word-level model
            'rnn_size': 128,  # number of LSTM cells of each layer
            'rnn_layers': 3,  # number of LSTM layers
            'rnn_bidirectional': False,  # consider text both forwards and backward
            'max_length': 5,  # number of tokens to consider before predicting the next
            'max_words': 25,  # maximum number of words to model; the rest will be ignored
        }

    def load_model(self, genre='All', sub_genre=''):
        model = None
        if sub_genre != '':
            model_path = os.path.join(self.__data_path, self.__structure_path, self.__model_path,
                                      genre,
                                      'Model_' + sub_genre)
        else:
            model_path = os.path.join(self.__data_path, self.__structure_path, self.__model_path,
                                      'Model_' + genre)
        weights_path = model_path + '_weights.hdf5'
        vocab_path = model_path + '_vocab.json'
        config_path = model_path + '_config.json'
        if os.path.exists(weights_path) and os.path.exists(vocab_path) and os.path.exists(
                config_path):
            model = textgenrnn(name=str(model_path),
                               weights_path=weights_path,
                               vocab_path=vocab_path,
                               config_path=config_path)
        return model

    def train_new_model(self, genre='All', sub_genre=''):
        if sub_genre != '':
            file = os.path.join(self.__data_path, self.__structure_path, self.__relative_path,
                                genre,
                                sub_genre + '.txt')
            model_name = 'Model_' + sub_genre
            model_path = os.path.join(self.__data_path, self.__structure_path, self.__model_path,
                                      genre)
        else:
            file = os.path.join(self.__data_path, self.__structure_path, self.__relative_path,
                                genre + '.txt')
            model_name = 'Model_' + genre
            model_path = os.path.join(self.__data_path, self.__structure_path, self.__model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model = textgenrnn(name=str(os.path.join(model_path, model_name)))
        train_function = model.train_from_file
        train_function(
            file_path=file,
            new_model=True,
            num_epochs=self.__train_cfg['num_epochs'],
            gen_epochs=self.__train_cfg['gen_epochs'],
            batch_size=1024,
            train_size=self.__train_cfg['train_size'],
            dropout=self.__train_cfg['dropout'],
            validation=self.__train_cfg['validation'],
            is_csv=self.__train_cfg['is_csv'],
            rnn_layers=self.__model_cfg['rnn_layers'],
            rnn_size=self.__model_cfg['rnn_size'],
            rnn_bidirectional=self.__model_cfg['rnn_bidirectional'],
            dim_embeddings=100,
            word_level=self.__model_cfg['word_level'],
            header=False)
        return model

    def train_model(self, model, genre='All', sub_genre=''):
        if sub_genre != '':
            file = os.path.join(self.__data_path, self.__structure_path, self.__relative_path,
                                genre,
                                sub_genre + '.txt')
        else:
            file = os.path.join(self.__data_path, self.__structure_path, self.__relative_path,
                                genre + '.txt')
        train_function = model.train_from_file
        while True:
            with io.StringIO() as buf, redirect_stdout(buf):
                train_function(
                    file_path=file,
                    new_model=False,
                    num_epochs=self.__train_cfg['num_epochs'],
                    gen_epochs=self.__train_cfg['gen_epochs'],
                    batch_size=1024,
                    train_size=self.__train_cfg['train_size'],
                    dropout=self.__train_cfg['dropout'],
                    validation=self.__train_cfg['validation'],
                    is_csv=self.__train_cfg['is_csv'],
                    rnn_layers=self.__model_cfg['rnn_layers'],
                    rnn_size=self.__model_cfg['rnn_size'],
                    rnn_bidirectional=self.__model_cfg['rnn_bidirectional'],
                    dim_embeddings=100,
                    word_level=self.__model_cfg['word_level'],
                    header=False)
                output = buf.getvalue().split('\n')
            err = 0.0
            j = 0
            for k in output:
                print(k)
                reg = re.search(r'(\d+\.\d+)', k)
                if reg is not None:
                    err = (err * j + float(reg.group(reg.lastindex))) / (j + 1)
                    j += 1
            if err < 0.7:
                break

    def generate_text_to_file(self, model):
        temperature = 1.0
        prefix = None
        number = 1
        max_gen_length = 500
        time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
        genre = os.path.basename(os.path.dirname(model.config['name']))
        file_path = os.path.join(self.__data_path, self.__structure_path, self.__output_path)
        if genre != self.__model_path:
            file_path = os.path.join(file_path, genre)
        file_path = os.path.join(file_path, os.path.basename(model.config['name'])[6:])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file = os.path.join(file_path, time_string + '.txt')
        model.generate_to_file(file, temperature=temperature, prefix=prefix, n=number,
                               max_gen_length=max_gen_length)
        with open(file, mode='r') as song:
            formated = song.readline().upper().split(' ')
        with open(file, mode='w') as song:
            for i in formated:
                song.write(i + '\n')
