import io
import os
import re
from contextlib import redirect_stdout
from datetime import datetime

from textgenrnn import textgenrnn


class Generator:
    def __init__(self, train_cfg, model_cfg, constants_cfg, error_threshold, mode):
        self.__train_cfg = train_cfg
        self.__model_cfg = model_cfg
        self.__data_path = constants_cfg['DATA_PATH']
        self.__mode = mode
        if self.__mode == 'structure':
            self.__category_path = constants_cfg['STRUCTURE_PATH']
            self.__source_path = constants_cfg['RELATIVE_PATH']
        elif self.__mode == 'chords':
            self.__category_path = constants_cfg['CHORDS_PATH']
            self.__source_path = constants_cfg['ANALYSIS_PATH']
        self.__model_path = constants_cfg['MODEL_PATH']
        self.__output_path = constants_cfg['OUTPUT_PATH']
        self.__error_threshold = error_threshold

    def load_model(self, genre='All', sub_genre=''):
        """
        Loads specified model.

        :param genre: str
        :param sub_genre: str
        :return: textgenrnn
        """
        model = None
        if sub_genre != '':
            model_path = os.path.join(self.__data_path, self.__category_path, self.__model_path,
                                      genre,
                                      'Model_' + sub_genre)
        else:
            model_path = os.path.join(self.__data_path, self.__category_path, self.__model_path,
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
        """
        Trains a new model for the specified genre or sub_genre, even if one existed before.
        Then the model gets saved.

        :param genre: str
        :param sub_genre: str
        """
        if sub_genre != '':
            model_name = 'Model_' + sub_genre
            model_path = os.path.join(self.__data_path, self.__category_path, self.__model_path,
                                      genre)
        else:
            model_name = 'Model_' + genre
            model_path = os.path.join(self.__data_path, self.__category_path, self.__model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model = textgenrnn(name=str(os.path.join(model_path, model_name)))
        self.train_model(model=model, genre=genre, sub_genre=sub_genre, new=True)

    def train_model(self, model, genre='All', sub_genre='', new=False):
        """
        Trains a given model for the specified genre or sub_genre.
        Then the model gets saved.

        :param model: textgenrnn
        :param genre: str
        :param sub_genre: str
        :param new: boolean
        """
        if sub_genre != '':
            file_path = os.path.join(self.__data_path, self.__category_path, self.__source_path,
                                     genre,
                                     sub_genre + '.txt')
        else:
            file_path = os.path.join(self.__data_path, self.__category_path, self.__source_path,
                                     genre + '.txt')
        num_words = 0
        with open(file_path, 'r') as file:
            for line in file:
                num_words += len(line.split())
        batch_size = 1024
        while num_words // batch_size < 3:
            batch_size = batch_size // 2
        train_function = model.train_from_file
        i = 1
        while True:
            with io.StringIO() as buf, redirect_stdout(buf):
                train_function(
                    file_path=file_path,
                    new_model=new,
                    num_epochs=self.__train_cfg['num_epochs'],
                    gen_epochs=self.__train_cfg['gen_epochs'],
                    batch_size=batch_size,
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
                new = False
                output = buf.getvalue().split('\n')
            err = 0.0
            j = 0
            for k in output:
                if 'Epoch' in k:
                    print('Epoch ' + str(i))
                else:
                    print(k)
                reg = re.search(r'(\d+\.\d+)', k)
                if reg is not None:
                    err = (err * j + float(reg.group(reg.lastindex))) / (j + 1)
                    j += 1
            if err < self.__error_threshold:
                break
            i += 1

    def generate_text_to_file(self, model, max_gen_length=256):
        """
        Generates an output that resembles the training data and stores is as a file.

        :param model: textgenrnn
        :param max_gen_length: int
        :return: list<str>
        """
        temperature = 1.0
        prefix = None
        number = 1
        time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
        genre = os.path.basename(os.path.dirname(model.config['name']))
        file_path = os.path.join(self.__data_path, self.__category_path, self.__output_path)
        if genre != self.__model_path:
            file_path = os.path.join(file_path, genre)
        file_path = os.path.join(file_path, os.path.basename(model.config['name'])[6:])
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file = os.path.join(file_path, time_string + '.txt')
        texts = model.generate(return_as_list=True, temperature=temperature, prefix=prefix,
                               n=number,
                               max_gen_length=max_gen_length)
        with open(file, 'w') as f:
            if self.__mode == 'structure':
                for text in texts:
                    f.write("{}\n".format(text))
            elif self.__mode == 'chords':
                f.write(", ".join(texts))
        return texts
