import csv
import json
import os
import re
import shutil
import util
import numpy as np


class StructureProcessor:
    def __init__(self, constants_cfg):
        """
        :param constants_cfg: dict<str:str>
        """
        self.__dataset_path = constants_cfg['DATASET_PATH']
        self.__data_path = constants_cfg['DATA_PATH']
        self.__structure_path = constants_cfg['STRUCTURE_PATH']
        self.__relative_path = constants_cfg['RELATIVE_PATH']
        self.__genre_annotation_path = constants_cfg['GENRE_ANNOTATION_PATH']
        self.__salami_path = constants_cfg['SALAMI_PATH']
        self.__meta_file_path = os.path.join(constants_cfg['DATASET_PATH'],
                                             constants_cfg['SALAMI_PATH'],
                                             constants_cfg['META_FILE_FOLDER'],
                                             constants_cfg['META_FILE'])
        self.__section_dict_path = constants_cfg['SECTION_DICT']
        self.__genres = self.__create_genres()
        self.__annotations = self.__create_annotations(self.__genres)
        self.__sections = self.__create_sections(self.__annotations)
        self.__meta_tokens = [['-1.0', '<s>'], ['-0.5', '<e>']]

    def __get_annotation_dir(self, salami_id):
        """
        Given an SALAMI ID, return a path to an annotation dir.

        :param salami_id: str
        :return: str
        """
        return os.path.join(self.__dataset_path, self.__salami_path, 'annotations', salami_id)

    def __get_annotation_file(self, salami_id, textfile_id='1'):
        """
        Given an SALAMI ID, and textfile ID return a path to an parsed functions-annotation.

        :param salami_id: str
        :param textfile_id: int
        :return: str
        """
        return os.path.join(self.__get_annotation_dir(salami_id), 'parsed',
                            'textfile' + textfile_id + '_functions.txt')

    def __fraction_round(self, raw_float, denominator=8):
        """
        Rounds a float with a given amount of digits behind the decimal point to be considered.

        :param raw_float: float
        :param denominator: int
        :return: int
        """
        rounded = int('{0:.0f}'.format(raw_float))
        rest = raw_float - int(raw_float)
        for i in range(0, denominator + 1, 2):
            if (i - 1) / 16 <= rest < (i + 1) / denominator:
                rounded = int(raw_float) + i / denominator
                break
        return rounded

    def __create_genres(self):
        """
        Creates a dict that contains all song_ids that are specified within __meta_file and have
        some form of information on their genre.

        The structure is Genre>Sub_Genre>Song_id
        :return: dict<str:dict<str:set<str>>>
        """
        genres = dict()
        classes = dict()
        with open(self.__meta_file_path, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if (row['GENRE'] != '') and (row['SONG_WAS_DISCARDED_FLAG'] != 'TRUE'):
                    reg = re.search('(?:_)?([A-Z_]*[A-Za-z]*)(?:_-_)', row['GENRE'])
                    if reg is None:
                        reg = re.search('^(?:(?!_-_).)*$', row['GENRE'])
                        reg = re.search('([A-Za-z]*)$', reg.group())
                    group = reg.group(reg.lastindex)
                    if group in genres:
                        if row['GENRE'] in genres[group]:
                            genres[group][row['GENRE']].add(row['SONG_ID'])
                        else:
                            genres[group][row['GENRE']] = {row['SONG_ID']}
                    else:
                        genres[group] = {row['GENRE']: {row['SONG_ID']}}
                elif (row['CLASS'] != '') and (row['SONG_WAS_DISCARDED_FLAG'] != 'TRUE'):
                    upper = str.capitalize(row['CLASS'])
                    if upper in classes:
                        classes[upper].add(row['SONG_ID'])
                    else:
                        classes[upper] = {row['SONG_ID']}
        for entry, content in classes.items():
            if entry in genres.keys():
                if entry in genres[entry].keys():
                    genres[entry][entry].add(content)
                else:
                    genres[entry][entry] = content
        return genres

    def __create_annotations(self, genres):
        """
        Creates a dict that contains all annotations for the given dict of genres or sub_genres.

        Structure is Genre>Sub_Genre>Annotation_file<Annotation
        :param genres: dict<str:dict<str:set<str>>>
        :return: dict<str:dict<str:dict<str:dict<str:str>>>>
        """
        # Filter out all the faulty and unwanted annotations
        with open(os.path.join(self.__data_path, self.__section_dict_path), 'r') as json_file:
            vocab = json.load(json_file)
        sections = dict()
        if type(genres) is dict:
            for genre, content in genres.items():
                sections[genre] = self.__create_annotations(content)
        elif type(genres) is set:
            for song_id in genres:
                number = len(next(os.walk(self.__get_annotation_dir(song_id)))[2])
                file_ids = []
                if number == 3:
                    file_ids = ['1', '2', '2b'] if os.path.isfile(
                        self.__get_annotation_file(song_id, '2b')) else ['1', '1b', '2']
                elif number == 2:
                    file_ids = ['1', '2'] \
                        if os.path.isfile(self.__get_annotation_file(song_id, '2')) \
                        else ['1', '1b'] \
                        if os.path.isfile(self.__get_annotation_file(song_id, '1b')) \
                        else ['2', '2b']
                elif number == 1:
                    file_ids = ['1'] if os.path.isfile(self.__get_annotation_file(song_id)) else [
                        '2']
                for file_id in file_ids:
                    sections[song_id + ' - ' + file_id] \
                        = self.__create_annotations(self.__get_annotation_file(song_id, file_id))
        elif type(genres) is str:
            with open(genres) as file:
                last_section = ''
                for line in file:
                    section = re.search('\t(.+)$', line).group(1).upper().replace('-', '').replace(
                        '_', '')
                    time = '%.2f' % float(re.search(r'(^[0-9]+\.[0-9]+)', line).group())
                    if section in vocab:
                        sections[time] = section
                        last_section = section
                    elif last_section != '' and section != 'END':
                        sections[time] = last_section
        return sections

    def __create_sections(self, annotations):
        """"
        Creates a dict that contains all annotated sections grouped and their occurrence for the
        given dict of annotations.

        Structure is Section>occurrence
        :param annotations: dict<str:dict<str:dict<str:dict<str:str>>>>
        :return: dict<str:list<str>>
        """
        sections = dict()
        if type(annotations) is dict:
            for dict_key, dict_value in annotations.items():
                count = self.__create_sections(dict_value)
                if type(count) is str:
                    if count in sections:
                        sections[count] = sections[count] + 1
                    else:
                        sections[count] = 1
                elif type(count) is dict:
                    if re.search('(^[0-9]+ - [0-9]+b?$)', dict_key) is not None:
                        for section, section_count in count.items():
                            for _ in range(section_count):
                                if section in sections:
                                    sections[section].append(dict_key)
                                else:
                                    sections[section] = [dict_key]
                    else:
                        for section, section_count_list in count.items():
                            if section not in sections:
                                sections[section] = section_count_list
                            else:
                                for file_name in section_count_list:
                                    sections[section].append(file_name)
        elif type(annotations) is str:
            return annotations
        return sections

    def get_genres(self):
        """
        Generic get-method for genres field.
        :return: dict<str:dict<str:set<str>>>
        """
        return self.__genres

    def get_annotations(self):
        """
        Generic get-method for annotations field.
        :return: dict<str:dict<str:dict<str:dict<str:str>>>>
        """
        return self.__annotations

    def get_sections(self):
        """
        Generic get-method for sections field
        :return: dict<str:list<str>>
        """
        return self.__sections

    def __create_genre_annotations(self, annotations):
        """
        Saves all annotations in numpy.ndarrays according to their genres and respective sub_genres.
        The output folder is defined as DATA_PATH/STRUCTURE_PATH/GENRE_ANNOTATION_PATH/.

        :param annotations: dict<str:dict<str:dict<str:dict<str:str>>>>
        """
        genre_annotation_dir = os.path.join(self.__data_path, self.__structure_path,
                                            self.__genre_annotation_path)
        if os.path.exists(genre_annotation_dir):
            shutil.rmtree(genre_annotation_dir)
        os.makedirs(genre_annotation_dir)
        npy_all = []
        max_sections_a = 0
        for genre, sub_genres in annotations.items():
            npy_genre = []
            max_sections_g = 0
            for sub_genre, track in sub_genres.items():
                npy_sub_genre = []
                max_sections_sg = 0
                for _, annotation in track.items():
                    np_track = list()
                    offset = 0.0
                    for time, section in annotation.items():
                        if offset == 0.0:
                            offset = float(time)
                        time = '%.2f' % (float(time) - offset)
                        np_track.append([time, section])
                    np_track[0] = self.__meta_tokens[0]  # exchange first SILENCE with metaToken
                    max_sections_a = len(np_track) if len(
                        np_track) >= max_sections_a else max_sections_a
                    max_sections_g = len(np_track) if len(
                        np_track) >= max_sections_g else max_sections_g
                    max_sections_sg = len(np_track) if len(
                        np_track) >= max_sections_sg else max_sections_sg
                    npy_all.append(np_track)
                    npy_genre.append(np_track)
                    npy_sub_genre.append(np_track)
                np_res = list()
                for np_track in npy_sub_genre:
                    while len(np_track) <= max_sections_sg:
                        np_track.append(self.__meta_tokens[1])
                    np_res.append(np_track)
                    if not os.path.exists(os.path.join(genre_annotation_dir, genre)):
                        os.makedirs(os.path.join(genre_annotation_dir, genre))
                np.save(os.path.join(genre_annotation_dir, genre, sub_genre), np.array(np_res))
            np_res = list()
            for np_track in npy_genre:
                while len(np_track) <= max_sections_g:
                    np_track.append(self.__meta_tokens[1])
                np_res.append(np_track)
            np.save(os.path.join(genre_annotation_dir, genre), np.array(np_res))
        np_res = list()
        for np_track in npy_all:
            while len(np_track) <= max_sections_a:
                np_track.append(self.__meta_tokens[1])
            np_res.append(np_track)
        np.save(os.path.join(genre_annotation_dir, 'All'), np.array(np_res))

    def load_genre_annotations(self, genre='All', sub_genre=''):
        """
        Loads all saved genre_annotations as numpy.ndarrays.

        :param genre: str
        :param sub_genre: str
        :return numpy.ndarray
        """
        if sub_genre == '':
            return np.load(
                os.path.join(self.__data_path, self.__structure_path, self.__genre_annotation_path,
                             genre + '.npy'))
        else:
            return np.load(
                os.path.join(self.__data_path, self.__structure_path, self.__genre_annotation_path,
                             genre,
                             sub_genre + '.npy'))

    def __relativize(self, genre='All', sub_genre=''):
        """
        Loads all saved genre_annotations as numpy.ndarrays and put the time series information from
        absolute values into relative ones. Then the sections get multiplied according to their
        relation. Also the relative genre annotations get saved as .txt file to
        DATA_PATH/STRUCTURE_PATH/RELATIVE_PATH/.

        :param genre: str
        :param sub_genre: str
        :return list<list<str>>
        """
        relative_genre_annotation = os.path.join(self.__data_path, self.__structure_path,
                                                 self.__relative_path)
        data = self.load_genre_annotations(genre=genre, sub_genre=sub_genre)
        times = np.delete(data, 1, 2)
        times = times.flatten().astype(np.float32, copy=False).reshape(times.shape)
        annotations = np.delete(data, 0, 2)
        for i in times:
            length = max(i)
            last = 0.0
            for j in i:
                if j[0] == -1.0:
                    j[0] = length
                    last = 0.0
                elif j[0] == -0.5:
                    last = 0.0
                else:
                    tmp = j[0]
                    j[0] = float(format(j[0] - last, '.2f'))
                    last = tmp
        res = list()
        tol = 0.2
        case_four = [1 / 4, 3 / 4]
        case_two = [1 / 2]
        for i in times:
            length = i[0][0]
            i[0][0] = -1.0
            div = max(i)
            for key, value in enumerate(i):
                if 0.0 < value[0] < div:
                    if div - value[0] > tol / 2:
                        if i[key + 1][0] != -0.5:
                            div = value[0]
                        else:
                            break
                elif value[0] == -0.5:
                    break
            tmp = list()
            factor = 1
            for key, value in enumerate(i):
                if value[0] == -0.5:
                    break
                elif value[0] == 0.0 and key == 1:
                    tmp.append(value[0])
                elif value[0] != -1.0:
                    dived = value[0] / div
                    if type(dived) is np.ndarray or type(dived) is not np.float32:
                        dived = dived[0]
                    rel = self.__fraction_round(dived)
                    tmp.append(rel)
                    t_factor = factor
                    if rel - int(rel) in case_four:
                        t_factor = 4
                    elif rel - int(rel) in case_two:
                        t_factor = 2
                    if t_factor > factor:
                        factor = t_factor

            total = 0
            for j in tmp:
                total = total + int(format(j * factor, '.0f'))
            while total * factor > length:
                factor = factor / 2

            for key, value in enumerate(tmp):
                tmp[key] = int(format(value * factor, '.0f'))
            res.append(tmp)

        sequences = list()
        for i, annotation in enumerate(annotations):
            sequence = list()
            for j, section in enumerate(annotation):
                if annotation[j + 1] == '<e>':
                    break
                for _ in range(res[i][j]):
                    sequence.append(section[0])
            if len(sequence) > 6 and len(set(sequence)) > 1:
                sequences.append(sequence)
        if len(sequences) is not 0:
            if sub_genre != '':
                file = os.path.join(relative_genre_annotation, genre, sub_genre + '.txt')
            else:
                file = os.path.join(relative_genre_annotation, genre + '.txt')
            if not os.path.exists(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))
            with open(file, mode='w') as file:
                for line_split in sequences:
                    file.write(' '.join(line_split) + '\n')
            return sequences

    def initiate(self):
        """
        Applies the relativize method to all genres and subgenres specified in genre_annotations.
        """
        self.__create_genre_annotations(self.__annotations)
        self.__relativize()
        for genre, sub_genres in self.__genres.items():
            self.__relativize(genre)
            for sub_genre, _ in sub_genres.items():
                self.__relativize(genre, sub_genre)


def main():
    """
    Loads constants dictionary and modifies all annotations into a usable format for the
    generation process.
    """
    constants_cfg = util.load_constants()
    sp = StructureProcessor(constants_cfg)
    sp.initiate()


if __name__ == '__main__':
    main()
