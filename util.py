import numpy as np
import csv
import os
import json
import h5py
import re
import shutil
import sys


class LmdUtils:
    def __init__(self, dataset_path='datasets', data_path='data_parsed', lmd_path='lmd',
                 score_file='match_scores.json'):
        self.__dataset_path = dataset_path
        self.__data_path = data_path
        self.__lmd_path = lmd_path
        self.__score_file = os.path.join(self.__dataset_path, self.__lmd_path, score_file)
        self.__genre_router = self.__generate_genre_router()
        self.__entries = self.__generate_entries_skeleton()
        self.__midi_group = None

    def __msd_id_to_dirs(self, msd_id):
        """Given an MSD ID, generate the path prefix. E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
        return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

    def __msd_id_to_h5(self, msd_id):
        """Given an MSD ID, return the path to the corresponding h5"""
        return os.path.join(self.__dataset_path, self.__lmd_path, 'lmd_matched_h5',
                            self.__msd_id_to_dirs(msd_id) + '.h5')

    def __get_midi_path(self, msd_id, midi_md5):
        """Given an MSD ID and MIDI MD5, return path to a MIDI file."""
        return os.path.join(self.__dataset_path, self.__lmd_path, 'lmd_matched', self.__msd_id_to_dirs(msd_id),
                            midi_md5 + '.mid')

    def __get_midi_dir(self, msd_id):
        """Given an MSD ID, return path to a MIDI directory."""
        return os.path.join(self.__dataset_path, self.__lmd_path, 'lmd_matched', self.__msd_id_to_dirs(msd_id))

    def __get_most_fitting(self, msd_id):
        """Given an MSD ID, return path to best fitting midi."""
        best_md5 = "No sufficient match has been found"
        best_score = 0.0
        with open(self.__score_file) as f:
            scores = json.load(f)
            for midi_md5, score in scores[msd_id].items():
                if score > best_score:
                    best_score = score
                    best_md5 = midi_md5
        return os.path.join(self.__get_midi_path(msd_id, best_md5))

    def __clean_str(self, h5_entry):
        """Strip the strings from the hdf5 files clean"""
        if (str(h5_entry).startswith('b\'') or str(h5_entry).startswith('b\"')) and \
                (str(h5_entry).endswith('\'') or str(h5_entry).endswith('\"')):
            return str(h5_entry)[2:-1]
        return h5_entry

    def __clean_array(self, h5_array):
        """Uses clean_str() on an array"""
        return [self.__clean_str(h5_entry) for h5_entry in h5_array]

    def __generate_genre_router(self, genre_list_file="genre_list.txt", possible_list_file="possible_list.txt"):
        # ^((?!Electronica|Techno|House|Electro|Rap|Hip-Hop|Rock|Punk|Metal|Humor|Blues|Reggae|Country|Jazz|Spoken|Folk|R_B|World|Classical|Pop|Other).)*$
        genre_list = os.path.join(self.__data_path, genre_list_file)
        possible_list = os.path.join(self.__data_path, possible_list_file)
        genre_router = dict()
        with open(genre_list, "r") as pos:
            reader = csv.DictReader(pos)
            for line in reader:
                genre_router[line["SUBSUBGENRE"]] = {"SUBGENRE": line["SUBGENRE"]}
        for t, tt in genre_router.items():
            with open(possible_list, "r") as gl:
                reader = csv.DictReader(gl)
                for line in reader:
                    if line["SUBGENRE"] == tt["SUBGENRE"]:
                        genre_router[t]["GENRE"] = line["GENRE"]
                        break
        return genre_router

    def __generate_entries_skeleton(self, possible_list_file="possible_list.txt", genre_list_file="genre_list.txt"):
        genre_list = os.path.join(self.__data_path, genre_list_file)
        possible_list = os.path.join(self.__data_path, possible_list_file)
        entries = dict()
        with open(possible_list, "r") as pos:
            reader = csv.DictReader(pos)
            for line in reader:
                if line["GENRE"] in entries:
                    entries[line["GENRE"]][line["SUBGENRE"]] = dict()
                else:
                    entries[line["GENRE"]] = {line["SUBGENRE"]: dict()}
        with open(genre_list, "r") as gl:
            reader = csv.DictReader(gl)
            for line in reader:
                for t, tt in entries.items():
                    if line["SUBGENRE"] in tt:
                        entries[t][line["SUBGENRE"]][line["SUBSUBGENRE"]] = set()
        return entries

    def group_midi(self):
        self.__midi_group = self.__entries
        with open(self.__score_file) as f:
            scores = json.load(f)
            i = 0
            for msd_id, midis in scores.items():
                h5 = h5py.File(self.__msd_id_to_h5(msd_id))
                terms = self.__clean_array(h5['metadata']['artist_terms'][0:])
                terms_weight = self.__clean_array(h5['metadata']['artist_terms_weight'][0:])
                terms_freq = self.__clean_array(h5['metadata']['artist_terms_freq'][0:])
                term_factor_max = 0.0
                genre = ""
                for term in terms:
                    term_factor = float(terms_freq[terms.index(term)]) * float(terms_weight[terms.index(term)])
                    if term_factor >= term_factor_max:
                        term_factor_max = term_factor
                        genre = term
                if genre in self.__genre_router:
                    self.__midi_group[self.__genre_router[genre]["GENRE"]][self.__genre_router[genre]["SUBGENRE"]][
                        genre].add(msd_id)
                i += 1
                print("\r" + str(i) + "/" + str(len(scores.items())) + " = " + str(i / len(scores.items()))[:4]
                      .replace(".", "") + "%", end='', flush=True)
        print("\n")

    def get_midi_group(self):
        return self.__midi_group


class SalamiUtils:
    def __init__(self, dataset_path='dataset', data_path='data_parsed', genre_annotation_path='genre_annotations',
                 relative_path='relative', structure_path='structue', salami_path='salami', meta_file_folder='metadata',
                 meta_file='metadata.csv', section_dict='section_dict.json'):
        self.__dataset_path = dataset_path
        self.__data_path = data_path
        self.__structure_path = structure_path
        self.__relative_path = relative_path
        self.__genre_annotation_path = genre_annotation_path
        self.__salami_path = salami_path
        self.__meta_file = os.path.join(dataset_path, salami_path, meta_file_folder, meta_file)
        self.__section_dict = section_dict
        self.__genres = self.__get_genres()
        self.__annotations = self.__get_annotations(self.__genres)
        self.__sections = self.__get_sections(self.__annotations)
        self.__meta_tokens = [['-1.0', '<s>'], ['-0.5', '<e>']]

    def __get_annotation_dir(self, salami_id):
        """Given an SALAMI ID, return a path to an annotation dir."""
        return os.path.join(self.__dataset_path, self.__salami_path, 'annotations', salami_id)

    def __get_annotation_file(self, salami_id, textfile_id="1"):
        """Given an SALAMI ID, and textfile ID return a path to an parsed functions-annotation."""
        return os.path.join(self.__get_annotation_dir(salami_id), 'parsed', 'textfile' + textfile_id + '_functions.txt')

    def __fraction_round(self, raw_float, denominator=8):
        rounded = int('{0:.0f}'.format(raw_float))
        rest = raw_float - int(raw_float)
        for i in range(0, denominator + 1, 2):
            if (i - 1) / 16 <= rest < (i + 1) / denominator:
                rounded = int(raw_float) + i / denominator
                break
        return rounded

    def __get_genres(self):
        """
        Creates a dict that contains all song_ids that are specified within __meta_file and have some form of information
        on their genre.

        The structure is Genre>Sub_Genre>Song_id
        :return: dict(str:dict(str:set(str)))
        """
        genres = dict()
        classes = dict()
        with open(self.__meta_file, mode='r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if (row["GENRE"] != '') and (row["SONG_WAS_DISCARDED_FLAG"] != "TRUE"):
                    reg = re.search("(?:_)?([A-Z_]*[A-Za-z]*)(?:_-_)", row["GENRE"])
                    if reg is None:
                        reg = re.search("^(?:(?!_-_).)*$", row["GENRE"])
                        reg = re.search("([A-Za-z]*)$", reg.group())
                    group = reg.group(reg.lastindex)
                    if group in genres:
                        if row["GENRE"] in genres[group]:
                            genres[group][row["GENRE"]].add(row["SONG_ID"])
                        else:
                            genres[group][row["GENRE"]] = {row["SONG_ID"]}
                    else:
                        genres[group] = {row["GENRE"]: {row["SONG_ID"]}}
                elif (row["CLASS"] != '') and (row["SONG_WAS_DISCARDED_FLAG"] != "TRUE"):
                    upper = str.capitalize(row["CLASS"])
                    if upper in classes:
                        classes[upper].add(row["SONG_ID"])
                    else:
                        classes[upper] = {row["SONG_ID"]}
        for entry, content in classes.items():
            if entry in genres.keys():
                if entry in genres[entry].keys():
                    genres[entry][entry].add(content)
                else:
                    genres[entry][entry] = content
        return genres

    def __get_annotations(self, genres):
        """
        Creates a dict that contains all annotations for the given dict of genres or sub_genres.

        Structure is Genre>Sub_Genre>Annotation_file<Annotation
        :param genres: structure of genres or sub_genres
        :return: dict(str:dict(str:dict(str:dict(str:str))))
        """
        # Filter out all the faulty and unwanted annotations
        with open(os.path.join(self.__data_path, self.__section_dict), 'r') as json_file:
            vocab = json.load(json_file)
        sections = dict()
        if type(genres) is dict:
            for genre, content in genres.items():
                sections[genre] = self.__get_annotations(content)
        elif type(genres) is set:
            for song_id in genres:
                nr = len(next(os.walk(self.__get_annotation_dir(song_id)))[2])
                file_ids = []
                if nr == 3:
                    file_ids = ['1', '2', '2b'] if os.path.isfile(self.__get_annotation_file(song_id, '2b')) else ['1',
                                                                                                                   '1b',
                                                                                                                   '2']
                elif nr == 2:
                    file_ids = ['1', '2'] if os.path.isfile(self.__get_annotation_file(song_id, '2')) \
                        else ['1', '1b'] if os.path.isfile(self.__get_annotation_file(song_id, '1b')) \
                        else ['2', '2b']
                elif nr == 1:
                    file_ids = ['1'] if os.path.isfile(self.__get_annotation_file(song_id)) else ['2']
                for file_id in file_ids:
                    sections[song_id + ' - ' + file_id] \
                        = self.__get_annotations(self.__get_annotation_file(song_id, file_id))
        elif type(genres) is str:
            with open(genres) as file:
                last_section = ''
                for line in file:
                    section = re.search("\t(.+)$", line).group(1).upper().replace("-", "").replace("_", "")
                    time = "%.2f" % float(re.search("(^[0-9]+\.[0-9]+)", line).group())
                    if section in vocab:
                        sections[time] = section
                        last_section = section
                    elif last_section != '' and section != 'END':
                        sections[time] = last_section
        return sections

    def __get_sections(self, annotations):
        """
        Creates a dict that contains all annotated sections grouped and their occurrence for the given dict of annotations.

        Structure is Section>occurrence
        :param annotations: structure of annotations
        :return: dict(str:list(str))
        """
        sections = dict()
        if type(annotations) is dict:
            for annotation, content in annotations.items():
                count = self.__get_sections(content)
                if type(count) is str:
                    if count in sections:
                        sections[count] = sections[count] + 1
                    else:
                        sections[count] = 1
                elif type(count) is dict:
                    if re.search("(^[0-9]+ - [0-9]+b?$)", annotation) is not None:
                        for section, content in count.items():
                            for i in range(content):
                                if section in sections:
                                    sections[section].append(annotation)
                                else:
                                    sections[section] = [annotation]
                    else:
                        for section, content in count.items():
                            if section not in sections:
                                sections[section] = content
                            else:
                                for file_name in content:
                                    sections[section].append(file_name)
        elif type(annotations) is str:
            return annotations
        return sections

    def get_genres(self):
        return self.__genres

    def get_annotations(self):
        return self.__annotations

    def get_sections(self):
        return self.__sections

    def __create_genre_annotations(self, annotations):
        genre_annotation_dir = os.path.join(self.__data_path, self.__structure_path, self.__genre_annotation_path)
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
                        time = "%.2f" % (float(time) - offset)
                        np_track.append([time, section])
                    np_track[0] = self.__meta_tokens[0]  # exchange first SILENCE with metaToken
                    max_sections_a = len(np_track) if len(np_track) >= max_sections_a else max_sections_a
                    max_sections_g = len(np_track) if len(np_track) >= max_sections_g else max_sections_g
                    max_sections_sg = len(np_track) if len(np_track) >= max_sections_sg else max_sections_sg
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
        if sub_genre == '':
            return np.load(
                os.path.join(self.__data_path, self.__structure_path, self.__genre_annotation_path, genre + '.npy'))
        else:
            return np.load(
                os.path.join(self.__data_path, self.__structure_path, self.__genre_annotation_path, genre,
                             sub_genre + '.npy'))

    def __relativate(self, genre='All', sub_genre=''):
        relative_genre_annotation = os.path.join(self.__data_path, self.__structure_path, self.__relative_path)
        data = self.load_genre_annotations(genre=genre, sub_genre=sub_genre)
        times = np.delete(data, 1, 2)
        times = times.flatten().astype(np.float32, copy=False).reshape(times.shape)
        sections = np.delete(data, 0, 2)
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
            for k, v in enumerate(i):
                if 0.0 < v[0] < div:
                    if div - v[0] > tol / 2:
                        if i[k + 1][0] != -0.5:
                            div = v[0]
                        else:
                            break
                elif v[0] == -0.5:
                    break
            tmp = list()
            factor = 1
            for k, v in enumerate(i):
                if v[0] == -0.5:
                    break
                elif v[0] == 0.0 and k == 1:
                    tmp.append(v[0])
                elif v[0] != -1.0:
                    dived = v[0] / div
                    if type(dived) is np.ndarray or type(dived) is not np.float32:
                        dived = dived[0]
                    rel = self.__fraction_round(dived)
                    tmp.append(rel)
                    t_factor = factor
                    if (rel - int(rel)) in case_four:
                        t_factor = 4
                    elif (rel - int(rel)) in case_two:
                        t_factor = 2
                    if t_factor > factor:
                        factor = t_factor

            total = 0
            for j in tmp:
                total = total + int(format(j * factor, '.0f'))
            while total * factor > length:
                factor = factor / 2

            for kj, vj in enumerate(tmp):
                tmp[kj] = int(format(vj * factor, '.0f'))
            res.append(tmp)

        sequences = list()
        for ki, vi in enumerate(sections):
            sequence = list()
            for kj, vj in enumerate(vi):
                if vi[kj + 1] == '<e>':
                    break
                for i in range(res[ki][kj]):
                    sequence.append(vj[0])
            sequences.append(sequence)
        if sub_genre is not '':
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
        self.__create_genre_annotations(self.__annotations)
        self.__relativate()
        for genre, sub_genres in self.__genres.items():
            self.__relativate(genre)
            for sub_genre, _ in sub_genres.items():
                self.__relativate(genre, sub_genre)


def init_salami():
    constants = dict()
    with open('constants.cfg', mode='r') as f:
        for line in f:
            if line[0] is not '#':
                tmp = line.rstrip('\n').split('=')
                constants[tmp[0]] = tmp[1]
    salami = SalamiUtils(dataset_path=constants['DATASET_PATH'],
                         data_path=constants['DATA_PATH'],
                         genre_annotation_path=constants['GENRE_ANNOTATION_PATH'],
                         relative_path=constants['RELATIVE_PATH'],
                         structure_path=constants['STRUCTURE_PATH'],
                         salami_path=constants['SALAMI_PATH'],
                         meta_file_folder=constants['META_FILE_FOLDER'],
                         meta_file=constants['META_FILE'],
                         section_dict=constants['SECTION_DICT'])
    salami.initiate()


def init_lmd():
    constants = dict()
    with open('constants.cfg', mode='r') as f:
        for line in f:
            if line[0] is not '#':
                tmp = line.rstrip('\n').split('=')
                constants[tmp[0]] = tmp[1]
    lmd = LmdUtils(dataset_path=constants['DATASET_PATH'],
                   data_path=constants['DATA_PATH'],
                   lmd_path=constants['LMD_PATH'],
                   score_file=constants['SCORE_FILE'])
    lmd.group_midi()
    midi_group = lmd.get_midi_group()
    # TODO create chord progressions


if sys.argv[1] == 'salami':
    init_salami()
elif sys.argv[1] == 'lmd':
    init_lmd()
