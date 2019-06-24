import csv
import json
import os
import pickle
import sys
from collections import OrderedDict
from multiprocessing import Pool, cpu_count, freeze_support
import util
import h5py
from music21 import midi, stream, roman, chord
from music21.analysis.discrete import DiscreteAnalysisException
from music21.exceptions21 import StreamException
from music21.meter import MeterException


class ChordProgressionProcessor:
    def __init__(self, constants_cfg):
        """

        :param constants_cfg: dict<str:str>
        """
        self.__dataset_path = constants_cfg['DATASET_PATH']
        self.__data_path = constants_cfg['DATA_PATH']
        self.__lmd_path = constants_cfg['LMD_PATH']
        self.__matched_path = constants_cfg['MATCHED_PATH']
        self.__h5_path = constants_cfg['H5_PATH']
        self.__chords_path = constants_cfg['CHORDS_PATH']
        self.__analysis_path = constants_cfg['ANALYSIS_PATH']
        self.__genre_dict_path = os.path.join(constants_cfg['DATA_PATH'],
                                              constants_cfg['GENRE_DICT_FILE'])
        self.__possible_dict_path = os.path.join(constants_cfg['DATA_PATH'],
                                                 constants_cfg['POSSIBLE_DICT_FILE'])
        self.__score_file_path = os.path.join(constants_cfg['DATASET_PATH'],
                                              self.__lmd_path, constants_cfg['SCORE_FILE'])
        self.__midi_dict_path = os.path.join(constants_cfg['DATA_PATH'],
                                             constants_cfg['MIDI_DICT'])
        self.__midi_group = None

    def __msd_id_to_dirs(self, msd_id):
        """
        Given an MSD ID, generate the path prefix. E.g. TRABCD12345678 -> A/B/C/TRABCD12345678

        :param msd_id:str
        :return str
        """
        return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

    def __msd_id_to_h5(self, msd_id):
        """
        Given an MSD ID, return the path to the corresponding h5

        :param msd_id:str
        :return str
        """
        return os.path.join(self.__dataset_path, self.__lmd_path, self.__h5_path,
                            self.__msd_id_to_dirs(msd_id) + '.h5')

    def __get_midi_path(self, msd_id, midi_md5):
        """
        Given an MSD ID and MIDI MD5, return path to a MIDI file.

        :param msd_id:str
        :param midi_md5:str
        :return str
        """
        return os.path.join(self.__dataset_path, self.__lmd_path, self.__matched_path,
                            self.__msd_id_to_dirs(msd_id),
                            midi_md5 + '.mid')

    def __get_midi_dir(self, msd_id):
        """
        Given an MSD ID, return path to a MIDI directory.

        :param msd_id:str
        :return str
        """
        return os.path.join(self.__dataset_path, self.__lmd_path, self.__matched_path,
                            self.__msd_id_to_dirs(msd_id))

    def __get_fitting(self, msd_id):
        """
        Given an MSD ID, return a sorted list of paths to all fitting midi.

        :param msd_id:str
        :return list<str>
        """
        with open(self.__score_file_path) as file:
            scores = json.load(file)
            results = sorted(scores[msd_id], key=scores[msd_id].get)
        results = [os.path.join(self.__get_midi_path(msd_id, midi_md5)) for midi_md5 in results]
        return results

    def __clean_str(self, h5_entry):
        """
        Strip the strings from the hdf5 files clean

        :param h5_entry: bytes
        :return str
        """
        if (str(h5_entry).startswith('b\'') or str(h5_entry).startswith('b\'')) and \
                (str(h5_entry).endswith('\'') or str(h5_entry).endswith('\'')):
            return str(h5_entry)[2:-1]
        return h5_entry

    def __clean_array(self, h5_array):
        """
        Uses clean_str() on an array

        :param h5_array: numpy.ndarray<bytes>
        :return list<str>
        """
        return [self.__clean_str(h5_entry) for h5_entry in h5_array]

    def __generate_genre_router(self):
        """
        Generates a genre router that maps from a sub_sub_genre to a sub_genre to a genre.
        Needed for sorting all the midi files according to genre.

        :return: dict<str:dict<str:str>>
        """
        genre_router = dict()
        with open(self.__genre_dict_path, 'r') as pos:
            reader = csv.DictReader(pos)
            for line in reader:
                genre_router[line['SUBSUBGENRE']] = {'SUBGENRE': line['SUBGENRE']}
        for key, value in genre_router.items():
            with open(self.__possible_dict_path, 'r') as genre_dict:
                reader = csv.DictReader(genre_dict)
                for line in reader:
                    if line['SUBGENRE'] == value['SUBGENRE']:
                        genre_router[key]['GENRE'] = line['GENRE']
                        break
        return genre_router

    def __generate_entries_skeleton(self):
        """
        Generates a data structure skeleton to store the chord progression in according to genre.

        :return: dict<str:dict<str:dict<str:set<>>>>
        """
        entries = dict()
        with open(self.__possible_dict_path, 'r') as genre_dict:
            reader = csv.DictReader(genre_dict)
            for line in reader:
                if line['GENRE'] in entries:
                    entries[line['GENRE']][line['SUBGENRE']] = dict()
                else:
                    entries[line['GENRE']] = {line['SUBGENRE']: dict()}
        with open(self.__genre_dict_path, 'r') as genre_dict:
            reader = csv.DictReader(genre_dict)
            for line in reader:
                for key, value in entries.items():
                    if line['SUBGENRE'] in value:
                        entries[key][line['SUBGENRE']][line['SUBSUBGENRE']] = set()
        return entries

    def group_midi(self, new=False):
        """
        Iterates through all entries listed in the score file and sort them by genre.
        Afterwards a dict file is saved that contains all the mappings.

        :param new: boolean
        """
        if not new and os.path.exists(self.__midi_dict_path):
            with open(self.__midi_dict_path, mode='rb') as file:
                self.__midi_group = pickle.load(file)
        else:
            genre_router = self.__generate_genre_router()
            self.__midi_group = self.__generate_entries_skeleton()
            with open(self.__score_file_path) as file:
                scores = json.load(file)
                nr_of_scores = len(scores.items())
                i = 0
                print('Grouping MIDI files by Genre')
                for msd_id, _ in scores.items():
                    h5 = h5py.File(self.__msd_id_to_h5(msd_id))
                    terms = self.__clean_array(h5['metadata']['artist_terms'][0:])
                    terms_weight = self.__clean_array(h5['metadata']['artist_terms_weight'][0:])
                    terms_freq = self.__clean_array(h5['metadata']['artist_terms_freq'][0:])
                    term_factor_max = 0.0
                    sub_sub_genre = ''
                    for term in terms:
                        term_factor = float(terms_freq[terms.index(term)]) * float(
                            terms_weight[terms.index(term)])
                        if term_factor >= term_factor_max:
                            term_factor_max = term_factor
                            sub_sub_genre = term
                    if sub_sub_genre in genre_router:
                        self.__midi_group[genre_router[sub_sub_genre]['GENRE']][
                            genre_router[sub_sub_genre]['SUBGENRE']][
                                sub_sub_genre].add(msd_id)
                    i += 1
                    print('\r' + str(i).zfill(len(str(nr_of_scores))) + '/' + str(
                        nr_of_scores) + ' = ' + str(
                            int(i / nr_of_scores * 100)).zfill(3) + '%', end='', flush=True)
            print('\n')
            with open(self.__midi_dict_path, mode='wb') as file:
                pickle.dump(self.__midi_group, file)

    def get_midi_group(self):
        """
        Genereic get-method for the midi_group field.

        :return:  dict<str:dict<str:dict<str:set<>>>>
        """
        return self.__midi_group

    def __open_midi(self, msd_id, remove_drums=False):
        """
        Opens the best fitting non error producing midi file specified with msd_id.
        For Chord analysis it is recommend to remove the drums.

        :param msd_id: str
        :param remove_drums: boolean
        :return: music21.stream.Part
        """
        result = None
        midi_files = self.__get_fitting(msd_id)
        for midi_file_path in midi_files:
            try:
                midi_file = midi.MidiFile()
                midi_file.open(midi_file_path)
                midi_file.read()
                midi_file.close()
                if remove_drums:
                    for i, _ in enumerate(midi_file.tracks):
                        midi_file.tracks[i].events = [event for event in midi_file.tracks[i].events
                                                      if event.channel != 10]
                result = midi.translate.midiFileToStream(midi_file)
            except (IndexError, midi.MidiException, MeterException):
                continue
            else:
                break
        return result

    def __note_count(self, measure, count_dict):
        """
        Counts all notes within a measure. Note that the values are stored within the count_dict
        parameter and the retun value represents only the most frequent note.

        :param measure: music21.stream.Measure
        :param count_dict: dict<str:int>
        :return: str
        """
        bass_note = None
        for chrd in measure.recurse().getElementsByClass('Chord'):
            note_length = chrd.quarterLength
            for note in chrd.pitches:
                note_name = str(note)
                if bass_note is None or bass_note.ps > note.ps:
                    bass_note = note
                if note_name in count_dict:
                    count_dict[note_name] += note_length
                else:
                    count_dict[note_name] = note_length
        return bass_note

    def __simplify_roman_name(self, roman_numeral):
        """
        Given a roman numeral representation of a chord it will get simplified and the be returned.

        :param roman_numeral: music21.roman.RomanNumeral
        :return: str
        """
        ret = roman_numeral.romanNumeral
        inversion_name = None
        inversion = roman_numeral.inversion()

        if (roman_numeral.isTriad() and inversion < 3) or (
                inversion < 4 and (roman_numeral.seventh is not None or roman_numeral.isSeventh())):
            inversion_name = roman_numeral.inversionName()

        if inversion_name is not None:
            ret = ret + str(inversion_name)
        elif roman_numeral.isDominantSeventh():
            ret = ret + 'M7'
        elif roman_numeral.isDiminishedSeventh():
            ret = ret + 'o7'
        return ret

    def __get_split_dict(self, midi_stream):
        """
        Creates a dict that specifies at what offset a midi stream should be split because of a
        change in either BPM or Time Signature.

        :param midi_stream: music21.stream.Part
        :return: collections.OrderedDict<float:list<>>
        """
        result = OrderedDict()
        time_signatures_count = dict()
        offset_tolerance = 20.0
        duration_tolerance = 5.0
        bpm_tolerance = 10.0

        delete_set = set()
        last_offset = -1.0
        for time_signature in midi_stream.getTimeSignatures(sortByCreationTime=True):
            last_duration = time_signature.offset - last_offset
            if last_duration < duration_tolerance and last_offset not in (0.0, -1.0):
                delete_set.add(last_offset)
            if time_signature.offset is not last_offset and last_offset is not -1.0:
                result[last_offset] = [-1.0,
                                       max(time_signatures_count, key=time_signatures_count.get)]
                time_signatures_count = dict()
            if time_signature.ratioString not in time_signatures_count:
                time_signatures_count[time_signature.ratioString] = 0
            time_signatures_count[time_signature.ratioString] += 1
            last_offset = time_signature.offset
        result[last_offset] = [-1.0, max(time_signatures_count, key=time_signatures_count.get)]

        last_offset = -offset_tolerance
        last_bpm = -bpm_tolerance
        for tempo_event in midi_stream.metronomeMarkBoundaries():
            offset = tempo_event[0]
            duration = tempo_event[1] - offset
            bpm = tempo_event[2].number
            if duration < duration_tolerance:
                continue
            if abs(midi_stream.quarterLength - offset) >= offset_tolerance * 1.5 \
                    and offset >= offset_tolerance:
                if abs(last_bpm - bpm) >= bpm_tolerance or offset in result.keys():
                    if abs(last_offset - offset) >= offset_tolerance or offset in result.keys():
                        if offset in result.keys():
                            result[offset][0] = bpm
                        else:
                            result[offset] = [bpm, -1.0]
            last_bpm = bpm
            last_offset = offset

        for offset in delete_set:
            del result[offset]

        delete_list = []
        last_values = [120, '4/4']
        for offset, values in result.items():
            if values[0] is -1.0:
                result[offset][0] = last_values[0]
            elif values[1] is -1.0:
                result[offset][1] = last_values[1]
            if values == last_values and offset != 0.0:
                delete_list.append(offset)
            last_values = values

        for offset in delete_list:
            del result[offset]

        result = OrderedDict(sorted(result.items()))
        return result

    def __split_midi(self, midi_stream, split_list):
        """
        Splits a midi stream according to the split list.

        :param midi_stream: music21.stream.Part
        :param split_list: list<float>
        :return: list<music21.stream.Part>
        """
        midi_stream_list = [midi_stream]
        for i, split in enumerate(split_list):
            part = midi_stream_list[i].splitAtQuarterLength(quarterLength=split, retainOrigin=True)
            midi_stream_list[i] = part[0]
            midi_stream_list.append(part[1])
        return midi_stream_list

    def analyze_file(self, msd_id):
        """
        Analyzes the best associated midi file with the give msd_id in hindsight of Key/Scale, BPM
        and the chord progression. If a change of BPM or Time Signature occurs the Midi file will be
        split to ensure that the information only gets analyzed for consistent pieces of music.

        :param msd_id: str
        :return: list<str>
        """
        midi_file = self.__open_midi(msd_id, remove_drums=True)
        if midi_file is None:
            return []
        tmp_midi = midi_file.chordify()

        split_dict = self.__get_split_dict(tmp_midi)
        split_list = list(split_dict.keys())
        split_list.remove(0.0)
        tmp_midis = self.__split_midi(tmp_midi, split_list)

        results = [(" ".join(str(event) for event in events)) for events in
                   list(split_dict.values())]
        delete_list = []
        for i, segment in enumerate(tmp_midis):
            try:
                key = tmp_midis[i].analyze('key')
            except DiscreteAnalysisException:
                delete_list.append(i)
                continue
            results[i] += ' ' + str(key.tonic) + str(key.mode)
            max_notes_per_chord = 4
            try:
                for measure in segment.measures(0, None):
                    if type(measure) != stream.Measure:
                        continue
                    count_dict = dict()
                    self.__note_count(measure, count_dict)
                    if not any(count_dict):
                        results[i] += ' -'
                        continue

                    sorted_items = sorted(count_dict.items(), key=lambda x: x[1])
                    sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
                    for j, sorted_note in enumerate(sorted_notes):
                        while True:
                            if '#-' in sorted_notes[j]:
                                sorted_notes[j] = sorted_note.replace("#-", "")
                            elif '-#' in sorted_notes[j]:
                                sorted_notes[j] = sorted_note.replace("-#", "")
                            else:
                                break
                    measure_chord = chord.Chord(sorted_notes)
                    roman_numeral = roman.romanNumeralFromChord(measure_chord, key)
                    results[i] += ' ' + self.__simplify_roman_name(roman_numeral)
            except StreamException:
                delete_list.append(i)
        for i in reversed(delete_list):
            del results[i]
        return results

    def analyze_batch(self, genre=''):
        """
        Analyzes all files associated to a specified genre.
        This is done multit-hreaded as the analyze process takes some computing time.
        The chord progression information get then stored at DATA_PATH/CHORDS_PATH/ANALYSIS_PATH/.

        :param genre: str
        """
        if self.__midi_group is None:
            self.group_midi()
        pool = Pool(cpu_count() - 1)
        for gnre, sub_genres in self.__midi_group.items():
            if genre in ('', gnre):
                print(gnre)
                genre_path = os.path.join(self.__data_path, self.__chords_path,
                                          self.__analysis_path, gnre + '.txt')
                if not os.path.exists(os.path.dirname(genre_path)):
                    os.makedirs(os.path.dirname(genre_path))
                genre_file = open(genre_path, 'w')
                for sub_genre, sub_sub_genres in sub_genres.items():
                    print('\t' + sub_genre)
                    sub_genre_path = os.path.join(self.__data_path, self.__chords_path,
                                                  self.__analysis_path, gnre,
                                                  sub_genre + '.txt')
                    if not os.path.exists(os.path.dirname(sub_genre_path)):
                        os.makedirs(os.path.dirname(sub_genre_path))
                    sub_genre_file = open(sub_genre_path, 'w')
                    for sub_sub_genre, msd_ids in sub_sub_genres.items():
                        i = 1
                        sub_sub_genre_path = os.path.join(self.__data_path, self.__chords_path,
                                                          self.__analysis_path,
                                                          gnre, sub_genre, sub_sub_genre + '.txt')
                        if not os.path.exists(os.path.dirname(sub_sub_genre_path)):
                            os.makedirs(os.path.dirname(sub_sub_genre_path))
                        sub_sub_genre_file = open(sub_sub_genre_path, 'w')
                        for result in pool.imap_unordered(self.analyze_file, msd_ids):
                            print('\r\t\t' + sub_sub_genre + ' - ' + str(i).zfill(
                                len(str(len(msd_ids)))) + '/' + str(
                                    len(msd_ids)) + ' = ' + str(int(i / len(msd_ids) * 100)).zfill(
                                        3) + '%', end='',
                                  flush=True)
                            i += 1
                            for analysis in result:
                                genre_file.write(analysis + '\n')
                                sub_genre_file.write(analysis + '\n')
                                sub_sub_genre_file.write(analysis + '\n')
                        print('\n')
                        sub_sub_genre_file.close()
                    sub_genre_file.close()
                genre_file.close()
        pool.close()


def main():
    """
    Loads constants dictionary and analyses all midi files or only the ones of the specified
    genre list.
    """
    constants_cfg = util.load_constants()
    cpp = ChordProgressionProcessor(constants_cfg)
    cpp.group_midi()
    genre_list = list(cpp.get_midi_group().keys())
    if len(sys.argv) != 1:
        for arg in sys.argv:
            if arg in genre_list:
                cpp.analyze_batch(genre=arg)
    else:
        cpp.analyze_batch()


if __name__ == '__main__':
    freeze_support()
    main()
