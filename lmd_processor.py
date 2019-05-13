import csv
import os
import json
import h5py
import pickle
from music21 import midi, stream, roman, chord, note, pitch
from music21.pitch import AccidentalException
from music21.exceptions21 import StreamException
from multiprocessing import Pool, cpu_count, freeze_support
import re


class LmdUtils:
    def __init__(self, dataset_path='datasets', data_path='data_parsed', analysis_path='analysis',
                 matched_path='lmd_matched', h5_path='lmd_matched_h5', score_file='match_scores.json',
                 chords_path='chords', lmd_path='lmd', genre_list_file='genre_list.txt',
                 possible_list_file='possible_list.txt', midi_dict='midi_dict.pkl'):
        self.__dataset_path = dataset_path
        self.__data_path = data_path
        self.__lmd_path = lmd_path
        self.__matched_path = matched_path
        self.__h5_path = h5_path
        self.__chords_path = chords_path
        self.__analysis_path = analysis_path
        self.__genre_list = os.path.join(self.__data_path, genre_list_file)
        self.__possible_list = os.path.join(self.__data_path, possible_list_file)
        self.__score_file = os.path.join(self.__dataset_path, self.__lmd_path, score_file)
        self.__midi_dict = os.path.join(self.__data_path, midi_dict)
        self.__midi_group = None

    def __msd_id_to_dirs(self, msd_id):
        """Given an MSD ID, generate the path prefix. E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
        return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

    def __msd_id_to_h5(self, msd_id):
        """Given an MSD ID, return the path to the corresponding h5"""
        return os.path.join(self.__dataset_path, self.__lmd_path, self.__h5_path, self.__msd_id_to_dirs(msd_id) + '.h5')

    def __get_midi_path(self, msd_id, midi_md5):
        """Given an MSD ID and MIDI MD5, return path to a MIDI file."""
        return os.path.join(self.__dataset_path, self.__lmd_path, self.__matched_path, self.__msd_id_to_dirs(msd_id),
                            midi_md5 + '.mid')

    def __get_midi_dir(self, msd_id):
        """Given an MSD ID, return path to a MIDI directory."""
        return os.path.join(self.__dataset_path, self.__lmd_path, self.__matched_path, self.__msd_id_to_dirs(msd_id))

    def __get_most_fitting(self, msd_id):
        """Given an MSD ID, return path to best fitting midi."""
        best_md5 = 'No sufficient match has been found'
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
        if (str(h5_entry).startswith('b\'') or str(h5_entry).startswith('b\'')) and \
                (str(h5_entry).endswith('\'') or str(h5_entry).endswith('\'')):
            return str(h5_entry)[2:-1]
        return h5_entry

    def __clean_array(self, h5_array):
        """Uses clean_str() on an array"""
        return [self.__clean_str(h5_entry) for h5_entry in h5_array]

    def __generate_genre_router(self):
        genre_router = dict()
        with open(self.__genre_list, 'r') as pos:
            reader = csv.DictReader(pos)
            for line in reader:
                genre_router[line['SUBSUBGENRE']] = {'SUBGENRE': line['SUBGENRE']}
        for t, tt in genre_router.items():
            with open(self.__possible_list, 'r') as gl:
                reader = csv.DictReader(gl)
                for line in reader:
                    if line['SUBGENRE'] == tt['SUBGENRE']:
                        genre_router[t]['GENRE'] = line['GENRE']
                        break
        return genre_router

    def __generate_entries_skeleton(self):
        entries = dict()
        with open(self.__possible_list, 'r') as pos:
            reader = csv.DictReader(pos)
            for line in reader:
                if line['GENRE'] in entries:
                    entries[line['GENRE']][line['SUBGENRE']] = dict()
                else:
                    entries[line['GENRE']] = {line['SUBGENRE']: dict()}
        with open(self.__genre_list, 'r') as gl:
            reader = csv.DictReader(gl)
            for line in reader:
                for t, tt in entries.items():
                    if line['SUBGENRE'] in tt:
                        entries[t][line['SUBGENRE']][line['SUBSUBGENRE']] = set()
        return entries

    def group_midi(self, new=False):
        if not new and os.path.exists(self.__midi_dict):
            with open(self.__midi_dict, mode='rb') as file:
                self.__midi_group = pickle.load(file)
        else:
            genre_router = self.__generate_genre_router()
            self.__midi_group = self.__generate_entries_skeleton()
            with open(self.__score_file) as f:
                scores = json.load(f)
                nr_of_scores = len(scores.items())
                i = 0
                print('Grouping MIDI files by Genre')
                for msd_id, midis in scores.items():
                    h5 = h5py.File(self.__msd_id_to_h5(msd_id))
                    terms = self.__clean_array(h5['metadata']['artist_terms'][0:])
                    terms_weight = self.__clean_array(h5['metadata']['artist_terms_weight'][0:])
                    terms_freq = self.__clean_array(h5['metadata']['artist_terms_freq'][0:])
                    term_factor_max = 0.0
                    genre = ''
                    for term in terms:
                        term_factor = float(terms_freq[terms.index(term)]) * float(terms_weight[terms.index(term)])
                        if term_factor >= term_factor_max:
                            term_factor_max = term_factor
                            genre = term
                    if genre in genre_router:
                        self.__midi_group[genre_router[genre]['GENRE']][genre_router[genre]['SUBGENRE']][
                            genre].add(msd_id)
                    i += 1
                    print('\r' + str(i).zfill(len(str(nr_of_scores))) + '/' + str(nr_of_scores) + ' = ' + str(
                        int(i / nr_of_scores * 100)).zfill(3) + '%', end='', flush=True)
            print('\n')
            with open(self.__midi_dict, mode='wb') as file:
                pickle.dump(self.__midi_group, file)

    def get_midi_group(self):
        return self.__midi_group

    def open_midi(self, msd_id, remove_drums=False):
        mf = midi.MidiFile()
        mf.open(self.__get_most_fitting(msd_id))
        mf.read()
        mf.close()
        if remove_drums:
            for i in range(len(mf.tracks)):
                mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]
        return midi.translate.midiFileToStream(mf)

    def __note_count(self, measure, count_dict):
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

    def analyze_file(self, msd_id):
        try:
            midi_file = self.open_midi(msd_id, remove_drums=True)
        except IndexError:
            return []
        tmp_midi = stream.Score()
        tmp_midi_chords = midi_file.chordify()
        tmp_midi.insert(0, tmp_midi_chords)

        time_signatures = dict()
        time_signatures_count = dict()
        last_offset = -1.0
        for ts in midi_file.getTimeSignatures(sortByCreationTime=True):
            if ts.offset is not last_offset and last_offset is not -1.0:
                time_signatures[last_offset] = max(time_signatures_count, key=time_signatures_count.get, default=0)
                time_signatures_count = dict()
            if ts.ratioString not in time_signatures_count:
                time_signatures_count[ts.ratioString] = 0
            time_signatures_count[ts.ratioString] += 1
            last_offset = ts.offset
        time_signatures[last_offset] = max(time_signatures_count, key=time_signatures_count.get)

        split_list = list(time_signatures.keys())
        split_list.remove(0.0)
        if len(split_list) is not 0:
            tmp_midis = tmp_midi.splitByQuarterLengths(quarterLengthList=split_list)
            tmp_midis_chords = tmp_midi_chords.splitByQuarterLengths(quarterLengthList=split_list)
        else:
            tmp_midis = [tmp_midi]
            tmp_midis_chords = [tmp_midi_chords]

        results = list(time_signatures.values())
        for i, segment in enumerate(tmp_midis_chords):
            key = tmp_midis[i].analyze('key')
            results[i] += ' ' + str(key.tonic) + str(key.mode)
            max_notes_per_chord = 4
            try:
                for m in segment.measures(0, None):
                    if type(m) != stream.Measure:
                        continue
                    count_dict = dict()
                    bass_note = self.__note_count(m, count_dict)
                    if len(count_dict) < 1:
                        results[i] += ' -'
                        continue

                    sorted_items = sorted(count_dict.items(), key=lambda x: x[1])
                    sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
                    for j, sorted_note in enumerate(sorted_notes):
                        while True:
                            if '#-' in sorted_notes[j]:
                                # print("\nError on {0}".format(msd_id))
                                # print('Unsupported Accidental \'#-\'')
                                sorted_notes[j] = sorted_note.replace("#-", "")
                            elif '-#' in sorted_notes[j]:
                                # print("\nError on {0}".format(msd_id))
                                # print('Unsupported Accidental \'-#\'')
                                sorted_notes[j] = sorted_note.replace("-#", "")
                            else:
                                break
                    measure_chord = chord.Chord(sorted_notes)
                    roman_numeral = roman.romanNumeralFromChord(measure_chord, key)
                    results[i] += ' ' + self.__simplify_roman_name(roman_numeral)
            except StreamException as e:
                # print("\nError on {0}".format(msd_id))
                # print(e)
                del results[i]
        return results

    def analyze_batch(self):
        if self.__midi_group is None:
            self.group_midi()
        pool = Pool(cpu_count() - 1)
        for genre, sub_genres in self.__midi_group.items():
            print(genre)
            genre_path = os.path.join(self.__data_path, self.__chords_path, self.__analysis_path, genre + '.txt')
            if not os.path.exists(os.path.dirname(genre_path)):
                os.makedirs(os.path.dirname(genre_path))
            genre_file = open(genre_path, 'w')
            for sub_genre, sub_sub_genres in sub_genres.items():
                print('\t' + sub_genre)
                sub_genre_path = os.path.join(self.__data_path, self.__chords_path, self.__analysis_path, genre,
                                              sub_genre + '.txt')
                if not os.path.exists(os.path.dirname(sub_genre_path)):
                    os.makedirs(os.path.dirname(sub_genre_path))
                sub_genre_file = open(sub_genre_path, 'w')
                for sub_sub_genre, msd_ids in sub_sub_genres.items():
                    i = 1
                    for result in pool.imap_unordered(self.analyze_file, msd_ids):
                        print('\r\t\t' + sub_sub_genre + ' - ' + str(i).zfill(len(str(len(msd_ids)))) + '/' + str(
                            len(msd_ids)) + ' = ' + str(int(i / len(msd_ids) * 100)).zfill(3) + '%', end='', flush=True)
                        i += 1
                        for analysis in result:
                            genre_file.write(analysis + '\n')
                            sub_genre_file.write(analysis + '\n')
                    print('\n')
                sub_genre_file.close()
            genre_file.close()
        pool.close()


def main():
    constants = dict()
    with open('constants.cfg', mode='r') as f:
        for line in f:
            if line[0] is not '#':
                tmp = line.rstrip('\n').split('=')
                constants[tmp[0]] = tmp[1]
    lmd = LmdUtils(dataset_path=constants['DATASET_PATH'],
                   data_path=constants['DATA_PATH'],
                   lmd_path=constants['LMD_PATH'],
                   matched_path=constants['MATCHED_PATH'],
                   h5_path=constants['H5_PATH'],
                   score_file=constants['SCORE_FILE'],
                   genre_list_file=constants['GENRE_LIST_FILE'],
                   possible_list_file=constants['POSSIBLE_LIST_FILE'],
                   midi_dict=constants['MIDI_DICT'],
                   chords_path=constants['CHORDS_PATH'],
                   analysis_path=constants['ANALYSIS_PATH'])
    lmd.group_midi()
    lmd.analyze_batch()


if __name__ == '__main__':
    freeze_support()
    main()
