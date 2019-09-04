import os

import chord_progression_generator
import structure_generator
import util


class SongStructure:
    def __init__(self, constants_cfg):
        """

        :param constants_cfg: dict<str:str>
        """
        self.__constants_cfg = constants_cfg
        self.__data_path = constants_cfg['DATA_PATH']
        self.__structure_path = constants_cfg['STRUCTURE_PATH']
        self.__chords_path = constants_cfg['CHORDS_PATH']
        self.__relative_path = constants_cfg['RELATIVE_PATH']
        self.__analysis_path = constants_cfg['ANALYSIS_PATH']
        self.__output_path = constants_cfg['OUTPUT_PATH']

    def generate(self, structure_genre, chords_genre, structure_sub_genre='', chords_sub_genre=''):
        """

        :param structure_genre: str
        :param structure_sub_genre: str
        :param chords_genre: str
        :param chords_sub_genre: str
        :return:
        """
        structure_gen = structure_generator.create_generator(self.__constants_cfg)
        chords_gen = chord_progression_generator.create_generator(self.__constants_cfg)

        structure_model = structure_gen.load_model(structure_genre, structure_sub_genre)
        chords_model = chords_gen.load_model(chords_genre, chords_sub_genre)

        structure = structure_gen.generate_text_to_file(structure_model)

        result = list()
        sections = dict()
        last_section = ''
        new_section = False
        solo_id = 1
        for section in structure:
            if section == 'SOLO':
                section += str(solo_id)
            if section != last_section:
                if 'SOLO' in last_section:
                    solo_id += 1
                if section not in sections.keys():
                    sections[section] = 1
                    new_section = True
                else:
                    new_section = False
                result.append(section)
            elif new_section:
                sections[section] += 1
            last_section = section

        for i, section in enumerate(result):
            result[i] += ' ' + ', '.join(
                chords_gen.generate_text_to_file(chords_model, sections[section]))

        return result

    def generate_for_strucutre(self, structure_file_path, structure_genre, chords_genre):
        """

        :param structure_file_path:str
        :param structure_genre: str
        :param chords_genre: str
        :return:
        """
        if not os.path.exists(structure_file_path):
            return None
        structure = list()
        with open(structure_file_path, mode='r') as file:
            for line in file:
                structure.append(line)

        return None


def main():
    constants = util.load_constants()
    song_structure = SongStructure(constants)


if __name__ == '__main__':
    main()
