def load_constants():
    constants_cfg = dict()
    with open('constants.cfg', mode='r') as file:
        for line in file:
            if line[0] != '#':
                tmp = line.rstrip('\n').split('=')
                constants_cfg[tmp[0]] = tmp[1]
    return constants_cfg
