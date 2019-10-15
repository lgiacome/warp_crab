import numpy as np


def find_line(filename, string):
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    
    pos = None
    for ii, line in enumerate(lines):
        if string in line:
            pos = ii
            return lines, pos
    print('Did not find: %s' % string)
    return None, None

def pos2dic(filename):
    title = '!    Changed by config_scan.py'
    lines, pos = find_line(filename, title)
    if title not in lines[pos]:
        raise ValueError('stg went wrong')
    
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    
    dict = {}
    dict['matsurf'] = float(lines[pos-2].split(' ')[1])
    line = lines[pos+1].split('  ')
    dict['E0tspk'] = float(line[0])
    dict['dtspk'] = float(line[1])
    dict['powts'] = float(line[2])
    dict['iprob'] = float(line[3])
    dict['tpar'] =  [float(i) for i in lines[pos+2].split(' ')[0:6]]
    dict['enpar'] = [float(i) for i in lines[pos+3].split(' ')[0:10]]
    dict['pnpar'] = [float(i) for i in lines[pos+4].split(' ')[0:10]]
    line = lines[pos+5].split(' ')
    dict['P1einf'] = float(line[0])
    dict['P1epk'] = float(line[1])
    dict['E0epk'] = float(line[2])
    dict['E0w'] = float(line[3])
    dict['powe'] = float(line[4])
    dict['epar'] = [float(i) for i in line[5:7]]
    dict['sige'] = float(line[7])
    line = lines[pos+6].split(' ')
    dict['Ecr'] = float(line[0])
    dict['P1rinf'] = float(line[1])
    dict['qr'] = float(line[2])
    dict['rpar'] = [float(i) for i in line[3:5]]
    dict['pr'] = float(line[5])
    line = lines[pos+25].split(', ')
    dict['dtotpk'] = float(line[0])
    remain = line[1].split(' ')
    dict['pangsec'] = float(remain[0])
    return dict
