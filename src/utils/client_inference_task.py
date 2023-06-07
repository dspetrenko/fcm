import requests
import glob
from os.path import basename

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--server", help="localhost or host ip", default='localhost')

args = argParser.parse_args()

END_POINT = f'http://{args.server}:8000/agbmfc/inference/task'


files = sorted(glob.glob(r'data_sample/*S2*.tif'))
print('files:')
for file in files:
    print('\t', basename(file))
print('-' * 50)


multiple_files = []
for file in files:
    with open(file, 'rb') as fd:
        multiple_files.append(('chip_files', (basename(file), fd.read())))


r = requests.post(END_POINT, files=multiple_files)

if r.status_code == 200:
    print(r.json()['task_id'])
else:
    print(f'response code: {r.status_code} - something went wrong')
