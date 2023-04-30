import argparse

import requests
import glob
from os.path import basename

argParser = argparse.ArgumentParser()
argParser.add_argument("-m", "--model_type", help="Model that will be used for prediction",
                       choices=['baseline-pixel', 'trivial'],
                       default='trivial')
args = argParser.parse_args()
params = {'model_type': args.model_type}

END_POINT = 'http://localhost:8000/agbmfc/inference'

files = sorted(glob.glob(r'data_sample/*S2*.tif'))
print('files:')
for file in files:
    print('\t', basename(file))
print('-' * 50)


multiple_files = []
for file in files:
    with open(file, 'rb') as fd:
        multiple_files.append(('chip_files', (basename(file), fd.read())))


r = requests.post(END_POINT, files=multiple_files, params=params)

if r.status_code == 200:
    with open('response.tif', 'wb') as fd:
        fd.write(r.content)

    print('response code: 200 - response content saved to response.tif')
else:
    print(f'response code: {r.status_code} - something went wrong')
