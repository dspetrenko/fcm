import requests
import glob
from os.path import basename

END_POINT = 'http://localhost:8000/agbmfc/inference'

files = sorted(glob.glob(r'data_sample/*S1*.tif'))
print('files:')
for file in files:
    print('\t', basename(file))
print('-' * 50)


multiple_files = []
for file in files:
    with open(file, 'rb') as fd:
        multiple_files.append(('chip_files', (basename(file), fd.read())))


r = requests.post(END_POINT, files=multiple_files)

print(r.status_code)
print(r.content)
