from os.path import basename
import requests


train_files = [
    'train_features/fed6bb57_S1_00.tif',
    'train_features/fed6bb57_S1_01.tif',
    'train_features/fed6bb57_S1_02.tif',
    'train_features/fed6bb57_S1_03.tif',
    'train_features/fed6bb57_S1_04.tif',
    'train_features/fed6bb57_S1_05.tif',
    'train_features/fed6bb57_S1_06.tif',
    'train_features/fed6bb57_S1_07.tif',
    'train_features/fed6bb57_S1_08.tif',
    'train_features/fed6bb57_S1_09.tif',
    'train_features/fed6bb57_S1_10.tif',
    'train_features/fed6bb57_S1_11.tif',
    'train_agbm/fed6bb57_agbm.tif',
]

for f in train_files:
    url = rf'https://drivendata-competition-biomassters-public-eu.s3.amazonaws.com/{f}'
    response = requests.get(url)
    if response.status_code == 200:
        with open(rf'data_sample/{basename(f)}', 'wb') as fd:
            fd.write(response.content)

    print(f'{f} - {response.status_code}')
