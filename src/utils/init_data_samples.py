from os.path import basename
import requests

CHIP = 'fed6bb57'

periods = [f'{idx:02}' for idx in range(12)]
train_files = [f'train_features/{CHIP}_S2_{period}.tif' for period in periods]
train_files.append(f'train_agbm/{CHIP}_agbm.tif')

for f in train_files:
    url = rf'https://drivendata-competition-biomassters-public-eu.s3.amazonaws.com/{f}'
    response = requests.get(url)
    if response.status_code == 200:
        with open(rf'data_sample/{basename(f)}', 'wb') as fd:
            fd.write(response.content)

    print(f'{f} - {response.status_code}')
