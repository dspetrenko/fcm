import requests

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--id", help="task id for task")

args = argParser.parse_args()
if args.id is None:
    print('Error: id is necessary')

END_POINT = f'http://localhost:8000/agbmfc/inference/result?task_id={args.id}'
print(END_POINT)

r = requests.get(END_POINT)

if r.status_code == 200:
    with open('response.tif', 'wb') as fd:
        fd.write(r.content)

    print('response code: 200 - response content saved to response.tif')
else:
    print(f'response code: {r.status_code} - something went wrong')
