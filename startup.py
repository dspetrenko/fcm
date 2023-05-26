import os

import boto3
import dotenv

from src.agbmfc.inference import MODEL_NAME

dotenv.load_dotenv()

ACCESS_KEY_ID = os.getenv('YA_SERVICE_KEY_ID', 'unknown_key_id')
ACCESS_SECRET = os.getenv('YA_SERVICE_KEY_TOKEN', 'unknown_key_token')


session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_SECRET,
)

response = s3.get_object(Bucket='fcm', Key=MODEL_NAME)

model_path = os.path.join('models', MODEL_NAME)
with open(model_path, 'wb') as fd:
    fd.write(response['Body'].read())
