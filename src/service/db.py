import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

_DATABASE_USER = os.getenv('POSTGRES_USER', 'unknown')
_DATABASE_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'unknown_pass')
_DATABASE_DB = os.getenv('POSTGRES_DB', 'unknown_pg')
_DATABASE_HOST = os.getenv('POSTGRES_HOST', 'unknown_host')

if _DATABASE_USER == 'unknown':
    # we didn't pick up .env file, so it mean that we runs loacally as 'python db.py' command
    ...
    import pathlib
    path = pathlib.Path(__file__).parent.parent.parent / '.env'
    print(path)
    load_dotenv(path)

    _DATABASE_USER = os.getenv('POSTGRES_USER', 'unknown')
    _DATABASE_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'unknown_pass')
    _DATABASE_DB = os.getenv('POSTGRES_DB', 'unknown_pg')
    _DATABASE_HOST = 'localhost'


DATABASE_URL = f'postgresql+psycopg2://{_DATABASE_USER}:{_DATABASE_PASSWORD}@{_DATABASE_HOST}:5432/{_DATABASE_DB}'

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autoflush=False, bind=engine)
Base = declarative_base()


if __name__ == '__main__':

    print(DATABASE_URL)

    # TODO: move it in the tests

    from src.app import get_db
    from src.service.crud import create_user
    from src.service import schemas

    db = get_db().__next__()
    u = schemas.UserCreate.parse_obj({'username': 'shepard', 'email': 'shepard@space.com', 'password': 'space1'})

    create_user(db, u)
    db.close()


