import os

from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

_DATABASE_USER = os.getenv('POSTGRES_USER')
_DATABASE_PASSWORD = os.getenv('POSTGRES_PASSWORD')
_DATABASE_DB = os.getenv('POSTGRES_DB')

DATABASE_URL = f'postgresql+psycopg2://{_DATABASE_USER}:{_DATABASE_PASSWORD}@localhost:5432/{_DATABASE_DB}'

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autoflush=False, bind=engine)
Base = declarative_base()


if __name__ == '__main__':
    # TODO: move it in the tests

    from src.app import get_db
    from src.service.crud import create_user
    from src.service import schemas

    db = get_db().__next__()
    u = schemas.UserCreate.parse_obj({'username': 'shepard', 'email': 'shepard@space.com', 'password': 'space1'})

    create_user(db, u)
    db.close()


