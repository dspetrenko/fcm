from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship

from src.service.db import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, autoincrement=True, primary_key=True, index=True, unique=True)
    username = Column(String, unique=True)
    password_hash = Column(String)
    email = Column(String)


class Token(Base):
    __tablename__ = 'tokens'

    id = Column(Integer, autoincrement=True, primary_key=True, index=True, unique=True)
    token_hash = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))

    user = relationship("User")
