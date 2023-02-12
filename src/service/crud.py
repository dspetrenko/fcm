import hashlib

from sqlalchemy.orm import Session

import src.service.models as models
import src.service.schemas as schemas


def create_user(db: Session, user: schemas.UserCreate):
    hashed_pass = hashlib.sha256(user.password.encode()).hexdigest()
    db_user = models.User(username=user.username,
                          password_hash=hashed_pass,
                          email=user.email,
                          )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


def get_users(db: Session):
    return db.query(models.User).all()

