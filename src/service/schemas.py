from pydantic import BaseModel


class UserBase(BaseModel):
    username: str
    email: str


class User(UserBase):
    class Config:
        orm_mode = True


class UserCreate(UserBase):
    password: str
