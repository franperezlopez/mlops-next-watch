import os
from typing import List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from sqlalchemy import Column, Integer, MetaData, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from conf import globals, paths

app = FastAPI()

load_dotenv(
    paths.get_path(
        paths.ENV,
        storage=globals.Storage.HOST,
    )
)

# Database connection
DATABASE_URL = f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_IP']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_APP_DATABASE']}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Define the SQLAlchemy models
Base = declarative_base()


# Dependency
class User(Base):
    __tablename__ = "users"
    userId = Column(Integer, primary_key=True, index=True)


metadata = MetaData()


@app.get("/user_ids/", response_model=List[int])
def get_user_ids(db: Session = Depends(get_db)):
    print("yo")
    users_list = [user_id for user_id, in db.query(User.userId).all()]
    print(users_list)
    return users_list


@app.post("/add_user/", response_model=None)
def add_user(user_id: int, db: Session = Depends(get_db)):
    db_user = User(userId=user_id)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    print("yo")
    return db_user
