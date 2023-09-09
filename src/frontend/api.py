import os
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from prometheus_fastapi_instrumentator import Instrumentator

from conf import catalog, globals, paths

app = FastAPI()
Instrumentator().instrument(app).expose(app)

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
    __tablename__ = os.getenv("POSTGRES_USERS_TABLE")
    userId = Column(Integer, primary_key=True, index=True)


class Recommendations(Base):
    __tablename__ = os.getenv("POSTGRES_RECOMMENDATIONS_TABLE")
    id = Column(Integer, index=True, primary_key=True)
    userid = Column(Integer)
    movieid = Column(Integer)
    prediction = Column(Float)
    rank = Column(Integer)
    __table_args__ = (UniqueConstraint("userid", "rank", name="unique_user_rank"),)


metadata = MetaData()


movies_df = pd.read_csv(
    paths.get_path(
        paths.DATA_01EXTERNAL,
        catalog.Sources.MOVIELENS,
        catalog.Datasets.MOVIES,
        suffix=catalog.FileFormat.CSV,
        storage=globals.Storage.DOCKER,
        as_string=True,
    )
)


@app.get("/user_ids/", response_model=List[int])
def get_user_ids(db: Session = Depends(get_db)):
    users_list = [user_id for user_id, in db.query(User.userId).all()]
    print(users_list)
    return users_list


@app.get("/recommendations/", response_model=List[Tuple])
def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    recommendations = db.query(Recommendations).filter(
        Recommendations.userid == user_id, Recommendations.rank <= 5
    )
    a = [
        (
            r.id,
            r.userid,
            r.movieid,
            movies_df[movies_df["movieId"] == r.movieid]["title"].item(),
            r.prediction,
            r.rank,
        )
        for r in recommendations
    ]
    print("\n\n ********fastapi***********: ", a)
    return a


@app.post("/add_user/", response_model=None)
def add_user(user_id: int, db: Session = Depends(get_db)):
    db_user = User(userId=user_id)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.get("/movies_list/", response_model=List[str])
def get_movies_list():
    return movies_df["title"].unique().tolist()


@app.post("/add_rating/", response_model=str)
def add_rating(rating: float, selected_movie: str, user_id: int):
    ratings_df_path = paths.get_path(
        paths.DATA_01EXTERNAL,
        catalog.Sources.MOVIELENS,
        catalog.Datasets.RATINGS,
        suffix=catalog.FileFormat.CSV,
        storage=globals.Storage.S3,
        as_string=True,
    )
    ratings_df = pd.read_csv(ratings_df_path)  # eeeeee

    movie_row = movies_df[movies_df["title"] == selected_movie]

    matching_rows = ratings_df[
        (ratings_df["movieId"] == movie_row["movieId"].item())
        & (ratings_df["userId"] == user_id)
    ]

    new_row = {
        "userId": user_id,
        "movieId": movie_row["movieId"].item(),
        "rating": rating,
        "timestamp": 122334,
        "datetime": pd.to_datetime(ratings_df["datetime"].max())
        + pd.to_timedelta(1, unit="m"),  # ()
    }

    # If there is a match, replace the existing row with the new values
    ratings_df.loc[
        matching_rows.index[0] if not matching_rows.empty else len(ratings_df)
    ] = new_row

    remote_ratings_path = paths.get_path(
        paths.DATA_01EXTERNAL,
        catalog.Sources.MOVIELENS,
        catalog.Datasets.RATINGS,
        suffix=catalog.FileFormat.CSV,
        storage=globals.Storage.S3,
        as_string=True,
        s3_protocol=globals.Protocols.S3,
    )
    # Save new ratings...
    ratings_df.to_csv(
        remote_ratings_path,
        index=False,
        storage_options={
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        },
    )

    return "Rating saved!!!"
