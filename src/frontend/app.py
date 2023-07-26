import pandas as pd
import requests
import streamlit as st

from conf import catalog, globals, paths

st.set_page_config(
    page_title="Next Watch",
    page_icon="🍿",  # 👋👋👋👋👋
)


user_limit = 1000000

# col1, col2 = st.columns(2)

if "user_id" not in st.session_state:
    st.session_state["user_id"] = -1

if "users_list" not in st.session_state:
    st.session_state.users_list = []


def print_session_user_id():
    print(st.session_state.user_id)


def register_user(users_list):
    print(st.session_state.user_id)
    if st.session_state.res_user_id not in users_list:
        st.session_state.user_id = st.session_state.res_user_id
        requests.post(
            "http://fastapi:8000/add_user/",
            params={"user_id": st.session_state.res_user_id},
        )
    else:
        st.session_state.user_id = -1
        st.markdown("#### User already exists!!!")


def save_rating(ratings_df, movie_row, ratings_df_path):
    matching_rows = ratings_df[
        (ratings_df["movieId"] == movie_row["movieId"].item())
        & (ratings_df["userId"] == st.session_state.user_id)
    ]
    new_row = {
        "userId": st.session_state.user_id,
        "movieId": movie_row["movieId"].item(),
        "rating": st.session_state.ratings,
        "timestamp": 122334,
    }
    # If there is a match, replace the existing row with the new values
    ratings_df.loc[
        matching_rows.index[0] if not matching_rows.empty else len(ratings_df)
    ] = new_row

    # Save new ratings...
    ratings_df.to_csv(ratings_df_path, index=False) ### Important !!!


if st.session_state.user_id == -1:

    # Get User ids in database through fastAPI
    response = requests.get("http://fastapi:8000/user_ids/")
    users_list = response.json()
    st.session_state.users_list = users_list

    st.markdown("# Next Watch: Login")
    existing_button = st.button("Use existing user id")
    register_button = st.button("Register new user id")

    if existing_button:
        st.markdown("### Select existing user id:")
        st.selectbox(
            "",
            users_list,
            key="user_id",
            on_change=print_session_user_id,
        )

    elif register_button:
        st.markdown("### Register new user id:")
        st.number_input(
            "",
            min_value=0,
            max_value=user_limit,
            value=0,
            step=1,
            format="%i",
            key="res_user_id",
            on_change=register_user,
            args=(users_list,),
        )

else:
    st.markdown(f"# Welcome back user {st.session_state.user_id}")

    st.markdown("### Rate a movie:")
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
    #    "/home/bruno/mlops-project/next-watch/data/01-external/movielens/movies.csv"
    ratings_df_path = paths.get_path(
        paths.DATA_01EXTERNAL,
        catalog.Sources.MOVIELENS,
        catalog.Datasets.RATINGS,
        suffix=catalog.FileFormat.CSV,
        storage=globals.Storage.DOCKER,
        as_string=True,
    )
    ratings_df = (
        pd.read_csv(ratings_df_path)
    )  # "/home/bruno/mlops-project/next-watch/data/01-external/movielens/ratings.csv")))
    movies_list = movies_df["title"].unique().tolist()
    selected_movie = st.selectbox("Select an existing user id:", movies_list)
    print(selected_movie)
    movie_row = movies_df[movies_df["title"] == selected_movie]
    print(movie_row)
    stars = st.slider(
        "",
        min_value=0.0,
        max_value=5.0,
        step=0.5,
        key="ratings",
        on_change=save_rating,
        args=(ratings_df, movie_row, ratings_df_path),
    )  # size=20,  size=20,  size=20,
    # print(stars)
    ##user_rating_info = {
    ##    "userId": st.session_state.user_id,
    ##    "movieId": movie_row["movieId"],
    ##    "rating": float(stars),
    ##    "timestamp": 1223,
    ##}
    # ratings_df.append()

    st.session_state.user_id = st.session_state.user_id

# with col2:

# if session_user_id < user_limit:
#    print(session_user_id)
#    st.markdown(f"# Recommending as user {session_user_id}")
#
