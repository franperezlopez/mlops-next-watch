import pandas as pd
import requests
import streamlit as st
from streamlit.runtime.state import session_state
from typing import List, Tuple

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


def save_rating(selected_movie):
    ## Save new ratings...
    response = requests.post(
        "http://fastapi:8000/add_rating/",
        params={
            "rating": st.session_state.ratings,
            "selected_movie": selected_movie,
            "user_id": st.session_state.user_id,
        },
    )

def get_recommendations() -> List[Tuple]:
    response = requests.get(
        "http://fastapi:8000/recommendations/",
        params={
            "user_id": st.session_state.user_id
        },
    )
    return response.json()


if st.session_state.user_id == -1:

    # Get User ids in database through fastAPI
    response = requests.get("http://fastapi:8000/user_ids/")
    users_list = response.json()
    st.session_state.users_list = users_list

    st.markdown("# Next Watch: Login")
    existing_button = st.button("Use existing user id")
    register_button = st.button("Register new user id")

    if existing_button:
        #  st.markdown("### Select existing user id:")
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
    ratings_df_path = paths.get_path(
        paths.DATA_01EXTERNAL,
        catalog.Sources.MOVIELENS,
        catalog.Datasets.RATINGS,
        suffix=catalog.FileFormat.CSV,
        storage=globals.Storage.DOCKER,
        as_string=True,
    )
    ratings_df = pd.read_csv(ratings_df_path)
    movies_list = requests.get("http://fastapi:8000/movies_list/").json()
    selected_movie = st.selectbox("Select an existing user id:", movies_list)
    stars = st.slider(
        "",
        min_value=0.0,
        max_value=5.0,
        step=0.5,
        key="ratings",
        on_change=save_rating,
        args=(selected_movie,),
    )

    recommendations = get_recommendations()

    recommendations_df = pd.DataFrame(recommendations, columns=["id", "userid", "movieid", "title", "Score Likelihood", "rank"])
    st.markdown("### Movie recommendations for you:")
    st.table(recommendations_df[["rank", "title", "Score Likelihood"]])

    st.session_state.user_id = st.session_state.user_id
