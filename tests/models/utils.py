def get_score(input_user_id, input_movie_id, predictor):
    return predictor(input_user_id, input_movie_id)
