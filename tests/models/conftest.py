import pytest


def _very_advanced_als_model(user_id, movie_id):
    if user_id > 10:
        return 4.5
    if movie_id > 10:
        return 3.5
    return 3


@pytest.fixture(scope="module")
def predictor():
    """Here we fetch our model and return its instance,
    however, since we do not have connection to the model registry,
    or training/experiment runs, we are mocking its behavior...
    """
    return _very_advanced_als_model
