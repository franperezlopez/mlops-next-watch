import pytest

from tests.models.utils import get_score


@pytest.mark.parametrize(
    "input_user_id, input_movie_id, ground_truth_score",
    [(11, 15, 4), (5, 15, 3), (5, 5, 3)],
)
def test_minimum_functionality(input_user_id, input_movie_id, ground_truth_score, predictor):
    score = get_score(input_user_id, input_movie_id, predictor)
    assert score >= ground_truth_score
