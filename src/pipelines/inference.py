import pandas as pd
from src.model.settings import MODEL_NAME
from src.services.mlflow import download_model


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    The function downloads a model from the model registry, uses it to make predictions on the input
    dataframe, and returns the dataframe with the predictions appended
    
    :param df: The dataframe to be predicted on
    :type df: pd.DataFrame
    :return: A dataframe with the predicted values
    """
    model = download_model(MODEL_NAME)

    y_hat = model.predict(df)
    return y_hat
