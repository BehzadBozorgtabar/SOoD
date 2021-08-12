from .base import BaseParams
from .params import Params
from argparse import Namespace

def load_params(model : str, dataset : str) -> Namespace:
    """Load parameters associated to a given model name

    Args:
        model (str): the name of the model
        dataset (str): the dataset name

    Returns:
        Namespace: The parameters to use in Namespace format
    """

    # Given the model, get the corresponding params to parse
    if model in ["CycleGAN", "SOoD", "TransAugSwav", "SOoDftClassifier", "SOoDftUns", "SimTriplet"]:
        params = Params(model, dataset)
    else:
        params = BaseParams(model, dataset)

    # We parse the arguments
    params = params.parse_set_default()

    return params


