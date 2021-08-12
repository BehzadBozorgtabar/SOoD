"""...
"""

from .base import BaseParams

import argparse

class TemplateParams(BaseParams):
    """...
    """

    def __init__(self, model : str, dataset : str):
        """Initializes the params
        """
        super().__init__(model, dataset)

    def set_defaults(self):
        # Set some parameters to default values
        pass


    def check_for_consistency(self, args : argparse.Namespace) -> argparse.Namespace:
        # Correct conflicting arguments

        return args
