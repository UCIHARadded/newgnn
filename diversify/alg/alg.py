# -*- coding: utf-8 -*-
"""alg

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aMe0ajGPpw6z4-8tanfRynwTfC1ex36T
"""

# -*- coding: utf-8 -*-
"""alg

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1578K9_VeAYWiaG2hcrcKlCiPJheYh3kW
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from .algs.diversify import Diversify

ALGORITHMS = [
    'diversify'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in ALGORITHMS:
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return Diversify
