import numpy as np

class portfolio:
    def __init__(self, securities, weights = None):
        self.securities = securities
        if self.weights is not None:
            self.weights = weights
        else:
            self.weights =
