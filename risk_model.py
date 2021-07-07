import numpy as np

from portfolio import Engine

class RiskModel(Engine):
    def __init__(self, securities: list):
        #factors in the risk_model
        super().__init__(securities)

