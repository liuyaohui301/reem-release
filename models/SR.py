import numpy as np

from config.config import PROJECT_ROOT


class SymbolRegression():
    def __init__(self, root_path, model_path):
        super().__init__()
        self.model_path = PROJECT_ROOT / root_path / model_path
        self.exp = open(self.model_path, 'r').read().strip()
        self.func = self.create_function_from_expression()

    def create_function_from_expression(self):
        def func_call(x):
            last_step = x[:, -1, :]
            intemp = last_step[:, 0]
            outtemp = last_step[:, 1]
            power = last_step[:, 2]

            result = eval(self.exp, {
                'in1temp': intemp,
                'outtemp': outtemp,
                'Power': power,
            })
            return result.reshape(-1, 1) if hasattr(result, 'reshape') else np.array([[result]] * len(intemp))

        return func_call

    def predict(self, x):
        return self.func(x)
