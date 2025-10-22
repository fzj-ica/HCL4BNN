import numpy as np

class Input:
    def __init__(self, x=None, y=None, func=None, n_samples=100, x_range=(0, 1)):
        """
        You can either:
        - Pass x and y manually
        - Or pass a function 'func' that generates y from x

        Example:
        >>> inp = Input(func=lambda x: 2*x + 1)
        """
        if x is None:
            self.x = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
        else:
            self.x = np.array(x).reshape(-1, 1)

        if y is None:
            if func is not None:
                self.y = func(self.x)
            else:
                raise ValueError("Either y or func must be provided.")
            
        else:
            self.y = np.array(y).reshape(-1, 1)

    def get_labels(self):
        return self.y
    
    def get_inputs(self):
        return self.x