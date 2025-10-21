class Input:
    def __init__(self, n, bits):
        self.n = n
        self.bits = bits

    def get_labeled_data(self):
        raise NotImplementedError
    
    def get_rand_data(self):
        raise NotImplementedError