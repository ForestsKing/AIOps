class Param:
    def __init__(self, **kwargs):
        self.max_normal_deviation = 0.05
        self.max_num_elements_single_cluster = 5
        self.ps_upper_bound = 0.90

        self.__dict__.update(kwargs)
