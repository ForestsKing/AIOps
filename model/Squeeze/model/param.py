class Param:
    def __init__(self, **kwargs):
        self.max_normal_deviation = 0.5
        self.max_num_elements_single_cluster = 20
        self.ps_upper_bound = 1

        self.__dict__.update(kwargs)
