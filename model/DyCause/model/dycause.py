import warnings
from collections import defaultdict

import numpy as np

from model.DyCause.model.Granger_all_code import loop_granger
from model.DyCause.model.causal_graph_build import get_segment_split, get_ordered_intervals, get_overlay_count, \
    normalize_by_column
from model.DyCause.model.ranknode import analyze_root

warnings.filterwarnings('ignore')


class DyCause:
    def __init__(self,
                 feature,
                 before_length=15,
                 after_length=25,
                 step=10,
                 significant_thres=0.05,
                 auto_threshold_ratio=0.5,
                 entry_point=14,
                 mean_method="harmonic",
                 max_path_length=None,
                 topk_path=15,
                 num_sel_node=3,
                 ):
        self.entry_point = entry_point
        self.local_data = None
        self.local_length = None
        self.before_length = before_length
        self.after_length = after_length
        self.step = step
        self.feature = feature
        self.significant_thres = significant_thres
        self.auto_threshold_ratio = auto_threshold_ratio
        self.mean_method = mean_method
        self.topk_path = topk_path
        self.num_sel_node = num_sel_node
        self.max_path_length = max_path_length

    def _granger_process(self, x_i, y_i):
        ret = loop_granger(
            self.local_data,
            path_to_output=None,
            array_data_head=self.feature,
            feature=self.feature[x_i],
            target=self.feature[y_i],
            significant_thres=self.significant_thres,
            test_mode="fast_version_3",
            trip=-1,
            lag=2,
            step=self.step,
            simu_real="simu",
            max_segment_len=self.local_length,
            min_segment_len=self.step,
            verbose=False,
            return_result=True,
        )
        return ret

    def location(self, data, error_time):
        # Select abnormal data
        self.local_data = data[max(0, error_time - self.before_length):
                               min(len(data) - 1, error_time + self.after_length), :]
        self.local_length = len(self.local_data)
        list_segment_split = get_segment_split(self.local_length, self.step)

        # get local results
        local_results = defaultdict(dict)
        for x_i in range(len(self.feature)):
            for y_i in range(len(self.feature)):
                if x_i == y_i:
                    continue
                total_time, time_granger, time_adf, array_results_YX, array_results_XY = self._granger_process(x_i, y_i)
                if array_results_YX is None and array_results_XY is None:
                    ordered_intervals = []
                else:
                    matrics = [array_results_YX, array_results_XY]
                    ordered_intervals = get_ordered_intervals(
                        matrics, self.significant_thres, list_segment_split
                    )
                local_results["%s->%s" % (x_i, y_i)]["intervals"] = ordered_intervals
                local_results["%s->%s" % (x_i, y_i)]["result_YX"] = array_results_YX
                local_results["%s->%s" % (x_i, y_i)]["result_XY"] = array_results_XY

        # region Construction impact graph using generated intervals
        # Generate dynamic causal curve between two services by overlaying intervals
        histogram_sum = defaultdict(int)
        for x_i in range(len(self.feature)):
            for y_i in range(len(self.feature)):
                if y_i == x_i:
                    continue
                key = "{0}->{1}".format(x_i, y_i)
                intervals = local_results[key]["intervals"]
                overlay_counts = get_overlay_count(self.local_length, intervals)
                histogram_sum[key] = sum(overlay_counts)

        # Make edges from 1 node using comparison and auto-threshold
        edge = []
        edge_weight = dict()
        for x_i in range(len(self.feature)):
            bar_data = []
            for y_i in range(len(self.feature)):
                key = "{0}->{1}".format(x_i, y_i)
                bar_data.append(histogram_sum[key])

            bar_data_thres = np.max(bar_data) * self.auto_threshold_ratio
            for y_i in range(len(self.feature)):
                if bar_data[y_i] >= bar_data_thres:
                    edge.append((x_i, y_i))
                    edge_weight[(x_i, y_i)] = bar_data[y_i]

        # Make the transition matrix with edge weight estimation
        transition_matrix = np.zeros([data.shape[1], data.shape[1]])
        for key, val in edge_weight.items():
            x, y = key
            transition_matrix[x, y] = val
        transition_matrix = normalize_by_column(transition_matrix)

        # region backtrace root cause analysis
        ranked_nodes, new_matrix = analyze_root(
            transition_matrix,
            self.entry_point,
            self.local_data,
            mean_method=self.mean_method,
            max_path_length=self.max_path_length,
            topk_path=self.topk_path,
            prob_thres=0.2,
            num_sel_node=self.num_sel_node,
            use_new_matrix=False,
            verbose=False,
        )
        print(ranked_nodes)
        return ranked_nodes
