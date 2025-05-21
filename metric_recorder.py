# tools/metric_recorder.py

import logging


class MetricRecorder:
    def __init__(self):
        self.metrics_list = []

    def add(self, metric_dict):
        """
        添加一组新的指标到记录器中。
        参数：
            metric_dict (dict): 例如 {'Loss': 0.235, 'Acc': 0.92, 'F1': 0.88}
        """
        self.metrics_list.append(metric_dict)

    def get(self):
        """
        返回所有已记录指标的平均值。
        返回：
            dict: 各个指标的平均值
        """
        if not self.metrics_list:
            return {}

        total = {}
        count = len(self.metrics_list)

        # 初始化总和
        for key in self.metrics_list[0]:
            total[key] = 0.0

        # 累加所有指标
        for entry in self.metrics_list:
            for key, value in entry.items():
                total[key] += value

        # 求平均
        avg_metrics = {key: total[key] / count for key in total}
        return avg_metrics

    def display(self, prefix=''):
        """
        显示当前记录的平均指标（打印格式化字符串）
        参数：
            prefix (str): 前缀标题，比如 epoch 编号
        """
        avg_metrics = self.get()
        metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in avg_metrics.items()])
        full_msg = f'{prefix}{metric_str}'
        print(full_msg)
        logging.info(full_msg)

    def reset(self):
        """
        清空当前记录器中所有指标
        """
        self.metrics_list = []
