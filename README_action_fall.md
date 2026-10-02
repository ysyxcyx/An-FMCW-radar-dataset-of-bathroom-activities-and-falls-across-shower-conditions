# RD/RA 动作分类与跌倒识别示例

脚本 `example_action_fall_classifier.py` 使用 `F:\FALL_RD_RA_RELEASE\data_v2\TOP_SPRAY` 的处理后 RD、RA、label.csv 和 session 元数据，完成两个 NumPy 基线任务：

- 14 类动作分类；
- 二分类跌倒识别，其中 `Fall down` 为正类，`Fall down and get up` 作为跌倒后的起身动作，归入二分类的 `Non-fall`。

脚本以标注区间中心为样本，每个样本读取中心附近16帧，分别对 RD 幅度和 RA 功率取 `log1p`，进行固定网格平均池化，再使用一层隐藏层的 MLP。训练/测试按照 session 划分，避免同一个 session 的相邻时间区间同时进入训练和测试。

运行：

```powershell
py -3.13 examples/example_action_fall_classifier.py `
  --data-root F:\FALL_RD_RA_RELEASE\data_v2 `
  --group TOP_SPRAY `
  --out-dir examples/action_fall_results
```

已运行的结果在 `action_fall_results` 中。混淆矩阵为：

- `action_classification_confusion_matrix.png/.csv`
- `fall_detection_confusion_matrix.png/.csv`

同时保存了指标 JSON、预测 CSV、标准化参数、模型参数和本次 session 划分。该示例用于展示数据读取、标签映射和评估流程，结果不是面向论文的最终模型性能。
