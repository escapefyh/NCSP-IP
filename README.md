## 文件

| 文件 | 用途 |
| --- | --- |
| `ncsp_ip/features.py` | 原始名称解析、92 维注释、仅在 TRAIN 阳性 RNA 上拟合的字符特征 |
| `ncsp_ip/branches.py` | 名称与图传播分支 |
| `ncsp_ip/prior.py` | 先验教师训练、邻居供体、PAVA 投影和软补全 |
| `ncsp_ip/decoder.py` | 当前 λ=0 解码器训练与权重推理；保留局部消融参数接口 |
| `ncsp_ip/evaluation.py` | 原始重复划分、未知配对池和 AUC/AP/AUPR 指标函数 |
| `train.py` | 使用准备好的 TRAIN 输入训练先验及解码器，或复用冻结先验教师 |
| `predict.py` | 从本代码包生成的模型和上下文重新计算指定目录索引的分数 |
| `examples/make_synthetic_inputs.py` | 生成不含真实实体或关联的演示数据 |
| `tests/test_core.py` | 核心数值与权重往返验证 |
| `SOURCE_MANIFEST.json` | 提取函数的来源文件名及原文件 SHA-256，无本机目录 |

## 环境与快速运行

建议 Python 3.11。依赖版本来自原研究环境；PyTorch 的 CPU/CUDA 构建根据运行机器选择。

```bash
python -m pip install -r requirements.txt
python -B examples/make_synthetic_inputs.py
python -B train.py --input data/synthetic/inputs.npz --negative data/synthetic/negative_pairs.npz --output outputs/synthetic --device cpu
python -B predict.py --model-dir outputs/synthetic --rna-id 10 --disease-id 2
python -B -m unittest discover -s tests -v
```

从本目录运行。示例训练包含先验和解码器训练，仅用于验证程序连通性，不代表真实数据上的效果。输出目录必须为空，防止覆盖已有模型。
