# NCSP-IP

tRNA–疾病关联预测算法的代码发布包，整理自当前采用的 **λ=0（去残差幅度平方惩罚）版本**。

代码包含名称字符 TF-IDF、92 维名称注释、名称与图传播双分支、顺序约束先验融合、冷 RNA 邻域软补全，以及三种子独立 sigmoid 解码器。保留参数正则 `1e-4`，残差上限为 `2`，温度为 `5`；λ=0 不代表移除所有正则项。

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
| `PRIVACY.md` | 发布范围和隐私处理说明 |

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

## 自备数据接口

本发布包不附带原始数据、实体表、实验分数或训练权重。请在有权使用的数据上自行准备 `inputs.npz`：

| 数组 | 形状及含义 |
| --- | --- |
| `A` | `(n, m)`，float32 的 TRAIN 阳性二值矩阵，不能包含验证/测试阳性边 |
| `x` | `(n, feature_width)`，float32 的名称 TF-IDF；调用 `features.vectorize(names, A)` 生成 |
| `ann` | `(n, 92)`，float32 名称注释；按目录顺序调用 `features.annotation(name)` 生成 |
| `P` | 可选 `(n, n)` TRAIN 名称邻居权重；缺省按原 top-3 规则计算 |
| `prior` | 复用教师时必需的 `(n, m)` 冻结先验；须与该折输入及教师一一对应 |

`negative_pairs.npz` 包含整数数组 `pairs`，形状 `(TRAIN阳性数, 2)`，列为 RNA 索引和疾病索引。当前采用模型的正负比为 1:1，每个 RNA 的代理负例数等于其 TRAIN 阳性数。输入验证会检查形状、边界、重复及 TRAIN 阳性冲突。

准备负例时还必须排除全目录已知阳性和同折验证/测试未知池；这些排除集合不在输入矩阵中，需由数据准备过程保证。代理负例不是实验确认的阴性。节点名称、疾病名称、特征、图、先验和教师状态必须保持相同索引顺序。

默认从输入 TRAIN 图重新训练先验教师。复用原教师时：

```bash
python -B train.py --input data/private/inputs_with_prior.npz --negative data/private/negative_pairs.npz --teachers data/private/teachers.pt --seed 2026111930 --output outputs/private_run --device cpu
```

教师 checkpoint 的键为 `101/off`、`101/real` 等，覆盖 offset `101, 202, 303`；每个值含原分支 `state` 和注释投影 `W`。`prior` 对应原先验 `scores.npz` 的 `soft`，不是最后的独立 sigmoid 分数。命令行默认 seed 对应原第 1 组；其他划分需要传入其实际 seed。

新 CLI 使用 PyTorch `weights_only=True` 读取张量 checkpoint。某些旧归档 checkpoint 含 NumPy 诊断对象，可能无法直接读取；可在可信的本地环境提取并另存仅含 `state` 和 `W` 的教师字典。不要为了加载未知来源的权重关闭安全加载。

## 复现范围与解释

算法函数由正式训练代码及其实际依赖提取，移除了文件系统归档操作和本机路径。新入口负责输入与输出，不复制历史文献检索、报告生成和实验调度。来源清单中的哈希对应提取前的原文件，而不是发布文件。

`evaluation.split_indices` 保留原来 5 次重复、每次 5 组的划分规则：约 80% TRAIN、10% VALIDATION、10% TEST；不是每折 20% TEST 的标准五折评价。`unknown_pairs` 的 `repeat` 参数和 `split_indices` 一样使用 0–4。本包未封装原 25 组实验、全套消融和外部算法比较的一键调度。

本地核验已用原第 1 组正式 checkpoint 比较重建预测，并在真实 TRAIN 上对原始与提取后的单种子解码器执行相同训练。合成示例也已完成先验训练、解码器训练和 checkpoint 推理。由于原始数据、负例名单、划分和权重未随包分发，下载代码本身不足以重现论文数值。不同设备与软件版本也可能影响优化结果。

输出是已知目录内配对的排序分数，不是经过临床校准的真实关联概率。本入口不宣称支持目录外 RNA 或疾病。名称注释是规则解析提示，不会自动更正生物学实体名称。原数据曾被历史实验反复探索，不能将这些随机划分称为独立外部验证。

## 发布说明

只上传本文件夹内的文件。数据和输出目录已被 `.gitignore` 排除，但公开前仍需检查实际暂存文件。本包未代替权利人选择开源许可证；如需授予第三方明确的使用、修改和分发权，请由权利人决定并添加 `LICENSE`。
