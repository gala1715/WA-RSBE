# WA-RSBE 代码与数据复现包

本仓库整理了当前 **Warp-Aware Road Slope and Bank Estimation (WA-RSBE)**
的 V5 代码快照、耦合坡原始数据、纯纵坡/横坡处理后指标以及论文复现实验脚本。

## 在线输入

WA-RSBE 在线仅使用：

- 六轴 IMU：`ax, ay, az, p, q, r`
- 四轮悬架位移：`FL, FR, RL, RR`

不使用纵向车速、轮速、轮胎力、附着系数或轮胎--地面模型作为在线递推输入。

## 快速运行

```bash
pip install -e .
python scripts/run_wa_rsbe.py \
  --zip data/example/o27r1_one_case.zip \
  --out outputs/example --max-plots 1
```

## 数据说明

`data/raw/coupled_o27r1/` 中包含 36 个横纵坡耦合工况的原始 NPZ 数据，
分别对应 2863、3108 和 3258 kg 三个质量组，每组 12 个工况。

当前工作区没有保留 O01/O02/O03/O09 的完整原始时序压缩包，因此仓库中对这些
工况只放入当前能够访问到的逐工况和汇总指标文件，没有伪造原始数据。完整说明见
`docs/DATASET.md`。

## 发布 GitHub 前

请确认实车数据的公开权限，并根据需要选择软件许可证。详见
`docs/GITHUB_PUBLISHING.md`。
