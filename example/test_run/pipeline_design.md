# cLoops2 通用自动化流程设计

## 1. 现有 `run.sh` 的问题

`run.sh` 本质上是一个线性的 demo 脚本，适合教学，不适合通用自动化。主要问题有：

- 所有输入、输出、染色体范围、参数都被硬编码。
- 同一个脚本里混杂了数据生产、质量评估、特征识别、比较分析、可视化和导出。
- 产物直接散落在当前目录，不利于重复运行、断点恢复和多项目并存。
- 没有“阶段”概念，失败后只能手工判断从哪里重跑。
- 对单样本、重复样本、组间比较的逻辑没有抽象。

## 2. 推荐的通用流程模型

建议把流程拆成 8 类稳定阶段：

1. `qc`
   入口质控，面向原始 BEDPE。
2. `pre`
   预处理为 cLoops2 数据目录，支持单样本和重复样本合并。
3. `estimate`
   估计分辨率、作用距离、样本相似性。
4. `calling`
   调 peaks、loops、domains。
5. `vis`
   聚合图、局部图、过滤后绘图、domain bigWig 转换。
6. `compare`
   抽样、统一参数 call loops、差异 loops、montage。
7. `export`
   导出 bed/bedpe/bdg/hic/matrix。
8. `quant` / `annotation`
   跨样本定量与 loop 注释。

这 8 类阶段里，只有 1 到 4 是“核心生产阶段”，其余都应视为可选。

## 3. 配置驱动，而不是脚本驱动

推荐把“项目差异”全部放进配置文件，不放进流程代码。配置最少应该包括：

- 项目名和输出目录
- 样本到输入文件的映射
- 组到重复样本的映射
- 每个组的构建方式：`pre` 或 `combine`
- 主分析组、对照组
- 染色体范围
- 参考文件：`gtf`、`chrom.sizes`、`bw`、`bed`
- 每个阶段的参数串
- 每个阶段是否启用

当前提供的模板文件是 [pipeline.config.sh.example](/home/irenadler/cLoops2/example/test_run/pipeline.config.sh.example)。

## 4. 目录约定

建议所有产物写到统一 `OUTPUT_ROOT` 下：

- `datasets/`: `cLoops2 pre/combine/samplePETs` 生成的数据目录
- `reports/`: peaks、loops、domains、diff loops、quant、annotation 等文本结果
- `figures/`: agg、plot、montage、network 图等
- `exports/`: bed、bedpe、bdg、hic、matrix 等导出结果
- `.state/`: 阶段完成标记
- `logs/`: 每个阶段的日志

这样有三个好处：

- 可重跑，不污染工作目录
- 可并行维护多个项目
- 失败后可按阶段恢复

## 5. 执行器设计

执行器脚本是 [pipeline.sh](/home/irenadler/cLoops2/example/test_run/pipeline.sh)，设计原则如下：

- `set -Eeuo pipefail`，默认严格失败。
- 每个阶段单独落日志。
- 每个阶段完成后写 `.state/<stage>.done`。
- 支持 `all` 和单阶段运行。
- 先校验输入文件，再执行。
- 支持 hook，在不改主流程代码的前提下插入项目特定命令。

## 6. 这个模板如何映射当前 `run.sh`

当前样例已经覆盖了 `run.sh` 的主要逻辑：

- `qc` 对应第 1 步
- `pre` 对应第 2 步
- `estimate` 对应第 3 到 5 步
- `calling` 对应第 6、8、10 步
- `vis` 对应第 7、7.1、9、9.1、11、12 步
- `compare` 对应第 13、16 步
- `export` 对应第 14 步
- `quant` 对应第 15 步
- `annotation` 对应第 17 步

最后两条分析命令 `getSigDist.py` 和 `getSS.py` 没有被硬编码进主流程，而是建议放进 `extra` hook，因为这类命令通常项目特异性很强。

## 7. 参数覆盖方式

现在流程支持两层参数来源：

1. `pipeline.config.sh` 作为项目默认配置
2. 命令行参数作为单次运行覆盖

推荐原则：

- 稳定参数放配置文件
- 试验性参数放命令行
- 没有单独别名的变量，用 `--set VAR=VALUE`

常用覆盖示例：

```bash
./pipeline.sh --config pipeline.config.sh --target calling --threads 8
./pipeline.sh --config pipeline.config.sh --target pre --chrom chr1
./pipeline.sh --config pipeline.config.sh --target calling \
  --call-loops-args "-eps 100,200 -minPts 5 -w -j"
./pipeline.sh --config pipeline.config.sh --disable-stage compare
./pipeline.sh --config pipeline.config.sh --set BW_CTCF=../data/custom.bw
```

目前已经覆盖到你这条样例流程里实际会用到的主要命令参数入口：

- `qc / pre / combine`
- `estRes / estDis / estSim`
- `callPeaks / callLoops / callDomains`
- `filterPETs / samplePETs / callDiffLoops`
- `agg / plot / montage`
- `dump / quant / anaLoops`

这些覆盖项的设计参考了本地 [README.md](/home/irenadler/cLoops2/README.md) 和对应子命令 `-h`。

## 8. 推荐使用方式

1. 复制配置模板：

```bash
cp pipeline.config.sh.example pipeline.config.sh
```

2. 先只跑核心阶段：

```bash
./pipeline.sh --config pipeline.config.sh --target qc
./pipeline.sh --config pipeline.config.sh --target pre
./pipeline.sh --config pipeline.config.sh --target estimate
./pipeline.sh --config pipeline.config.sh --target calling
```

3. 核心结果确认后，再打开 `vis / compare / export / quant / annotation`。

## 9. 后续如果你要继续升级

如果你准备长期维护这个流程，下一步建议是：

- 把 bash 配置迁移到 `YAML` 或 `TOML`
- 用 `Python` 做参数校验和命令组装
- 引入 `Snakemake` 或 `Nextflow` 做真正的 DAG 调度
- 为每个阶段定义“输入产物存在性检查”，不要只靠 `.done` 标记

如果你希望，我下一步可以继续帮你做两件事中的一个：

1. 把这个 bash 版直接收敛成你当前项目可运行的正式版本
2. 进一步升级成 `Snakemake` 版工作流
