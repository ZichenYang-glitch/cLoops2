# cLoops2 Snakemake Core

这是基于 `run.sh` 抽出来的 Snakemake 工作流，当前已经覆盖：

- `qc`
- `pre`
- `normalize`
- `estimate`
- `calling`
- `compare`
- `vis`
- `annotation`

它仍然是一个偏保守的工作流版本，但已经能把最常用的分析链路串起来。

## 文件

- [Snakefile](/home/irenadler/cLoops2/snakemake_core/Snakefile)
- [config.yaml.example](/home/irenadler/cLoops2/snakemake_core/config.yaml.example)

## 当前能力

- 对所有样本做 `cLoops2 pre`
- 对组数据按两种方式构建：
  - `pre`: 直接把多个 replicate 输入给 `cLoops2 pre`
  - `combine`: 先单样本 `pre`，再 `cLoops2 combine`
- 运行 `qc`
- 可选地对指定组运行 `samplePETs`
  - 自动读取各组 `petMeta.json` 的 `Unique PETs`
  - 以最小 `Unique PETs` 为基准，并向下取整到整百万作为统一深度
- 运行 `estRes / estDis / estSim`
- 对指定 `calling_groups` 运行：
  - `callPeaks`
  - `callLoops`
  - `callDomains`
- 对指定 compare 组运行：
  - 直接使用 `analysis` 阶段产出的数据目录
  - 统一参数 `callLoops`
  - `callDiffLoops`
  - 可选 `montage`
- 对主分析组运行：
  - `agg` for peaks / viewpoints / loops / domains
  - `filterPETs`
  - 可选 `plot` for domain / loop / filtered / arch
- 对主分析组 loops 运行：
  - `anaLoops`
  - 默认使用 `-gtf` 和 `-net`

## 配置方式

目录里已经给了一份可直接修改的 [config.yaml](/home/irenadler/cLoops2/snakemake_core/config.yaml)，模板保留在 [config.yaml.example](/home/irenadler/cLoops2/snakemake_core/config.yaml.example)。

如果你想从模板重建：

```bash
cd snakemake_core
cp config.yaml.example config.yaml
```

主要配置项：

- `project_name`: 项目标识
- `output_root`: 输出根目录
- `cloops2_cmd`: 真正执行 `cLoops2` 的命令，默认通过 `conda run -n cLoops2 cLoops2`
- `species`: 当前项目物种，如 `human` 或 `mouse`
- `chrom_whitelist`: 不同物种的 canonical chromosome 白名单
- `samples`: 样本名到 BEDPE 文件路径
- `groups`: 分组、replicate、组构建方式
- `primary_group`: 主分析组
- `calling_groups`: 哪些组执行 peaks/loops/domains
- `target_chroms`: 目标染色体
- `normalization.*`: 是否启用、哪些组参与、输出后缀
- `compare.*`: 是否启用 compare、哪些组参与、compare 输出后缀
- `compare.montage_enabled`: 是否把 `montage` 纳入默认 `all`
- `vis.*`: 是否启用可视化，`plot_enabled` 控制 `plot_*` 是否进入默认 `all`
- `annotation.enabled`: 是否把 `anaLoops` 纳入默认 `all`
- `references.*`: `gtf`、`bw`、`bed`、`chrom.sizes` 等参考文件
- `params.*`: 对应 cLoops2 子命令参数串

## 关于 plot 开关

`plot` 更偏探索式分析，不一定适合默认放进主工作流。所以现在拆成两层：

- `vis.enabled: true`
  控制 `agg_*`、`filterPETs` 这类适合自动化的步骤
- `vis.plot_enabled: false`
  控制 `plot_domain`、`plot_loop`、`plot_filtered`、`plot_arch` 是否进入默认 `all`

默认配置里 `plot_enabled` 是关闭的。你需要时手动打开：

```yaml
vis:
  enabled: true
  plot_enabled: true
```

这样 Snakemake 才会把 plot 相关目标也纳入默认执行。

如果你只是想手动出图，不建议为了这个改默认 `all`。现在可以直接单独触发：

```bash
snakemake --cores 4 plot_domain
snakemake --cores 4 plot_loop
snakemake --cores 4 plot_filtered
snakemake --cores 4 plot_arch
snakemake --cores 4 plot_all
```

这里的含义是：

- `plot_domain`: 画 domain 示例区域
- `plot_loop`: 画 loop 示例区域
- `plot_filtered`: 画过滤后的 loop 区域
- `plot_arch`: 画过滤后数据的 arch 视图
- `plot_all`: 一次性把上面四类固定 plot 都重建

所以 `plot_enabled` 只控制它们是否进入默认 `all`，不影响你手动按 target 运行。

## 关于 annotation

`annotation` 对应 `run.sh` 里的 `cLoops2 anaLoops`。

默认会对主分析组的 loops 运行：

```bash
cLoops2 anaLoops -loops <primary_group>_loops.txt -o <primary_group>_loops -gtf <references.gtf> -net
```

配置项是：

```yaml
annotation:
  enabled: true

params:
  annotation: "-net"
```

运行后，主结果文件是：

- `<primary_group>_loops_LoopsGtfAno.txt`

如果启用了 `-net`，还会额外得到：

- `_mergedAnchors.txt`
- `_mergedAnchors.bed`
- `_loop2anchors.txt`
- `_targets.txt`
- `_ep_net.sif`

你也可以单独手动触发：

```bash
snakemake --cores 4 annotation
snakemake --cores 4 annotate_loops
```

`plot` 的 region 现在也不再埋在 `params` 字符串里，而是单独放在 `plot_regions`：

```yaml
plot_regions:
  domain:
    chrom: "chr21"
    start: 35800000
    end: 36700000
    bs: 5000
  loop:
    chrom: "chr21"
    start: 38752604
    end: 38839334
    bs: 500
```

而非 region 的附加参数放在：

```yaml
params:
  plot_domain_extra: "-log -1D -corr"
  plot_loop_extra: "-triu -1D -m obs -log -vmax 1"
```

这样以后改 region 会更清楚，也更不容易把 `-start/-end/-bs` 和别的参数混在一起。

## 关于“去批次效应”

这里要说准确一点：当前工作流做的不是广义统计意义上的 batch correction，而是更接近“测序深度标准化”。

实现方式是：

- `cLoops2 pre` 产出每个组的 `petMeta.json`
- 从中读取 `Unique PETs`
- 对 `normalization.groups` 指定的所有组，取最小 `Unique PETs`
- 再把这个最小值向下取整到整百万，作为最终目标值
- 对这些组执行 `cLoops2 samplePETs -tot <rounded_min_unique_pets>`
- 下游 `estimate / calling` 默认使用标准化后的组目录

这比较适合你说的场景：不同批次或样本总深度不同，需要先统一到相同 PET 深度再继续分析。

## 关于“不要无法辨识的染色体”

这个需求在当前实现里放在 `pre` 阶段完成，也就是通过 `params.pre` 里的 `-c` 显式传入白名单染色体。

现在改成了 `species + chrom_whitelist` 模式。默认是：

```yaml
species: "human"
chrom_whitelist:
  human: "chr1,...,chr22,chrX,chrY"
  mouse: "chr1,...,chr19,chrX,chrY"
params:
  pre_extra: ""
```

`Snakefile` 会自动把它组装成 `cLoops2 pre -c ...`。

如果你要跑小鼠，只需要把：

```yaml
species: "mouse"
```

如果你只想跑特定染色体，也可以不改白名单表，而是在 `pre_extra` 里附加别的 `pre` 参数。

```yaml
params:
  pre_extra: "-mapq 10"
```

这样像 `chr22_KI270876v1` 这类不想保留的染色体就会在 `pre` 阶段直接被排掉。

## 运行

当前环境里这套工作流已经在 `snakemake` 环境里实际验证过 `qc`、核心 calling，以及轻量的 compare / vis 目标。

典型命令是：

```bash
cd snakemake_core
source /home/irenadler/miniconda3/etc/profile.d/conda.sh
conda activate snakemake
XDG_CACHE_HOME=/tmp/snakemake-cache SNAKEMAKE_OUTPUT_CACHE=/tmp/snakemake-cache snakemake -n
XDG_CACHE_HOME=/tmp/snakemake-cache SNAKEMAKE_OUTPUT_CACHE=/tmp/snakemake-cache snakemake --cores 4
```

如果你只想跑某个目标文件，比如 loops：

```bash
snakemake --cores 4 results_snakemake/cloops2_core/reports/gm_loops.txt
```

## 设计说明

这个版本依然保持了几个保守设计：

- 组装方式仍然兼容你现在 bash 版里的 `pre` / `combine` 两种逻辑
- 对 `estimate` 阶段先使用 marker 文件表示完成状态，避免过早绑定不稳定的输出文件名
- 深度标准化通过 `petMeta.json` 的 `Unique PETs` 自动确定目标值

如果你下一步继续推进，最自然的扩展顺序是：

1. `quant`
2. `annotation`
3. `export`
4. 将 rule 再拆分到 `rules/*.smk`
