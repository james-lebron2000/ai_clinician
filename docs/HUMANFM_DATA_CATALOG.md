# HumanFM数据目录与规模

## 公开和受控资源

这些数据源可以支持单模态预训练或群体层知识，但除非资源本身提供可合法使用的同一参与者关联键，否则不能跨资源拼接成个体级多模态样本。

| 数据层 | 推荐资源 | 可支持任务 | 关键限制 |
|---|---|---|---|
| 多组织基因组与表达 | [GTEx](https://gtexportal.org/home/documentationPage) | 基因—组织表达、eQTL、正常组织参照 | V10约946名供体、54种组织、19,788个RNA-seq样本；原始序列和完整供体元数据为受控访问 |
| 单细胞多组学 | [Human Cell Atlas](https://data.humancellatlas.org/) | 细胞类型和状态编码器 | 多研究异质性；不能假设与其他队列个体配对 |
| 空间组学与组织图谱 | [HuBMAP](https://hubmapconsortium.org/hubmap-data/) | 细胞—组织空间对齐 | 需要保留供体、器官、组织块和数据集标识关系 |
| 组织/细胞蛋白表达 | [Human Protein Atlas](https://www.proteinatlas.org/about/download) | 蛋白定位、组织和病理弱监督 | 抗体、实验和版本差异需要显式建模 |
| 基因组＋EHR＋可穿戴 | [All of Us CDR v9](https://support.researchallofus.org/hc/en-us/articles/50653909888788-Our-Largest-Genomic-Dataset-Curated-Data-Repository-version-9) | 超过74.7万参与者；个体级纵向多模态锚点 | 必须在Researcher Workbench使用；[AI政策](https://support.researchallofus.org/hc/en-us/articles/34814131370388-Policy-Questions)还可能限制参与者级数据训练权重的下载和传播 |
| 基因组＋影像＋健康记录 | [UK Biobank](https://community.ukbiobank.ac.uk/hc/en-gb/articles/23472796568861-What-types-of-data-are-available-in-UK-Biobank) | 约50万人、全员WGS、约10万影像子队列及健康记录 | 受DUA和安全环境约束；[AI政策](https://community.ukbiobank.ac.uk/hc/en-gb/articles/21922841787293-Use-of-Artificial-Intelligence-AI-applications-and-models)也约束参与者数据进入生成式AI和衍生模型共享 |
| 住院与ICU时间序列 | [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/) | EHR和生理时间序列预训练 | 约364,627名患者，主要来自单一医疗系统；缺少广泛组织和组学配对 |
| 内部疾病模型 | CRO/医院前瞻队列 | 原组织—类器官—组学—药物干预真实配对 | 需要明确AI训练、商业用途、跨项目复用和申报使用同意 |

补充资源包括：[CELLxGENE Census](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_data_release_info.html)提供大规模单细胞语料，但需要用`is_primary_data`等字段去重；[TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)可在同一case内连接癌症多组学、临床和病理，但每例模态并不完整；[NCI Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/)可提供大规模癌症影像及部分TCGA/CPTAC关联；[Vivli](https://vivli.org/resources/requestdata/)提供经审查的临床试验个体数据，用于真实治疗—结局研究；[LINCS L1000](https://www.broadinstitute.org/publications/broad158356)提供约130万细胞系扰动表达profile，但它不是患者治疗结局。

All of Us明确将EHR、基因组、体格测量、调查和部分可穿戴数据结合在受控工作台中，并以OMOP标准化EHR；这是可借鉴的数据治理方式，而不是可以自由下载的训练语料。[All of Us方法](https://www.researchallofus.org/data-tools/methods/)

数据使用协议不仅约束原始数据，也可能约束embedding、检查点和模型权重。数据注册表因此单独记录`derived_model_policy`；如果计划导出模型，而任一训练数据源没有预先允许导出，审计会失败。受控基因组数据尤其不能默认认为“训练后权重已经脱敏”。

## 必须建设的内部数据

公开图谱主要提供相关性和健康参照。要预测干预后的真实人体状态，还需要内部或合作生成：

- 同一供体的原组织、病理、类器官/共培养、组学和功能读出。
- 药物、刺激、剂量、时间点、恢复和机制对照。
- 独立培养批次、实验室、仪器、试剂和SOP变更。
- 治疗前后纵向样本及明确的临床结局；只有同意覆盖时才关联。
- 失败实验、污染、失焦、无响应和适用域外样本。
- 随机或受控干预，以及可用于因果验证的负对照和正交读出。

## 工程规模目标

下表是建设通用化能力的工程目标，不是FDA规定的最低样本量。最终规模应由COU、任务难度、群体异质性和学习曲线决定。

| 数据域 | HumanFM-v0 | 长期通用化目标 |
|---|---:|---:|
| 多组织单细胞/空间组学 | 复用公开预训练模型＋内部100–500名配对供体 | 1,000万–1亿细胞，数千至上万供体，多器官/年龄/祖源 |
| 病理WSI | 1万–5万张，多中心 | 10万–50万张，5万名以上患者 |
| 放射影像 | 10万–50万检查 | 100万–500万检查，10万名以上患者 |
| 纵向EHR | 10万名以上、多年记录 | 50万名以上、10个以上中心 |
| 生理波形/可穿戴 | 5,000–10,000人 | 1万–10万人纵向记录 |
| 基因组/外显子 | 1万–10万人，优先多祖源 | 10万–50万人 |
| 三种以上模态的同人锚点 | 1,000–5,000人 | 1万–5万人 |
| 类器官/离体干预 | 现有100个模型用于COU适配 | 1,000名以上供体、10万级干预—条件组合 |

现有方案的60个肿瘤模型和40个免疫模型能够用于类器官领域适配、任务头和概念验证，但不能支撑跨器官、跨疾病、跨人群的通用人体Foundation Model。当前工作区也没有真实影像、组学或EHR训练资产；正式训练前必须先完成数据盘点和受控接入。

## 算力与存储分档

下面是基于已有基础模型训练报告给出的工程起点，不是固定硬件下限。真正瓶颈由token长度、原始图像解码、跨模态batch、互联带宽和墙钟目标共同决定。

| 路线 | 推荐起点 | 适用范围 |
|---|---:|---|
| 单模态探针、LoRA/Adapter | 1张48–80 GB GPU | 冻结公开编码器，训练任务头或小规模继续预训练 |
| HumanFM-v0融合MVP | 2–4张80 GB GPU | 冻结大部分单模态编码器，训练尺度latent、跨模态融合和小型世界模型 |
| 中型多模态继续预训练 | 8张80 GB GPU | 真实配对锚点训练、较大batch和多任务课程；建议NVLink/NVSwitch |
| 从零复刻单细胞、空间或病理FM | 12–64张A100/H100级GPU | 单个专业域的大规模预训练；同时需要高速对象存储、CPU解码和多节点网络 |
| 真正全模态人体FM | 多节点集群，按试验标定 | 不能仅按“几张卡”估算；先做缩放实验再锁定预算 |

公开训练给出了量级锚点：Nicheformer的约4930万参数空间模型在12张A100 40 GB上训练约10天；Prov-GigaPath训练使用64张A100 80 GB且大规模切片预处理需要额外CPU集群；EchoPrime通过复用预训练编码器和层级聚合，仅用2张50 GB A6000训练。这说明优先复用公开权重、冻结单模态编码器，往往比从零堆算力更经济。[Nicheformer](https://www.nature.com/articles/s41592-025-02814-z)、[Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w)、[EchoPrime](https://www.nature.com/articles/s41586-025-09850-x)

建议MVP配套至少准备100–300 TB可扩展对象存储、20–50 TB NVMe热缓存和40/100 GbE；进入百万级WSI、长视频或原始组学时，应按数据副本、预处理缓存和不可变快照重新估算，通常很快进入PB级。先在1%、5%、10%数据上测量GPU利用率、I/O等待、显存峰值和学习曲线，再采购完整集群。

## 数据覆盖矩阵

每个数据快照至少报告：

- 独立参与者、亲缘簇、标本、站点、仪器和时间跨度。
- 年龄、性别、祖源、地域、疾病、严重程度和治疗分布。
- 各模态及其真正配对数量，不只报告总记录数。
- 缺失模式、失败数据、质控拒绝和批次分布。
- 训练、调参、内部测试、前瞻测试和中心外测试的完全隔离情况。
- 同意、许可证、商业用途、跨境、保留期限和撤回处理。
- 数据及预处理SHA-256、格式、pipeline和ontology版本。

最大的数据缺口通常不是未配对图像数量，而是可合法使用、同一供体、跨尺度、带时间和干预的锚点数据。
