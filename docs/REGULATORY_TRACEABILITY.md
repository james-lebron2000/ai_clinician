# Legacy non-clinical CRO/FDA traceability (research_only)

> This document belongs exclusively to the retained `virtualhuman_agents` non-clinical R&D package. Its CRO, GLP, DDT, and submission-package logic is not used by `ai_clinician`. See `CLINICAL_REGULATORY_ROUTE.md` for the clinical research project.

本表将监管期望映射到已实现的软件控制，并明确仍需由CRO/申办方质量体系完成的工作。“已实现”只表示代码存在相应控制，不代表机构或部署实例已经通过验证或检查。

| 需求域 | 已实现的软件控制 | 主要证据对象 | 生产部署仍需完成 |
|---|---|---|---|
| Context of Use | 强制记录COU、Question of Interest、模型角色、错误后果和监管声明 | `ResearchGoal.regulatory`、就绪报告 | 与FDA审评部门确认COU和证据等级 |
| CRO研究治理 | 申办方/CRO/Study ID、受控Protocol/SAP、研究负责人、独立QAU、供应商资质 | `CROStudyGovernance` | SOW、主服务协议、人员培训与职责授权 |
| GLP职责分离 | 禁止研究负责人与QAU为同一用户；QAU审核分析和包释放 | `ElectronicApproval`、审计事件 | 21 CFR Part 58适用性判定、机构SOP、QAU检查计划和设施档案 |
| 21 CFR Part 11 | 对象哈希绑定签名、身份再认证事件引用、时间戳、追加审计哈希链 | `ElectronicApproval`、`AuditEvent` | 验证身份系统、签名声明、权限管理、记录复制/保留、IQ/OQ/PQ |
| ALCOA+数据完整性 | 原始数据URI和SHA-256、分析员/仪器/试剂批号、不可静默覆盖的结果 | `ResultBundle`、`ArtifactRef` | WORM存储、时钟同步、备份恢复和数据完整性调查SOP |
| 方案遵循 | 工单绑定Protocol/SOP/SAP；版本不匹配自动暂停 | `ExperimentPlan`、`WorkOrder`、`ResultBundle` | DMS受控生效、现场培训和方案偏差审批 |
| 偏差/CAPA | 关键、未关闭或使结果失效的偏差阻断分析 | `ProtocolDeviation` | 完整CAPA工作流、根因分析、有效性检查和趋势分析 |
| AI可信度 | COU和错误后果、模型卡/训练数据/软件验证受控文件、中心性能和OOD门 | `ControlledArtifact`、QC、就绪报告 | 冻结权重/代码/容器、独立验证集、漂移监控和生命周期计划 |
| 方法学验证 | CV、Z′、批间CV、ICC、保真度、变异一致性、影像Dice预设阈值 | `QualityThresholds`、`AnalysisReport` | 用真实样本确认准确度、精密度、范围、稳健性和缺失数据策略 |
| 独立复现 | 注册包要求至少两个站点且ICC达到阈值 | 站点分离的`ResultBundle` | 盲法样本分配、实验室资质和外部统计复核 |
| DDT资格认定 | 单独路径和DDT暂存目录；资格声明限定于COU | `RegulatoryPathway.DDT_QUALIFICATION` | LOI、Qualification Plan、Full Qualification Package及FDA互动 |
| 产品特定申报 | IND/NDA/BLA路径和Module 4内容映射 | `RegulatoryPackage` | 申办方决定申报位置和论证；已验证发布系统生成正式eCTD |
| FDA数据标准 | 记录SEND/自定义＋审评者指南的选择 | `study_data_standard`、reviewers guide | 每次提交前核对FDA Data Standards Catalog和技术符合性指南 |
| 最终QA释放 | 就绪闸门全通过后才能构建包；申办方与QAU双签才能释放 | 就绪报告、包SHA-256、审批记录 | QA对源数据、研究报告和发布序列执行独立审核 |

## 监管就绪硬闸门

系统只有在下列条件全部满足时才把包标为 `ready_for_qa`：非研究用途的监管路径、完整COU、项目完成、锁定的前瞻验证、全部受控文件、Protocol/SAP追溯、连续质量通过、双中心复现、原始数据校验值、偏差关闭、监管证据、角色化电子审批和审计哈希链有效。

`qa_released`还需要申办方代表与独立QAU对包本身的SHA-256签署。任何内容变化都会导致签名哈希不匹配。

## 监管声明模板

推荐使用：

> 本疾病模型在已声明的Context of Use、样本范围、方案版本和质量阈值内完成技术及生物学验证，可作为该产品研发计划的支持性非临床证据。其监管适用性仍由FDA结合完整申报资料审评确定。

禁止使用：

> 本平台符合FDA，因此模型已获FDA批准，可替代全部动物或临床证据。
