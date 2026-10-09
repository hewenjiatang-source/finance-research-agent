<div align="center">

# 🚀 DeepResearch Agent

### *从复杂 Query 到结构化深度研究报告，全链路自动化*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Async](https://img.shields.io/badge/Async-asyncio-orange.svg)](https://docs.python.org/3/library/asyncio.html)

</div>

---

## 🏦 Finance Edition：财报 / 金融文档研究（本分支）

基于上游 [qiqihezh/deepresearch-agent](https://github.com/qiqihezh/deepresearch-agent)（MIT）改造：**LLM 换成 Claude**，场景换成**财报研究**，并新增**引用核对 + 数据准确性**自研评测。

```bash
cp .env.template .env        # 填 ANTHROPIC_API_KEY、SEC_USER_AGENT（"Your Name you@example.com"）、搜索 Key
python scripts/run_finance.py "对比苹果 FY2023 与 FY2022 的营收、净利润和稀释 EPS，并概括主要驱动因素"
# 产出 outputs/finance/report_*.md 以及 report_*.evidence.json（证据侧车文件）
```

**改了什么**

| 方面 | 做法 |
|---|---|
| Claude 后端 | `src/models/claude_policy.py`：OpenAI 风格消息 ↔ Claude Messages API（tool_use/tool_result 配对修复、thinking 块回传、重试、>16K 输出自动流式）。`configs/default.yaml` 默认 `backend: claude`，各模块可单独指定模型（compressor 用 haiku，judge 可用 opus）。 |
| 取数工具 | `src/tools/sec_edgar.py`：`sec_filings`（申报索引）、`sec_facts`（XBRL 结构化数值，含期间/表格/申报号溯源）、`sec_filing`（10-K/10-Q 章节与关键词窗口）。数字优先走结构化数据而不是让模型从 HTML 里"读"。 |
| 证据账本 | `src/finance/evidence.py`：每次工具调用即登记原文并返回稳定的 `evidence_id`；研究员 → 合成器 → Red/Blue → 评测全程共用同一套 `[n]`。 |
| Agent/Prompt | `src/finance/{agents,prompts}.py`：禁止凭记忆补数、数字须带期间与口径、派生数必须走 calculator、查不到写"未检索到"。 |
| 框架修复 | Red/Blue 原先只看报告前 4000 字，且 Blue 会用截断版覆盖全文；现可配置送审长度，并把修复结果与未送审的尾部拼回。 |

**评测（`evaluation/finance/`，`python scripts/run_finance_eval.py`）**

1. **引用核对**：悬空编号 / 被引来源不含该数字（misattributed、unsupported）/ 无引用数字是否有据；子句级归因。
2. **数据准确性**：数字归一（million/billion/亿）后按"写出的精度"与 XBRL 金标准比较，错误归类为 scale / period / metric / sign / imprecise / wrong_value；同比与毛利率等派生数由金标准复算。
3. **评测器自检（元评测）**：对金标准生成的干净报告注入 9 类已知错误，报告检出率、归类正确率与误报率；`--meta-only` 即可离线运行。
4. 可选 **LLM 判官**（`--judge`）：只审非数字断言的蕴含，带缓存，不并入硬指标。

用例：`scripts/build_finance_cases.py` 从 `watchlist.yaml` + SEC companyfacts 生成带金标准的 `real_cases.json`；`--run` 让 Agent 跑完并评测，或对已有报告目录离线评测。

**已验证 / 未验证（诚实说明）**

- 已验证（离线，`python -m unittest tests.test_claude_policy tests.test_finance_tools tests.test_finance_e2e tests.test_finance_eval`，72 项）：Claude 消息/工具转换、XBRL 选期（含 10-K/A、累计季度陷阱）、SEC 工具、证据账本、整条流水线（假模型 + 合成 SEC 夹具）、评测各层与元评测。
- **未验证**：真实 Anthropic API 调用、真实 EDGAR 联网、LLM 判官效果——开发环境没有 API Key 且无法访问 data.sec.gov。夹具里的 Acme Corp 是**虚构公司**。首次使用请先对 1–2 家公司人工抽查。
- 元评测的注入错误是人造的、句式规整，检出率是评测器的上界；逗号并列子句共享一个引用时，"漏引"无法判定（元评测里该项仅约 50%）。
- 数字→(指标, 期间) 映射是启发式，映射不了的计入"未映射"而非"正确"；MD&A 分部数据、非 GAAP、指引、日期尚不在自动核对范围；港股/A 股只能走 web_search，没有结构化金标准。

---

## 📖 项目背景

大语言模型在单一问答场景表现优异，但在**复杂深度研究任务**中面临三个核心挑战：

1. 🔥 **信息爆炸与上下文遗忘** —— 长文本检索后关键信息淹没在噪声中，模型难以聚焦
2. 👻 **幻觉与事实漂移** —— 多轮推理过程中，模型倾向于"编造"未经验证的事实
3. 📊 **缺乏系统性评估** —— 现有评测多以单轮 QA 为主，缺少对"深度研究报告"这一输出形态的端到端评价体系

本项目从零构建了一套**面向深度研究任务的 Agent 系统**，覆盖规划、执行、记忆、对抗、进化、评测全链路。

---

## 🎯 实验动机

> **"如果一个 Agent 只能回答简单问题，那它和搜索引擎有什么区别？"**

我们的动机是：**让 AI 真正具备"深度研究"的能力**——不只是检索信息，而是像人类研究员一样：
- 🧩 **拆解复杂问题** → 将模糊的研究目标分解为可执行的子任务
- 🔍 **多源信息整合** → 从网页、论文、数据库等多渠道收集证据
- ⚖️ **批判性审视** → 主动发现并修正报告中的错误和偏见
- 📝 **结构化输出** → 生成带引用、有逻辑、可验证的研究报告

---

## 💡 解决方法

### 六大模块协同工作

| 模块 | 职责 | 核心技术 |
|------|------|---------|
| 🎛️ **M1 Orchestrator** | 多智能体编排与调度 | 自研 asyncio + DAG 执行引擎，9 状态状态机 |
| 🗺️ **M2 Planner** | 复杂问题拆解 | JSON DAG 动态规划，支持执行中 replan |
| 🗜️ **M3 Compressor** | 长上下文压缩 | Embedding 语义三级过滤 + TextRank 关键句提取 |
| 🧠 **M4 Memory Store** | 跨 Agent 共享记忆 | SQLite + numpy 向量索引，去重/矛盾检测/LRU 淘汰 |
| ⚔️ **M5 Adversarial Loop** | 对抗降噪 | Red-Blue 循环攻击-修复，内置收敛与震荡检测 |
| 🧬 **M6 Evolution Engine** | 在线自进化 | GRPO 强化学习 + 符号规则学习（预留接口） |

### 数据流全景

```raw
用户 Query
    ↓
🗺️ Planner 拆解为 DAG 子任务图
    ↓
🎛️ Orchestrator 按拓扑排序并发调度
    ↓
🤖 Worker Agents 调用 🔍 搜索 / 📄 论文 / 🌐 网页 工具
    ↓
🧠 Memory Store 写入中间结果（去重 + 矛盾检测）
    ↓
🗜️ Compressor 压缩长上下文（L1→L2→L3）
    ↓
⚔️ Red Agent 攻击 → Blue Agent 修复 → 评分引擎评估
    ↓
📝 Summarizer 合成最终 Markdown 报告
    ↓
📤 输出带元信息的结构化研究报告
```

---

## ✨ 项目精彩之处

### 🏗️ 1. 自研编排引擎，不依赖 LangGraph/AutoGen

> 为什么不用现成的框架？因为深度研究任务需要**完全可控的调度逻辑**。

- 基于 `asyncio` + `Semaphore` 实现 **DAG 拓扑并发执行**
- **9 状态状态机**：IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING → ADVERSARIAL → DONE
- **三级降级策略**：单任务超时标记继续 → >50% 失败触发 replan → 全局超时强制合成

### ⚔️ 2. Red-Blue 对抗降噪 —— 主动抑制幻觉

> 灵感来自 GAN 的对抗训练思想，但应用于**文本质量优化**。

- **Red Agent** 从 5 个维度攻击报告：事实性、逻辑一致性、引用质量、覆盖面、时效性
- **Blue Agent** 执行 4 种修复操作：ADD / DELETE / MODIFY / VERIFY
- **收敛控制**：评分达标（≥8.0）/ 变化收敛（Δ < 0.3）/ 轮数上限（3 轮）三选一终止
- **震荡检测**：已修复问题重新出现 → 判定震荡 → 优雅终止

### 🗜️ 3. 语义级上下文压缩 —— 不是简单截断

> 关键词匹配会丢失语义，简单截断会丢失关键信息。**我们用 Embedding 做语义压缩**。

- **L1 粗过滤**：cosine similarity < 0.6 丢弃，> 0.95 完整保留
- **L2 细筛选**：TextRank + Query-Biased 提取关键句
- **L3 精保留**：高度相关内容保留原文，避免摘要失真

### 🧠 4. 跨 Agent 共享记忆 —— 会"反思"的系统

- 写前自动**去重**（cosine > 0.92）
- **矛盾检测**：启发式反义词 + 语义对立识别
- 三种**矛盾消解策略**：Majority Vote / Source Weight / LLM Judge
- **Session 隔离**：不同用户/会话的记忆物理隔离

### 🔌 5. 多后端 LLM 路由 —— 零源码切换模型

```yaml
# configs/default.yaml
model:
  backend_mapping:
    solver: "deepseek"      # 强推理
    planner: "deepseek"     # 结构化输出
    red_agent: "mimo"       # 稳定、低成本
    blue_agent: "mimo"
    judge: "mimo"
    compressor: "mimo"
```

- 支持 DeepSeek / MiMo 2.5 Pro / vLLM / OpenAI **热切换**
- 模块级采样参数集中管理，**避免配置漂移**
- `.env` 驱动，**零源码修改**接入新后端

### 📊 6. 完整的深度研究评测体系

> 不做"跑几个例子看看"的评测，做**可复现、可量化、有统计显著性**的评测。

| 评测层级 | 方法 | 特点 |
|---------|------|------|
| 📏 **规则指标** | 事实准确率 / 幻觉率 / 引用覆盖率 / 逻辑一致性 | 免费、可复现、零 API 成本 |
| 📚 **公共数据集** | HotpotQA 多跳 QA 深度研究变体 | 传统 EM/F1 + 新增语义覆盖度 |
| 🏗️ **自建评测集** | ResearchBench 35 题 × 11 领域 | 含 expected_topics + ground_truth |
| 👨‍⚖️ **LLM-as-Judge** | MiMo 5 维度 0-10 分深度评分 | 定性+定量互补 |
| 🥊 **Head-to-Head** | Agent vs 单轮 LLM 直接对比 | pairwise 更可靠 |
| 📈 **统计显著性** | Bootstrap 95% CI + Cohen's d + t-test | 拒绝"随机波动" |

---

## 🚀 快速开始

### 环境准备

```bash
# 1. 克隆项目
git clone https://github.com/qiqihezh/deepresearch-agent.git
cd deep_research_agent

# 2. 创建 uv 虚拟环境并激活
uv venv .venv
source .venv/bin/activate

# 3. 安装核心依赖
pip install -r requirements.txt

# 4. 配置 API Key（复制模板后填入）
cp .env.template .env
# 编辑 .env：填入 ANTHROPIC_API_KEY、BOCHA_API_KEY 等（DeepSeek/vLLM 仍可作为备选后端）
```

### 三种运行方式

**🎯 单条 Query（单次深度研究）**
```bash
python scripts/run_single.py \
    --query "2024-2025年大模型Agent技术趋势与落地案例研究" \
    --config configs/default.yaml
```

**💬 交互式 REPL（支持 Session 继承与连续追问）**
```bash
python scripts/run_repl.py
# 交互命令: ls / sessions / save / q
```

**🔬 批量实验（全量评测体系，overnight 可跑完）**
```bash
python scripts/run_all_experiments.py \
    --report_file outputs/reports1/report_xxx.md \
    --report_query "你的研究问题"
```

> 批量实验默认配置：模块消融 5×12 题 + 轮数消融 4×12 题 + 标准评测 35 题 + 领域对比 3×5 题 + Agent vs LLM 3 题 + Judge 1 次 = **165 次独立研究运行**

---

## 📁 仓库结构

```raw
deep_research_agent/
├── 📁 configs/                    # YAML 配置中心
│   ├── default.yaml               # 全局默认配置
│   ├── agents/                    # Agent 行为配置
│   ├── interaction_config/        # 交互层配置
│   └── tool_config/               # 工具层配置
│
├── 📁 src/                        # 核心源码（~5000 行）
│   ├── 📁 core/                   # 核心运行层
│   │   ├── runner.py              # 初始化模块 + 执行完整研究流程
│   │   ├── judge.py               # MiMo Judge 统一接口
│   │   └── ablation.py            # 消融实验通用框架
│   │
│   ├── 📁 orchestrator/           # 🎛️ M1: 多智能体编排器
│   ├── 📁 planner/                # 🗺️ M2: 自适应规划器
│   ├── 📁 compressor/             # 🗜️ M3: 上下文压缩器
│   ├── 📁 memory/                 # 🧠 M4: 共享记忆存储
│   ├── 📁 adversarial/            # ⚔️ M5: 对抗降噪循环
│   ├── 📁 evolution/              # 🧬 M6: 自进化引擎
│   ├── 📁 agents/                 # 🤖 Agent 实现
│   ├── 📁 models/                 # 🔌 模型路由层
│   ├── 📁 tools/                  # 🛠️ 工具层
│   └── 📁 utils/                  # 🧰 工具函数
│
├── 📁 evaluation/                 # 评测体系（~2000 行）
│   ├── benchmarks/                # 评测集（ResearchBench / HotpotQA）
│   ├── metrics/                   # 指标（规则 / Judge / 统计 / 综合）
│   └── analyze_ablation.py        # 消融实验结果分析
│
├── 📁 scripts/                    # 可执行脚本
│   ├── run_single.py              # 🎯 单条 query CLI
│   ├── run_repl.py                # 💬 交互式 REPL
│   ├── run_all_experiments.py     # 🔬 一键批量实验
│   ├── run_ablation.py            # 消融实验独立入口
│   ├── run_benchmark.py           # 🥊 Agent vs LLM
│   ├── run_eval.py                # 标准评测入口
│   ├── run_judge.py               # 👨‍⚖️ Judge 深度评分
│   └── validate_env.py            # 环境配置检查
│
├── 📁 verl/                       # veRL 训练框架（GRPO）
├── requirements.txt               # 依赖清单（分级安装）
└── README.md                      # 📖 本文件
```

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 🐍 语言 | Python 3.11 |
| ⚡ 异步框架 | asyncio |
| 🧠 LLM 后端 | DeepSeek API / MiMo 2.5 Pro / vLLM / OpenAI |
| 🔢 嵌入模型 | sentence-transformers (`all-MiniLM-L6-v2`) |
| 💾 持久化 | SQLite + numpy 向量索引 |
| 🎓 训练框架 | veRL (GRPO) |
| 🔭 可观测性 | LangSmith |
| 📦 虚拟环境 | uv |

---

## 🗺️ Roadmap

- [x] 自研编排引擎（asyncio + DAG）
- [x] Red-Blue 对抗降噪
- [x] 语义级上下文压缩
- [x] 跨 Agent 共享记忆
- [x] 多后端 LLM 路由
- [x] 完整评测体系（规则 + Judge + 统计显著性）
- [x] REPL 交互式会话
- [ ] 实验结果填充（进行中 🔥）
- [ ] Web UI（Gradio/Streamlit）
- [ ] 用户反馈闭环
- [ ] 多模态支持（图像/表格）

---

## 🤝 贡献

欢迎提交 Issue 和 PR！无论是 bug 修复、功能增强还是文档改进，我们都非常感谢。

---

## 📄 License

[MIT](LICENSE) © 2025 DeepResearch Agent Contributors
