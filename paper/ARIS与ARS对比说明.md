# ARIS 与 ARS 对比说明

> 写于 2026-06-18。两者都已装好，可直接使用。

---

## 一、一句话定位

| | ARIS | ARS |
|---|---|---|
| **全名** | Auto-Research-In-Sleep | Academic Research Skills |
| **核心定位** | **科研全流程**自动化（含跑实验、分析数据、写论文） | **论文写作全流程**（调研→写→审→改，不含实验） |
| **理念** | "我替你全做，你去睡觉" | "AI 是副驾驶，你来主导" |
| **适合场景** | ML/AI 方向，有实验要跑的科研 | 各类学术论文写作，跨学科 |

---

## 二、文件在哪里

### ARIS

**仓库本体**（原始代码）：
```
C:\Users\a1382\Desktop\ARIS\
├── skills\          ← 80 个技能文件夹，每个里面是 SKILL.md（纯文字说明书）
├── mcp-servers\     ← 真正的 Python 代码（让 Claude 能"打电话给另一个模型"）
│   ├── gemini-review\server.py   ← 调 Gemini 审稿（当前账号被封，暂时用不了）
│   ├── manual-review\server.py   ← 手动审稿（弹窗，贴给 DeepSeek/Kimi）
│   ├── llm-chat\server.py
│   └── ...
└── tools\install_aris.ps1  ← 安装脚本
```

**装到的位置**（Claude Code 实际读的地方）：
```
C:\Users\a1382\Desktop\aris-research\.claude\skills\
└── （80 个 Junction 链接，指向 ARIS\skills\ 里的各个文件夹）
```

⚠️ **重要**：用 ARIS 技能必须在 `C:\Users\a1382\Desktop\aris-research\` 目录打开 Claude Code，否则找不到这些技能。

---

### ARS

**仓库本体**（原始代码）：
```
C:\Users\a1382\Desktop\ARS\
├── deep-research\          ← 文献调研技能
├── academic-paper\         ← 写论文技能
├── academic-paper-reviewer\ ← 模拟同行评审技能
├── academic-pipeline\      ← 全流程编排技能
├── agents\                 ← 各 Agent 的详细定义
├── shared\                 ← 共享引用文件（引用验证、防幻觉协议等）
├── scripts\                ← 引用验证缓存等辅助脚本
└── docs\SETUP.md           ← 安装说明
```

**装到的位置**（Claude Code 实际读的地方）：
```
C:\Users\a1382\.claude\skills\          ← 全局目录，任何项目都能用
├── deep-research           (Junction → ARS\deep-research)
├── academic-paper          (Junction → ARS\academic-paper)
├── academic-paper-reviewer (Junction → ARS\academic-paper-reviewer)
└── academic-pipeline       (Junction → ARS\academic-pipeline)
```

✅ **任何目录打开 Claude Code 都能用 ARS 技能**，不需要进特定文件夹。

---

## 三、分别作用于科研的哪个部分

```
完整科研流程：
[想 Idea] → [查文献] → [设计实验] → [跑实验/分析数据] → [写论文] → [审稿/修改] → [投稿]
    ↑              ↑          ↑              ↑                  ↑            ↑
   ARIS          ARIS        ARIS           ARIS              ARIS+ARS      ARIS+ARS
                 ARS                                           ARS           ARS
```

**ARIS 覆盖的部分**（80 个技能，几乎全流程）：

| 技能 | 阶段 |
|------|------|
| `/idea-discovery` | 从文献发现新 Idea |
| `/idea-creator` | 创造新研究想法 |
| `/novelty-check` | 查新颖性（这个 Idea 有没有人做过）|
| `/research-lit` | 文献综述 |
| `/experiment-plan` | 实验设计 |
| `/experiment-bridge` | 连接实验环境（⚠️ 要 GPU，谨慎用）|
| `/run-experiment` | 跑实验（⚠️ 要 GPU，谨慎用）|
| `/analyze-results` | 分析实验结果 |
| `/paper-writing` | 写论文（⚠️ 要 LaTeX，你还没装）|
| `/auto-review-loop` | 跨模型循环审稿 |
| `/rebuttal` | 写 Rebuttal（回复审稿意见）|
| `/research-pipeline` | 完整流程一键跑 |

**ARS 覆盖的部分**（4 个技能，专注论文写作）：

| 技能 | 阶段 |
|------|------|
| `deep-research` | 文献调研、系统综述、事实核查 |
| `academic-paper` | 写论文（12 种模式，输出 Markdown/DOCX）|
| `academic-paper-reviewer` | 模拟 5 位评审人同行评审 |
| `academic-pipeline` | 调研→写→审→改→定稿全流程（10 阶段）|

---

## 四、怎么调用——提示词怎么写

### ARIS 调用方式

**必须在 `aris-research` 目录打开 Claude Code。**

命令格式：
```
/技能名 "你的研究描述"  — reviewer: manual
```

- 引号里 = 用中文/英文描述你的问题或话题，随便写
- `— reviewer: manual` 固定写这个（**`—` 是长破折号**，直接复制这里的符号）
- 当前只能用 `manual`（Gemini 账号被封了），审稿时会弹出提示让你把内容贴给 DeepSeek/Kimi

**例子**：
```
/idea-discovery "细粒度多模态情感识别里，音频和视频信息经常被文本信息压制"  — reviewer: manual

/novelty-check "用辅助单模态推理缓解多模态情感识别中的模态不均衡"  — reviewer: manual

/research-lit "多模态情感识别 survey 2022-2025"  — reviewer: manual
```

**不需要 reviewer 的纯文本技能**（可以直接用，不加 `— reviewer`）：
```
/idea-discovery "你的话题"
/research-lit "你的话题"
/interview-cheatsheet "AI 多模态情感识别"
```

---

### ARS 调用方式

**任何目录打开 Claude Code 都可以用。**

**不需要写斜杠命令**，直接用中文说，Claude 自动识别用哪个技能：

| 你说的话 | 触发的技能 |
|---------|-----------|
| "帮我查一下多模态情感识别的文献" | `deep-research` |
| "帮我写一篇关于 EmoSync 的论文" | `academic-paper` |
| "帮我审一下这篇论文" | `academic-paper-reviewer` |
| "帮我完成完整的论文写作流程" | `academic-pipeline` |
| "我有个模糊的想法，帮我理清楚研究方向" | `deep-research`（苏格拉底模式）|

也可以指定模式：
```
帮我做文献综述，快速模式，主题是"多模态情感识别中的模态不均衡问题"

帮我写论文，计划模式，章节规划一下就行，先不全写

帮我审稿，全模式，论文如下：[粘贴论文内容]
```

---

## 五、什么时候用哪个

| 你想做的事 | 用哪个 |
|-----------|-------|
| 从零找研究 Idea | ARIS `/idea-discovery` |
| 查文献综述 | 都可以；ARS 更严谨（引用核验），ARIS 更快 |
| 设计实验方案 | ARIS `/experiment-plan` |
| 写论文 | 都可以；ARS 更适合正式学术论文（多种引用格式、防幻觉机制强）|
| 模拟同行评审 | ARS `academic-paper-reviewer`（5 位评审人）|
| 回复审稿意见(Rebuttal) | ARIS `/rebuttal` |
| 全自动跑完一篇论文 | ARIS `/research-pipeline` 或 ARS `academic-pipeline` |

---

## 六、当前已知限制

| 限制 | 影响 |
|------|------|
| ARIS 的 Gemini 审稿被封 | 只能用 `manual-review`（手动贴给 DeepSeek/Kimi） |
| LaTeX 未装 | ARIS 的 `/paper-writing`、`/paper-slides` 无法用（输出 PDF）；ARS 可以输出 Markdown/DOCX |
| ARIS 的实验相关技能 | `/experiment-bridge`、`/run-experiment` 要操作 GPU 服务器，需谨慎，别乱跑 |
| ARS 引用核验 | 需要 Semantic Scholar API（默认用公开接口，无需 key，已自动缓存）|
