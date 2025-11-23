# 医疗诊断 AI 智能体（AI-Agents-for-Medical-Diagnostics）

<img width="900" alt="image" src="https://github.com/user-attachments/assets/b7c87bf6-dfff-42fe-b8d1-9be9e6c7ce86">

一个用于构建专门化 **LLM 医学智能体** 的 Python 项目，可对复杂病例进行协同分析。  
系统整合多名专科医生的洞察，输出全面的健康评估与干预建议，  
展示了多学科医疗场景中 AI 协作诊断的潜力。

⚠️ **免责声明**：本项目仅供研究与教学使用，  
**严禁**将其直接用于临床诊疗。

---

## ✨ 最新更新

- 新增 **MIT License**  
- 修复若干缺陷并更新 `requirements.txt` 依赖  
- 添加 `.gitignore` 配置  
- 核心 LLM 升级为 **GPT-5**  

---

## 🚀 工作原理

当前版本使用 **三名 GPT-5 智能体**，分别负责不同医学维度的分析。  
系统将同一份医疗报告同时传递给各个智能体，并以**多线程**方式并行运行。  
随后整合所有回复，总结出**三个可能的健康问题**及对应理由。

### 智能体角色

**1. 心脏科智能体（Cardiologist Agent）**  
- *关注点*: 捕捉心律失常、结构异常等心脏风险。  
- *建议*: 进一步心血管检测、监测方案与管理策略。  

**2. 心理科智能体（Psychologist Agent）**  
- *关注点*: 判断恐慌症、焦虑等心理状况。  
- *建议*: 心理治疗、压力管理或药物调整。  

**3. 肺科智能体（Pulmonologist Agent）**  
- *关注点*: 分析哮喘、呼吸障碍等呼吸系统原因。  
- *建议*: 肺功能检测、呼吸训练与相关治疗。  

---

## 📂 仓库结构

- `Medical Reports/` → 人工合成的医疗报告样本  
- `Results/` → 智能体生成的输出结果  

---

## ⚡ 快速开始

1. **克隆仓库：**
   ```bash
   git clone https://github.com/MZCnyhhd/Medical-Diagnostics
   cd Medical-Diagnostics
   ```
2. **创建虚拟环境并安装依赖：**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 请使用 venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **配置 API 凭证：**
   - 在项目根目录创建 apikey.env 文件。
   - 写入 OpenAI（或其他 LLM 服务）凭证：
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```
4. **运行系统：** `python main.py`

---

## 🔮 未来规划

后续版本预计将加入：

- **专家扩展**：新增神经科、内分泌科、免疫科等智能体。  
- **本地 LLM 支持**：通过 Ollama、vLLM、llama.cpp 等集成 **Llama 4**，并提供函数调用式 Hook 与安全执行。  
- **视觉能力**：让智能体分析**影像学数据**与其他医学图像，支持多模态决策。  
- **实时数据工具**：整合基于 LLM 的**实时搜索**与结构化医疗数据查询。  
- **高级解析**：通过 JSON Schema 等结构化输出，更好处理复杂报告。  
- **自动化测试**：加入评估流水线与模拟 LLM 调用的冒烟测试，确保可复现性。  

---

## 📜 许可证

本仓库采用 **MIT License** 授权。  

遵循 [LICENSE](LICENSE) 中的条款，你可以自由使用、复制、修改、合并、发布、分发、再授权及销售本软件。  

软件按 **“现状”** 提供，不附带任何明示或暗示的担保。
