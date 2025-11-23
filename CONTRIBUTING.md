# Contributing to AI-Agents-for-Medical-Diagnostics

🎉 **Thank you for your interest in contributing!**  
This project explores how agentic AI systems can assist in multidisciplinary
medical diagnostics. Contributions of any size — from typo fixes to new agent
modules — are welcome.

---

## 🧠 Philosophy

This repository aims to stay **transparent, educational, and safe**.  
We use simulated and anonymized data only. No real medical records should ever
be uploaded or referenced.

---

## 🧩 How to Contribute

### 1. Fork and Clone
```bash
git clone https://github.com/<your-username>/AI-Agents-for-Medical-Diagnostics.git
cd AI-Agents-for-Medical-Diagnostics
```

### 2. Create a New Branch
Name it after what you’re working on:
```bash
git checkout -b add-cardiology-agent
```

### 3. Install Dependencies
Create a virtual environment and install requirements:
```bash
pip install -r requirements.txt
```

Add your `.env` file locally (never commit it!):
```
OPENAI_API_KEY=your_key_here
```

### 4. Make Your Changes
You can contribute in several ways:
- Add new **medical reports** under `/Medical Reports/`
- Improve or extend **agent logic** inside `/Utils/Agents.py`
- Add new **provider adapters** (e.g., `/Utils/open_router.py`)
- Refine **prompts**, output formats, or diagnostic explanations
- Improve **documentation**, comments, or examples

> 💡 **If you want to test a different model or agent logic**,  
> please **create a new file** under `/Utils/` (for example:  
> `Agents_GPT4.py` or `Agents_LocalLLM.py`) instead of modifying  
> the main `Agents.py`.  
> This keeps the repository stable and allows multiple model setups to coexist.

### 5. Test Before You Commit
Run:
```bash
python Main.py
```
and confirm the system executes without errors.

### 6. Commit Clearly
```bash
git add .
git commit -m "Add new medical report for COPD case"
```

### 7. Push and Open a Pull Request
Push your branch:
```bash
git push origin add-cardiology-agent
```
Then open a Pull Request (PR) on GitHub.

---

## 🧹 Code Style and Safety

- **No secrets or patient data** — keep `.env` and sensitive info out of commits.
- Keep PRs **atomic**: one feature or fix per PR.
- For report additions, use the naming format:  
  `Medical Report - [Name] - [Condition].txt`
- Follow Python’s [PEP8](https://peps.python.org/pep-0008/) guidelines.
- Include comments for any new class or function.

---

## 🧪 Optional: Add a Test Report
If you add a new medical report, include the generated diagnostic output under:
```
/Results/final_diagnosis_[PatientName].txt
```

---

## 🤝 Review and Merge Process

1. Each PR is reviewed for correctness, readability, and ethical compliance.  
2. Continuous integration (CI) runs lightweight tests on `Main.py`.  
3. Once approved, it will be squash-merged into `main`.

---

## ⚠️ Ethical Reminder

This repository is for **educational and research purposes** only.  
The generated results **must not** be used for real patient diagnosis or
medical decision-making.

---

## ❤️ Acknowledgments

Thank you for helping advance open, explainable AI in medicine.  
Your contributions make this project stronger and more meaningful for everyone.
