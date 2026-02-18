# 🧰 Key Libraries & Tools — Safe, Robust & Explainable RL

> Every library used or referenced in Phase 5 + Phase 6 of the RL Roadmap, with install commands and usage context.

---

## 🧠 Core Deep RL Stack

### PyTorch
```bash
pip install torch torchvision
```
- **Use:** Q-networks, policy networks, autograd for FGSM attacks
- **Key features:** `nn.Module`, `optim.Adam`, `torch.autograd`
- 🔗 [pytorch.org](https://pytorch.org/)

### Gymnasium (OpenAI Gym successor)
```bash
pip install gymnasium
```
- **Use:** CartPole-v1 base environment, CMDP wrapper parent
- **Key features:** `env.reset()`, `env.step()`, Box/Discrete spaces
- 🔗 [gymnasium.farama.org](https://gymnasium.farama.org/)

### Safety Gymnasium
```bash
pip install safety-gymnasium
```
- **Use:** Benchmarking safe RL agents (Phase 6 extensions)
- **Key features:** SafetyGoal, SafetyButton, cost signals built-in
- 🔗 [safety-gymnasium.readthedocs.io](https://safety-gymnasium.readthedocs.io/)

### Stable Baselines 3
```bash
pip install stable-baselines3
```
- **Use:** PPO, SAC, TD3 reference implementations
- **Key features:** Easy eval callbacks, VecEnv wrappers
- 🔗 [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/)

---

## 🔍 Explainability & Interpretability

### SHAP
```bash
pip install shap
```
- **Use:** KernelSHAP for Q-network feature attribution, beeswarm plots
- **Key features:** `shap.KernelExplainer`, `shap.summary_plot`, `shap.force_plot`
- **Paper:** Lundberg & Lee, NeurIPS 2017
- 🔗 [shap.readthedocs.io](https://shap.readthedocs.io/)

### LIME
```bash
pip install lime
```
- **Use:** Local surrogate explanations — alternative to SHAP
- **Key features:** `lime.lime_tabular.LimeTabularExplainer`
- 🔗 [github.com/marcotcr/lime](https://github.com/marcotcr/lime)

### scikit-learn (Policy Distillation)
```bash
pip install scikit-learn
```
- **Use:** `DecisionTreeClassifier` for policy distillation + `export_text`
- **Key features:** `tree.export_text()`, `tree.export_graphviz()`
- 🔗 [scikit-learn.org](https://scikit-learn.org/)

### Captum (PyTorch XAI)
```bash
pip install captum
```
- **Use:** Integrated Gradients, Grad-CAM on PyTorch models
- **Key features:** `IntegratedGradients`, `LayerGradCam`, `Saliency`
- 🔗 [captum.ai](https://captum.ai/)

---

## ⚡ Robust RL / Adversarial

### Advertorch
```bash
pip install advertorch
```
- **Use:** Ready-made FGSM, PGD, CW attack implementations
- **Key features:** `GradientSignAttack` (FGSM), `LinfPGDAttack`
- 🔗 [github.com/BorealisAI/advertorch](https://github.com/BorealisAI/advertorch)

### ART (Adversarial Robustness Toolbox) — IBM
```bash
pip install adversarial-robustness-toolbox
```
- **Use:** Comprehensive adversarial attack & defense library
- **Key features:** Certified defenses, detector evasion, RL attacks
- 🔗 [adversarial-robustness-toolbox.readthedocs.io](https://adversarial-robustness-toolbox.readthedocs.io/)

### auto_LiRPA
```bash
pip install auto_LiRPA
```
- **Use:** Certifiable robustness via interval bound propagation (IBP/CROWN)
- **Key features:** Tight certified bounds for neural networks
- 🔗 [github.com/Verified-Intelligence/auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)

---

## 🛡️ Safe RL Frameworks

### SafePO-Baselines
```bash
pip install safepo
```
- **Use:** PPO-Lagrangian, CPO, PCPO implementations
- **Key features:** Drop-in replacements with cost tracking
- 🔗 [github.com/PKU-Alignment/Safe-Policy-Optimization](https://github.com/PKU-Alignment/Safe-Policy-Optimization)

### FSRL (Foundation Safe RL)
```bash
pip install fsrl
```
- **Use:** Safe RL benchmark — Lagrangian PPO, CVPO, FOCOPS
- 🔗 [github.com/liuzuxin/FSRL](https://github.com/liuzuxin/FSRL)

---

## 📊 Visualization & Analysis

### Matplotlib
```bash
pip install matplotlib
```
- **Use:** Learning curves, SHAP bar charts, robustness plots
- 🔗 [matplotlib.org](https://matplotlib.org/)

### Seaborn
```bash
pip install seaborn
```
- **Use:** Distribution plots, correlation heatmaps, styled charts
- 🔗 [seaborn.pydata.org](https://seaborn.pydata.org/)

### Plotly
```bash
pip install plotly
```
- **Use:** Interactive dashboards, 3D surface plots
- 🔗 [plotly.com/python](https://plotly.com/python/)

### Pandas
```bash
pip install pandas
```
- **Use:** Tabular results, episode logs, data management
- 🔗 [pandas.pydata.org](https://pandas.pydata.org/)

---

## 🔧 Development & Experiment Tracking

### tqdm
```bash
pip install tqdm
```
- **Use:** Training progress bars (`trange`, `tqdm`)

### Weights & Biases (wandb)
```bash
pip install wandb
```
- **Use:** Experiment tracking, hyperparameter sweeps, reward curve logging
- 🔗 [wandb.ai](https://wandb.ai/)

### MLflow
```bash
pip install mlflow
```
- **Use:** Local experiment tracking, model registry
- 🔗 [mlflow.org](https://mlflow.org/)

### Hydra (config management)
```bash
pip install hydra-core
```
- **Use:** Managing complex hyperparameter configs for RL experiments
- 🔗 [hydra.cc](https://hydra.cc/)

---

## 🚀 Complete Install (This Project)

```bash
# Core
pip install gymnasium torch numpy matplotlib seaborn

# Explainability
pip install shap scikit-learn captum

# Safe RL
pip install safety-gymnasium stable-baselines3

# Robustness
pip install advertorch

# Utilities
pip install pandas tqdm wandb
```

---

> 💡 **For GPU acceleration:** Install PyTorch with CUDA from [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
