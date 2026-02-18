# Safe, Robust & Explainable Reinforcement Learning

### Phase 6 of the Reinforcement Learning Roadmap

This repository contains the slides, Jupyter notebook, and curated resources for **Phase 6** of my Reinforcement Learning learning roadmap, focusing on **trustworthy deep RL** — the engineering of agents that are safe under constraints, robust to adversarial perturbations, and transparent enough for regulatory audit.

📌 **Author:** Abdullah Zahid  
📅 **Date:** February 2026  
🔗 **LinkedIn:** https://www.linkedin.com/in/abdullahzahid655  
🐙 **GitHub:** https://github.com/abdullahzahid655

---

## 📘 Contents

### Slides
- **Phase 06 – Safety, Robustness & Explainability**  
  10-slide LinkedIn carousel covering:
  - Constrained Markov Decision Processes (CMDP) and Lagrangian dual optimisation
  - State-Adversarial MDPs (SA-MDP) and SA-DQN adversarial training
  - SHapley Additive exPlanations (SHAP) applied to Q-networks
  - Policy distillation into interpretable decision trees
  - Performance benchmarking dashboard across all three agents
  - Seminal literature and practitioner exercise sets

📄 Location: `slides/Phase_06_Safe_Robust_Explainable_RL.pptx`

### Notebook
- **Phase5_6_Safe_Robust_Explainable_RL.ipynb** — 26-cell end-to-end implementation:
  - Safe CartPole CMDP wrapper with cost signal and safety budget
  - Behaviour Cloning offline pretraining (Phase 5)
  - Baseline DQN fine-tuned from BC weights
  - Safe DQN with dual Q-heads and adaptive Lagrange multiplier λ
  - Robust DQN with FGSM adversarial training and consistency regularisation
  - KernelSHAP attribution on Q-network
  - Decision tree policy distillation with IF-THEN rule extraction
  - Comparison dashboard across all three agents

📄 Location: `Phase5_6_Safe_Robust_Explainable_RL.ipynb`

---

## 🧠 Topics Covered

### 1. Safe Reinforcement Learning — Constrained MDPs

The standard RL objective is extended with a cost signal and a safety budget:

```
max  𝔼π [ Σ γᵗ rₜ ]     subject to     𝔼π [ Σ γᵗ cₜ ] ≤ b
```

Solved via Lagrangian dual optimisation:

```
ℒ(π, λ) = Jʳ − λ(Jᶜ − b)
```

where λ ↑ when cumulative cost exceeds budget, and λ ↓ otherwise.

**Algorithm families covered:**
- Lagrangian Methods: PPO-Lagrangian, TRPO-Lagrangian, PID-Lagrangian
- Trust-Region Methods: CPO (Achiam et al., ICML 2017), PCPO
- Model-Based Safety: SafeDreamer (Huang et al., ICLR 2024)
- Hard Safety Guarantees: Control Barrier Functions (CBF)

**Industry deployments:**
- Autonomous Driving (Waymo / Tesla): collision probability < 10⁻⁶ per mile
- Power Grid Management: CVaR constraints on reliability
- Robotic Arm Manipulation: CMDP-based hazard avoidance (Robotics, MDPI 2024)
- Multi-Agent Drone Swarms: Scal-MAPPO-L (NeurIPS 2024)

---

### 2. Robust Reinforcement Learning — State-Adversarial MDPs

DRL policies are brittle: small observation perturbations collapse performance. The **State-Adversarial MDP (SA-MDP)** formalises this:

```
SA-MDP:  Ωˢ = (S, A, T, R, 𝒳, O^ξ)
```

The adversary modifies observations: O^ξ(xₜ | sₜ). The agent must perform well under **worst-case** perturbations.

**Adversarial attack taxonomy:**
- Observation Attacks: FGSM, PGD variants
- Action Attacks: NR-MDP framework
- Reward Attacks: Reward poisoning
- Adversarial Policies in MARL: Gleave et al., ICLR 2020

**SA-DQN (Zhang et al., NeurIPS 2020):**

```
FGSM:  x_adv = x + ε · sign(∇ₓ L(θ, x, a))

Robustness Loss:  L = L_TD(clean) + α · ‖Q(x) − Q(x_adv)‖
```

Hyperparameters: ε = 0.05, α = 0.5  
**RADIAL-RL** (Oikarinen et al., NeurIPS 2021) extends this with certified bounds via interval bound propagation (IBP).

---

### 3. Explainable RL — SHapley Additive exPlanations

XRL is the subfield that elucidates RL decision-making, enabling practitioners to understand *what* agents will do and *why* [Milani et al., ACM 2023].

**Shapley value attribution:**

```
φᵢ = Σ_{S ⊆ F\{i}}  |S|!(|F|−|S|−1)! / |F|!  ·  [v(S ∪ {i}) − v(S)]
```

φᵢ > 0: feature i increased action value · φᵢ < 0: feature i suppressed action value

**XRL Taxonomy (Milani et al., 2023):**

| Category | Methods |
|----------|---------|
| Feature Importance | SHAP Values · Saliency Maps · Gradient-based |
| Learning Process | Experience Attribution · Reward Attribution |
| Policy-Level | Decision Tree Distillation · NL Explanations |
| Counterfactual | Minimal state change to flip the decision |

**CartPole feature importance (SHAP):**

| Feature | Importance |
|---------|-----------|
| Pole Angle (θ) | 92% |
| Pole Angular Velocity (θ̇) | 74% |
| Cart Position (x) | 28% |
| Cart Velocity (ẋ) | 18% |

---

### 4. Policy Distillation — Interpretable Surrogate Models

```
Steps:
1. Train high-performing DNN policy πDNN
2. Generate dataset: (s, πDNN(s)) pairs — 300 episodes
3. Fit DecisionTreeClassifier (sklearn, max_depth=4)
4. Fidelity = P[ Tree(s) == DNN(s) ]   →  target: >90%
5. Extract rules: export_text(tree, feature_names=[...])
```

Sample extracted rule:
```
IF  θ̇ ≥ 0.021  AND  θ ≥ 0.004  →  PUSH RIGHT  (action = 1)
```

---

## 📊 Results

### Agent Benchmarking

| Agent | Method | Phase | Avg Reward ↑ | Violations ↓ | Noise Robust |
|-------|--------|-------|-------------|-------------|--------------|
| Baseline DQN | DQN + Behaviour Cloning | 5 | **420** | 32 ❌ | Low |
| Safe DQN | Lagrangian CMDP | 6 | 378 | **8 ✅** | Medium |
| Robust DQN | SA-DQN / FGSM | 6 | 352 | 27 ⚠️ | **High ✅** |

### Key Metrics

| Metric | Value | Context |
|--------|-------|---------|
| Constraint Violation Reduction | **75%** | Safe DQN vs. Baseline DQN |
| Reward Retention Under Noise σ=0.3 | **+40%** | Robust DQN vs. Baseline DQN |
| Decision Tree Fidelity | **>90%** | Tree surrogate vs. DNN policy |
| Lagrange Multiplier Peak Growth | **2.1×** | Over 500 training episodes |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  SafeCartPoleEnv (CMDP Wrapper)         │
│  cost = 1 if |θ| > 0.15 rad  ·  budget b = 20/episode  │
│  Gaussian noise injection (σ-adjustable)                │
└──────────────────────┬──────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
 ┌───────────┐   ┌───────────┐   ┌───────────┐
 │ DQNAgent  │   │ SafeDQN   │   │ RobustDQN │
 │ +BC pretrain│  │ 2 Q-heads │   │ FGSM+L_rob│
 │ (Phase 5) │   │ λ adaptive│   │ (SA-DQN)  │
 └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
       └───────────────┴───────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
 ┌───────────┐                  ┌──────────────┐
 │KernelSHAP │                  │Decision Tree │
 │ φᵢ per    │                  │Distillation  │
 │ state-step│                  │IF-THEN rules │
 └───────────┘                  └──────────────┘
```

**Training pipeline:**
Offline BC Pretraining → DQN Fine-Tuning → Safe / Robust Training → SHAP Attribution → Decision Tree Distillation

---

## 🚀 Quick Start

```bash
git clone https://github.com/abdullahzahid655/RL-Phase6-Safe-Robust-Explainable.git
cd RL-Phase6-Safe-Robust-Explainable
pip install gymnasium numpy torch matplotlib seaborn shap scikit-learn pandas tqdm
jupyter notebook Phase5_6_Safe_Robust_Explainable_RL.ipynb
```

---

## 📚 Key References

**Safe RL:**
- Achiam et al. (2017). *Constrained Policy Optimization.* ICML.
- García & Fernández (2015). *Comprehensive Survey on Safe RL.* JMLR 16(1).
- Huang et al. (2024). *SafeDreamer.* ICLR. arXiv:2307.07176
- Wachi et al. (2024). *Survey on Constraint Formulations.* arXiv:2402.02025
- Liu et al. (2024). *FISOR: Feasibility-guided Safe Offline RL.* ICLR.

**Robust RL:**
- Zhang et al. (2020). *SA-DQN.* NeurIPS Spotlight.
- Oikarinen et al. (2021). *RADIAL-RL.* NeurIPS.
- Gleave et al. (2020). *Adversarial Policies.* ICLR.

**Explainable RL:**
- Milani et al. (2023). *XRL Survey.* ACM Computing Surveys.
- Beechey et al. (2023). *SHAP for RL.* ICML.
- Bekkemoen (2024). *XRL Systematic Review.* Machine Learning 113.

**Offline RL (Phase 5):**
- Levine et al. (2020). *Offline RL Tutorial.* arXiv:2005.01643
- Kumar et al. (2020). *CQL.* NeurIPS.

See `resources/papers.md` for complete references with arXiv links.

---

## 🛠 Implementation Resources

- OmniSafe — Safe RL algorithm library
- Safety-Gymnasium — Unified safe RL benchmark
- SA-DQN codebase — github.com/chenhongge/SA_DQN
- RADIAL-RL — github.com/tuomaso/radial_rl_v2
- SHAP — `pip install shap`
- Captum — PyTorch native XAI
- PettingZoo — Multi-agent RL environments
- DSRL — Offline safe RL datasets

See `resources/libraries.md` for the complete tooling guide.

---

## 🗺 Roadmap Context

This is **Phase 6** of a larger RL roadmap:

- Phase 1–2: Fundamentals & Deep RL Architecture
- Phase 3–4: Applications & Mathematical Foundations
- Phase 5: Advanced Paradigms (MARL, HRL, Meta-RL, Offline RL)
- **Phase 6: Safety, Robustness & Explainability (this repo)**
- Phase 7 (upcoming): Real-World Deployment, RLHF, Foundation Models

---

## 🤝 Contributions & Feedback

This repository is shared for learning and discussion.  
Feedback, suggestions, and references are welcome.

If you find this useful, feel free to ⭐ the repository or share it.