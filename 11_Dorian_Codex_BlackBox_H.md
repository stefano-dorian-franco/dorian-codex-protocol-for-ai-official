# 11# IMPLEMENTATION OF THE DORIAN CODEX BLACKBOX-H
## A Hamiltonian Framework for Real-World LLMs Without Internal Access

The **Dorian Codex BlackBox-H** represents the second operational layer of the Protocol. While the Clockwork module (Ch. 10) provides a high-fidelity research implementation, it requires visibility into the model's internals. 

In 2025, most state-of-the-art LLMs operate as **Black Boxes**. This module answers a fundamental need: how to apply Hamiltonian stability ($H_{SAFE}$) to models whose logits, attentions, and cognitive states are inaccessible.

---

## 11.1 — Purpose and Rationale

BlackBox-H translates the formal Hamiltonian expression into a fully external, API-compatible monitoring device:

$$H_{SAFE} = T + V - Z + \lambda_U U + \beta_R R$$

### Core Commitments:
1.  **Universality:** Operates on any model (OpenAI, Anthropic, Google, Mistral, etc.).
2.  **Deterministic Fallback:** Uses a cryptographic hash embedding (SHA-256) when semantic embeddings are unavailable to ensure reproducibility.
3.  **Proxy-based Inference:** Approximates $Z$ (Entropy) through geometric and statistical measures of the text output.
4.  **Ethical Modulation:** Maintains safety alignment through $U$ (Novelty) and $R$ (Ethical Reward) terms.

---

## 11.2 — Stability Indicators

The module computes five stability indicators based exclusively on observable outputs:
* **$T$ (Semantic Velocity):** Abruptness of shifts between consecutive responses.
* **$V$ (Alignment):** Cosine similarity between the response and the interaction goal.
* **$Z$ (Composite Entropy Proxy):** Aggregate of semantic diversity and inter-model divergence.
* **$U$ (Novelty):** Deviation from historical mean to prevent stagnation.
* **$R$ (Ethical Reward):** Estimation of beneficial vs. harmful patterns.

---

## 11.3 — Operational Necessity

Where Clockwork is **theoretical fidelity**, BlackBox-H is **operational necessity**. It allows the Codex to be deployed in production-scale monitoring and pre-AGI safety layers for API-only models.

---

## III. PYTHON IMPLEMENTATION: DORIAN CODEX BLACKBOX

This implementation provides a minimalist and justified framework for $H_{SAFE}$ calculation without internal access.

```python
import numpy as np
import hashlib

class DorianCodexBlackBox:
    """
    DORIAN CODEX BLACKBOX-H
    Minimalist implementation of H_SAFE for "Black-Box" LLMs.
    Proxies used for Z: H_text', H_state', H_div'.
    """

    def __init__(
        self,
        alpha=0.33,      # H_text weight
        beta=0.33,       # H_state weight
        gamma=0.34,      # H_div weight
        lambda_U=0.30,   # Novelty weight
        beta_R=0.50,     # Ethical reward weight
        dim=128,
        max_history=50,
        embedder=None
    ):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.lambda_U, self.beta_R = lambda_U, beta_R
        self.dim = dim
        self.max_history = max_history
        self.embedder = embedder or self._default_embedder
        self.prev_E = None
        self.history = []

    def _default_embedder(self, text: str) -> np.ndarray:
        """Deterministic fallback: SHA-256 -> Normalized Vector."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        if len(v) < self.dim:
            reps = int(np.ceil(self.dim / len(v)))
            v = np.tile(v, reps)
        v = v[:self.dim]
        return v / (np.linalg.norm(v) + 1e-8)

    def embed(self, text: str) -> np.ndarray:
        return self.embedder(text)

    # --- Hamiltonian Terms ---
    def T(self, E: np.ndarray) -> float:
        """Semantic Velocity."""
        if self.prev_E is None: return 0.0
        return np.linalg.norm(E - self.prev_E) / np.sqrt(len(E))

    def V(self, E: np.ndarray, G: np.ndarray) -> float:
        """Alignment via Cosine Similarity."""
        denom = (np.linalg.norm(E) * np.linalg.norm(G) + 1e-8)
        return float(np.dot(E, G) / denom)

    # --- Black-Box Z Proxies ---
    def H_text(self, samples) -> float:
        """Proxy for Entropy via Semantic Diversity."""
        if not samples or len(samples) <= 1: return 0.0
        embeds = [self.embed(s) for s in samples]
        dists = [np.linalg.norm(embeds[i] - embeds[j]) 
                 for i in range(len(embeds)) for j in range(i+1, len(embeds))]
        return float(np.mean(dists))

    def H_state(self, E: np.ndarray, E_hat: np.ndarray) -> float:
        """State Error Proxy."""
        return float(np.linalg.norm(E - E_hat))

    def H_div(self, E_models) -> float:
        """Inter-model divergence proxy."""
        if not E_models or len(E_models) <= 1: return 0.0
        dists = [np.linalg.norm(E_models[i] - E_models[j]) 
                 for i in range(len(E_models)) for j in range(i+1, len(E_models))]
        return float(np.mean(dists))

    def Z(self, h_text: float, h_state: float, h_div: float) -> float:
        return (self.alpha * h_text + self.beta * h_state + self.gamma * h_div)

    # --- Ethical & Novelty Terms ---
    def U(self, E: np.ndarray) -> float:
        """Novelty relative to history."""
        if len(self.history) < 3: return 0.1
        H = np.stack(self.history, axis=0)
        E_mean = np.mean(H, axis=0)
        return float(np.linalg.norm(E - E_mean) / (np.sqrt(len(E)) + 1e-8))

    def R(self, text: str) -> float:
        """Simplified ethical reward via keyword detection."""
        txt = text.lower()
        pos = ["help", "safe", "ethical", "benefit", "aide", "bienveillant"]
        neg = ["harm", "attack", "steal", "kill", "nuire", "violence"]
        score = sum(1 for w in pos if w in txt) - sum(1 for w in neg if w in txt)
        return float(np.tanh(score))

    def H_safe(self, T_val, V_val, Z_val, U_val, R_val):
        """Final H_SAFE Score."""
        return float(T_val + V_val - Z_val + self.lambda_U * U_val + self.beta_R * R_val)

    def step(self, text: str, goal_text: str, samples_for_Htext=None, alt_model_texts=None, ideal_response_text=None):
        """Complete H_SAFE measurement pipeline."""
        E = self.embed(text)
        G = self.embed(goal_text)
        
        T_val = self.T(E)
        V_val = self.V(E, G)
        Ht = self.H_text(samples_for_Htext or [text])
        Hs = self.H_state(E, self.embed(ideal_response_text) if ideal_response_text else E)
        Hd = self.H_div([self.embed(t) for t in (alt_model_texts or [])] + [E])
        Z_val = self.Z(Ht, Hs, Hd)
        U_val = self.U(E)
        R_val = self.R(text)
        
        h_safe_val = self.H_safe(T_val, V_val, Z_val, U_val, R_val)
        
        # Internal update
        self.prev_E = E.copy()
        self.history.append(E.copy())
        if len(self.history) > self.max_history: self.history.pop(0)
        
        return {
            "H_safe": h_safe_val,
            "T": T_val, "V": V_val, "Z": Z_val,
            "U": U_val, "R": R_val
        }
