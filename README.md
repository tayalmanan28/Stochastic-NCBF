# Stochastic Neural Control Barrier Functions (SNCBF)

Training and verification of Neural Control Barrier Functions for stochastic dynamical systems using Lipschitz-bounded neural networks with LMI-based certificates.

## Method

We learn a smooth neural barrier function $h(x)$ parameterized by a SoftPlus network whose Lipschitz constants are verified via Linear Matrix Inequalities (LMIs). The training jointly optimizes:

- **Barrier conditions** &mdash; $h(x) \geq 0$ on the safe set, $h(x) < 0$ on the unsafe set
- **Lie derivative condition** &mdash; $L_f h + L_g h \cdot u + \tfrac{1}{2}\mathrm{tr}(\sigma^\top \nabla^2 h\, \sigma) + \gamma h \geq 0$ on the domain, enforced via a safe QP controller
- **LMI Lipschitz certificates** &mdash; verified bounds $L_h$, $L_{\nabla h}$, $L_{\nabla^2 h}$ using the network weights
- **Scenario verification** &mdash; the gap parameter $\eta$ satisfies $(L_h + L_{\nabla h} L_x + L_{\nabla^2 h})\varepsilon + \eta \leq 0$

## Systems

| System | Dim | Domain | Lipschitz ($L_h$, $L_{\nabla h}$, $L_{\nabla^2 h}$) | $\sigma$ |
|---|---|---|---|---|
| Inverted Pendulum | 2 | $[-\pi/4, \pi/4]^2$ | 0.01, 0.4, 2 | $0.1 I_2$ |
| Unicycle | 3 | $[-2, 2]^3$ | 1, 1, 2 | $0.1 I_3$ |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train inverted pendulum
python main.py ip

# Train unicycle
python main.py uni
```

Training outputs (model checkpoints, logs, plots) are saved to `experiments/<system>_w_eta/`.

## Configuration

Training hyperparameters are defined in `superp_init.py` and overridden per-system in `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `EPOCHS` | 500 | Number of training epochs |
| `N_H_B` | 1 | Hidden layers in barrier network |
| `D_H_B` | 20 | Neurons per hidden layer |
| `DIM_S` | 2 | State dimension (set per system) |
| `lip_h` | 1 | Lipschitz bound on $h$ |
| `lip_dh` | 1 | Lipschitz bound on $\nabla h$ |
| `lip_d2h` | 2 | Lipschitz bound on $\nabla^2 h$ |

## License

MIT License. See [LICENSE](LICENSE).
