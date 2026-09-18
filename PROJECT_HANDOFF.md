# Fly-Brain Game: Project Handoff

Last updated: 2026-09-18

> **Current-direction note:** The product concept evolved after this handoff was written. `PROJECT_PLAN.md` now defines the current game: a tiny winged human dodges the hands of a giant front-facing fly in a complex-2D browser game. Where the two documents conflict, follow `PROJECT_PLAN.md`.

> **Environment correction (2026-09-18):** Direct diagnostics identified the development GPU as an NVIDIA GeForce RTX 4060 Ti with 16,380 MiB reported VRAM and driver 591.86. All 5060 Ti references below are superseded historical assumptions.

## Purpose of this document

This file transfers the product intent, technical research, constraints, and decisions from a resume-planning conversation into the standalone game/ML project. It is context for planning, not a claim that any implementation or result already exists.

The next Codex session should read this entire file before proposing an implementation plan or modifying the project.

## Why this project exists

The project is intended to become a memorable portfolio piece for software engineering and ML recruiting. Its first layer must be understandable to a recruiter who may know little about machine learning; its deeper layer must give engineers and interviewers substantial architecture, training, systems, and experimental-design material.

The recruiter-facing hook is approximately:

> You control a fly trying to survive while an AI built around the measured wiring of roughly 165,000 fruit-fly neurons controls a human arm and tries to swat you.

This is more immediately visual and memorable than leading with a model name or an abstract ML technique. The finished project should nevertheless be technically honest and experimentally defensible.

Naming is intentionally unresolved. `flybrain-game` is only the folder name. Ideas discussed included "FlyBrain Swatter," "Reverse Human Fly," "SwatCNS," and "The Fly Brain Hunts You," but no product name has been chosen.

## Exact game concept

- A human player controls a small fly and tries to dodge attacks and survive.
- The AI controls a human character in the background, particularly an articulated arm and hand.
- The AI must predict the fly's future location, move the hand into an intercept trajectory, and slap the fly.
- The project is not conventional tag and the connectome does not control the player's fly.
- A missed slap should incur a cooldown or stamina cost so the AI cannot cover the screen by attacking continuously.
- The initial objective can be survival for a fixed duration. Later gameplay may include food collection, obstacles, hiding regions, multiple hands, and difficulty levels.

The core interaction is predictive interception rather than simply chasing the fly's current coordinates. If the arm takes approximately `tau` seconds to reach the target, a simple expert might begin from:

```text
predicted_position = current_position + velocity * tau
```

The learned recurrent controller should eventually improve on this simple assumption when the player turns, accelerates, feints, or develops a recognizable movement style.

## Honest scientific framing

Do not claim that the project uploads, recreates, or simulates a literal biological fly mind.

Preferred descriptions:

- "A recurrent neural network constrained by the measured topology of the MaleCNS fruit-fly connectome."
- "An AI whose recurrent architecture is based on the measured wiring of approximately 165,000 fruit-fly neurons."
- "A MaleCNS-constrained recurrent controller."

The connectome primarily supplies which traced neurons connect to which. Existing implementations still make substantial engineered choices about neuron dynamics, sensory injection, trainable parameters, readouts, and the controlled body. Important biological details such as ion channels, synaptic dynamics, neuromodulation, learning rules, and exact encoding are missing or simplified.

Using a fly connectome to control a human arm is deliberately anatomically unrelated. That is acceptable and can become part of the research question:

> Does biological connectome topology provide a useful inductive bias for predictive interception through an anatomically unrelated embodiment?

Success would demonstrate a trainable connectome-constrained computational substrate, not reanimated fly cognition.

## Relevant upstream work

### MaleCNS

- Official site: https://male-cns.janelia.org/
- Complete adult male Drosophila central nervous system connectome, including the brain and ventral nerve cord.
- The derived project must preserve and document all upstream data and licensing requirements.

### flyhard

- Repository: https://github.com/MarkUnthank/flyhard
- Uses the MaleCNS topology as a sparse recurrent rate model.
- Current published pilot retains 165,122 traced neurons and 25,563,197 measured neuron-pair connections.
- Learns one gain per connection and one leak per neuron while keeping graph topology fixed: 25,728,319 trainable parameters.
- A bounded steering-wheel task reportedly trained for 600 updates in 186 seconds on an RTX A6000 and used approximately 3 GB of peak allocated GPU memory.
- The ordinary PyTorch sparse backward initially attempted a roughly 101 GB dense intermediate. The repository's custom sparse gradient implementation is therefore essential; do not replace it casually with a dense or naive sparse path.
- The published result is preliminary and uses one training seed. Treat it as an engineering starting point, not conclusive evidence that biological topology is superior.

### fly-self-driving

- Repository: https://github.com/suanmiao/fly-self-driving
- Uses `flyhard` and supplies a faster CUDA backward patch.
- Maps a 64 x 32 observation into 4,114 optic sensory neurons and reads 708 motor neurons.
- Carries recurrent state between decisions and performs four graph updates per 50 ms decision.
- Reports that preserving state was critical on its first task.
- Training recipe: expert demonstrations, behavior cloning with truncated backpropagation through time, then DAgger.
- Published configuration uses batch size 4, windows of 16 decisions, 1,200 behavior-cloning updates, and three DAgger rounds of 500 updates.
- Reported training time is approximately 35 minutes on one H100.
- Results are self-reported, recent, and generally one training seed per condition. Reproduce locally before relying on behavior or performance claims.

### Scientifically grounded references

- Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing," Nature (2024): https://doi.org/10.1038/s41586-024-07763-9
- Reference implementation: https://github.com/philshiu/Drosophila_brain_model
- FlyGym/NeuroMechFly may be useful as a reference for embodied simulation, but the first game should not depend on a complete 3D fly or CARLA environment.

## Hardware and environment

The original discussion assumed an NVIDIA GeForce RTX 5060 Ti. Direct diagnostics later confirmed an NVIDIA GeForce RTX 4060 Ti with 16,380 MiB reported VRAM; see `docs/environment.md` for the current baseline.

First diagnostic command:

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Also verify PyTorch after installation:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_device_properties(0).total_memory / 1024**3)"
```

Current assessment:

- The confirmed 16 GB card is a plausible fit for full-graph experiments, but this must be measured locally.
- Use a recent NVIDIA driver and a PyTorch build compatible with both the installed driver and upstream requirements. `flyhard` reports a tested PyTorch 2.8.0 + CUDA 12.8 environment.
- Results reported on an H100 or A6000 must not be converted into a confident 4060 Ti estimate. Expect hours and possibly overnight experiments until measured.
- Train headlessly and render polished demonstrations separately.
- WSL2/Linux may be easier for upstream repositories than native Windows. Do not assume this until a small compatibility spike compares the practical options.

Record peak VRAM, GPU utilization, host RAM, compilation/startup time, steps per second, and end-to-end update time during the first benchmark.

## Major architecture decision

Do not initially ask the connectome to emit raw torques for a full 3D shoulder, elbow, wrist, palm, and fingers. That unnecessarily turns the project into a difficult robotics problem.

Use a hierarchical controller:

```text
game observation
    -> sensory encoder
    -> MaleCNS-constrained recurrent network
    -> high-level target and strike decision
    -> deterministic inverse kinematics / motion controller
    -> articulated human arm and palm hitbox
```

Initial network outputs should be only:

```text
[target_x, target_y, slap_probability]
```

An analytic two-link inverse-kinematics controller can translate the target into shoulder and elbow motion. Later versions may add hand selection, palm orientation, reach speed, and independent targets for two hands.

This engineered interface must be documented. It is not cheating; existing connectome projects also require explicit sensory and motor mappings.

## Scope progression

### MVP: articulated 2D game

- Python simulation with a headless mode.
- Pygame is the leading option for the first playable visualization, but this is not yet a locked choice.
- Player-controlled fly with keyboard and/or mouse movement.
- One human arm with shoulder and elbow joints plus a palm hitbox.
- Limited arm speed, acceleration, reach, and slap cooldown.
- Collision detection, survival timer, reset, deterministic seeds, and replay logging.
- Decouple simulation state from rendering so training can run faster than real time.

### Later presentation layer

- Improve the visual design after learning works.
- Godot or Unity may become appropriate for a polished 2D/2.5D or browser demo.
- A trained PyTorch policy could be exported through ONNX or queried through a local service.
- Full 3D physics and direct joint torques are stretch goals, not MVP requirements.
- Fingers should not be modeled until they contribute meaningfully to gameplay or research.

## Input progression

Begin with structured state so failures in the learning pipeline are diagnosable:

```text
fly position and velocity
hand position and velocity
arm joint angles
slap cooldown
recent trajectory / recurrent state
```

After the structured-input controller works, add a pixel-observation condition. A final pixel-based version would be more visually and scientifically interesting because sensory pixels can be mapped into annotated optic sensory neurons, similar to `fly-self-driving`.

Keep structured and pixel versions as separate experimental conditions rather than silently replacing one with the other.

## Training plan

Pure reinforcement learning with only a successful-slap reward is likely to be unnecessarily sparse and unstable. Start with imitation learning.

### Expert controller

- Estimate future fly position based on position, velocity, estimated hand travel time, reachability, and cooldown.
- Clamp targets to the arm's reachable workspace.
- Use inverse kinematics to execute the strike.
- Delay or cancel attacks with poor interception confidence.
- Log expert targets, strike decisions, and complete environment state.

### Curriculum

1. Stationary targets.
2. Constant-velocity scripted targets.
3. Turning and accelerating scripted trajectories.
4. Random-waypoint fly controller.
5. Simple evasive bot.
6. Recorded human-player trajectories.
7. Live human players.

### Learning sequence

1. Train small conventional baselines first to validate data and loss functions.
2. Train the connectome controller through behavior cloning with truncated backpropagation through time.
3. Preserve recurrent state between decisions; also implement a reset-state ablation.
4. Run DAgger: allow the learned controller to create its own states, relabel those states with the expert, and retrain.
5. Optionally fine-tune through reinforcement learning only after supervised interception works.

Potential imitation objectives:

- Regression loss for target coordinates.
- Binary cross-entropy for slap/no-slap.
- Optional auxiliary loss for predicted fly position at hand-arrival time.

Potential later reward terms:

- Positive reward for a valid hit.
- Shaped reward for smaller closest-approach distance.
- Penalties for wasted slaps, excessive motion, cooldown violations, energy use, and unnatural joint changes.

## Required baselines and ablations

The demo alone is memorable, but the controlled experiment creates technical depth.

Compare at least:

1. Real measured MaleCNS topology.
2. Degree-preserving shuffled topology.
3. Random sparse recurrent topology.
4. Parameter- or compute-matched conventional RNN/GRU.
5. Simple MLP or analytic expert as diagnostic baselines.
6. Connectome with recurrent state carried versus reset each decision.

Use multiple seeds before making claims that the measured topology is better. Three seeds is a minimum exploratory target; five or more is preferable if compute permits.

Candidate metrics:

- Hit rate.
- Median player survival time.
- Time to first hit.
- Closest near-miss distance.
- Invalid or wasted slap rate.
- Sample efficiency and wall-clock training time.
- Generalization to unseen scripted trajectories.
- Generalization to unseen human players and movement styles.
- Robustness to observation delay, noisy observations, and increased fly speed.
- Behavior after neuron or connection perturbation.

Negative or equal results are acceptable if reported honestly. The project should not claim that biological topology is superior unless the replicated comparisons support it.

## Recommended implementation phases

### Phase 0: repository and feasibility benchmark

- Initialize an independent Git repository.
- Add a concise README explaining the concept and its honest framing.
- Record the GPU, driver, Python, PyTorch, CUDA, OS/WSL, RAM, and storage environment.
- Inspect upstream licenses and decide whether to depend on, fork, vendor, or reimplement the minimal reusable connectome core.
- Download and verify the connectome through the upstream acquisition process.
- Run the upstream full-graph benchmark before building the game around it.
- Measure compatibility, VRAM, throughput, and numerical correctness on the RTX 4060 Ti 16 GB.
- Run a tiny bounded training job and save its metrics.
- Do not begin a long full experiment before this gate passes.

### Phase 1: deterministic game environment

- Define coordinate system, fixed timestep, seeded reset, observation space, action space, and termination rules.
- Implement the fly, one articulated arm, analytic inverse kinematics, palm hitbox, cooldown, and headless stepping.
- Separate simulation, rendering, player input, policies, and logging.
- Add tests for kinematics, reachability, collision, cooldown, reproducibility, and trajectory logging.

### Phase 2: expert and dataset

- Implement predictive expert interception.
- Generate curriculum trajectories and record expert labels.
- Add visual replay inspection and quantitative expert evaluation.

### Phase 3: baseline learning

- Train a small MLP/RNN/GRU before introducing the full connectome.
- Validate sequence batching, recurrent-state carry, target loss, strike loss, checkpoints, and evaluation.

### Phase 4: connectome integration

- Integrate the tested sparse MaleCNS core.
- Define and document structured sensory injection and motor readout.
- Start with batch size 1 or 2 and a short TBPTT window; scale only after profiling.
- Confirm gradients are finite and meaningful without changing graph topology.
- Compare initial and trained performance on held-out trajectories.

### Phase 5: DAgger and human trajectories

- Collect states visited by the learned controller.
- Relabel with the expert and retrain over several bounded rounds.
- Add recorded human-play sessions before live online adaptation.

### Phase 6: evaluation

- Automate baselines, ablations, multiple seeds, metrics, plots, and failure replays.
- Separate exploratory results from replicated results.
- Preserve configurations, source hashes, checkpoints, and evaluation seeds.

### Phase 7: recruiter-facing polish

- Build an immediately understandable playable demo or short video.
- Show the player fly, human arm, palm trajectory, survival timer, and neural activity together.
- Provide a simple architecture diagram and a plain-language explanation before technical detail.
- Make failures and limitations visible rather than selecting only one successful clip.

## Portfolio and resume strategy

The project needs three layers:

1. Recruiter: an unusual, instantly understood game premise and polished visual.
2. Engineer: concrete architecture, real-time system, data pipeline, tests, and measured performance.
3. Interviewer/researcher: biological-topology comparison, recurrent dynamics, DAgger, ablations, limitations, and failed experiments.

Potential future resume wording, with every placeholder replaced only by measured results:

> Built an interactive game where players control a fly while a recurrent network constrained by the connectivity of 165K mapped fruit-fly neurons controls an articulated human arm to predict and swat them.

> Trained the controller through imitation learning and DAgger on scripted, autonomous, and human-recorded flight trajectories, achieving [measured result] against previously unseen players.

> Compared measured biological connectivity against shuffled and parameter-matched recurrent networks across [N] trials, evaluating interception accuracy, sample efficiency, and robustness.

Never manufacture metrics or imply results before they exist.

## Decisions still open

- Final project/product name.
- Whether the confirmed RTX 4060 Ti 16 GB can run the bounded upstream workload at a useful iteration speed.
- Native Windows versus WSL2 for the ML environment.
- Pygame versus another lightweight renderer for the MVP.
- Exact relationship to `flyhard`: dependency, fork, vendored core, or clean-room adaptation under its license.
- Whether the first connectome condition uses structured state exclusively or immediately includes a small pixel condition after the benchmark.
- Exact sensory-neuron selection and output/readout scheme.
- Whether internal edge gains and neuron leaks are both trained in the first experiment.
- Minimum baseline matching criterion: parameter count, active parameter count, compute, or degree distribution.

Do not lock these prematurely. Resolve them with a short technical spike, profiling data, upstream license review, and explicit user agreement.

## Recommended first prompt for a new Codex session

```text
Read PROJECT_HANDOFF.md and PROJECT_PLAN.md completely. This is a standalone project, and nothing described as a future result should be treated as implemented without checking the repository. Continue the staged implementation beginning with environment verification and a bounded RTX 4060 Ti 16 GB full-connectome benchmark. Verify current upstream repositories and compatibility before depending on them. Clearly separate decisions, assumptions, open questions, and measurable phase gates. Do not fabricate biological or performance claims, and do not begin a long training run yet.
```
