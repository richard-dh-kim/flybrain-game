import Phaser from "phaser";
import initWasm, { Simulation } from "./generated/wasm/flybrain_game.js";
import {
  decideScriptedAction,
  type PolicyAction,
  type ScriptedPolicyMode,
} from "./scriptedPolicy";
import { LearnedGruPolicy } from "./learnedPolicy";
import {
  ConnectomePolicy,
  type ConnectomePolicySnapshot,
} from "./connectomePolicy";
import { FlyBrainVisualization } from "./brainVisualization";
import "./style.css";

const WIDTH = 960;
const HEIGHT = 540;
const TABLE_TOP = 438;
const UNITS_PER_PIXEL = 1_024;
const ROUND_ACTIVE = 0;
const ROUND_HIT = 1;
const ROUND_SURVIVED = 2;
const HAND_PHASE_TRACK = 0;
const HAND_PHASE_WIND_UP = 1;
const HAND_PHASE_STRIKE = 2;
const HAND_PHASE_IMPACT = 3;
const HAND_PHASE_RECOVER = 4;
const HAND_WIND_UP_TICKS = 12;
const HAND_STRIKE_TICKS = 14;
const IMPACT_EFFECT_MS = 360;
const REACTION_TEXT_MS = 620;
const HAND_PHASE_NAMES = ["track", "wind-up", "strike", "impact", "recover"] as const;
type PolicyMode = ScriptedPolicyMode | "learned" | "connectome";

interface BrowserSnapshot {
  tick: number;
  roundStatus: number;
  handPhase: number;
  playerX: number;
  playerY: number;
  policyMode: PolicyMode;
  pointerActive: boolean;
  debugEnabled: boolean;
  debugPanelVisible: boolean;
  reactionText: string | null;
  simulationRateHz: number;
  renderRateFps: number;
  connectome: ConnectomePolicySnapshot;
}

interface HumanTrajectoryRow {
  tick: number;
  destinationXUnits: number;
  destinationYUnits: number;
}

declare global {
  interface Window {
    __flybrainGame?: {
      snapshot: () => BrowserSnapshot;
      humanTrajectoryCsv: () => string;
    };
  }
}

class GrayboxScene extends Phaser.Scene {
  private player!: Phaser.GameObjects.Container;
  private flyHead!: Phaser.GameObjects.Container;
  private flyWings!: Phaser.GameObjects.Graphics;
  private flyExpression!: Phaser.GameObjects.Graphics;
  private hands!: Phaser.GameObjects.Graphics;
  private telegraphGraphics!: Phaser.GameObjects.Graphics;
  private effectsGraphics!: Phaser.GameObjects.Graphics;
  private reactionText!: Phaser.GameObjects.Text;
  private debugGraphics!: Phaser.GameObjects.Graphics;
  private simulation!: Simulation;
  private restartKey!: Phaser.Input.Keyboard.Key;
  private debugKey!: Phaser.Input.Keyboard.Key;
  private exportKey!: Phaser.Input.Keyboard.Key;
  private timerText!: Phaser.GameObjects.Text;
  private brainText!: Phaser.GameObjects.Text;
  private statusText!: Phaser.GameObjects.Text;
  private debugText!: Phaser.GameObjects.Text;
  private connectomeStatusElement!: HTMLElement;
  private brainVisualization!: FlyBrainVisualization;
  private policyKeys!: Record<PolicyMode, Phaser.Input.Keyboard.Key>;
  private policyMode: PolicyMode = "connectome";
  private learnedPolicy = new LearnedGruPolicy();
  private connectomePolicy = new ConnectomePolicy();
  private connectomeAction: PolicyAction | null = null;
  private pointerDestination = new Phaser.Math.Vector2(WIDTH / 2, HEIGHT / 2);
  private pointerActive = false;
  private debugEnabled = false;
  private latestAction: PolicyAction = {
    targetXUnits: (WIDTH / 2) * UNITS_PER_PIXEL,
    targetYUnits: (HEIGHT / 2) * UNITS_PER_PIXEL,
    strike: false,
  };
  private accumulatedTime = 0;
  private humanTrajectory: HumanTrajectoryRow[] = [];
  private simulationRateHz = 0;
  private rateSampleStarted = performance.now();
  private rateSampleTick = 0;
  private presentationTimeMs = 0;
  private previousHandPhase = HAND_PHASE_TRACK;
  private previousRoundStatus = ROUND_ACTIVE;
  private impactEffectRemainingMs = 0;
  private missReactionRemainingMs = 0;
  private hitReactionRemainingMs = 0;
  private reactionTextRemainingMs = 0;
  private reactionTextOriginY = 0;
  private attackTarget = new Phaser.Math.Vector2(WIDTH / 2, HEIGHT / 2);
  private effectTarget = new Phaser.Math.Vector2(WIDTH / 2, HEIGHT / 2);

  create(): void {
    this.cameras.main.setBackgroundColor("#dca982");
    this.drawRoom();
    this.drawFly();
    this.simulation = new Simulation();
    this.telegraphGraphics = this.add.graphics().setDepth(7);
    this.hands = this.add.graphics().setDepth(8);
    this.effectsGraphics = this.add.graphics().setDepth(18);
    this.debugGraphics = this.add.graphics().setDepth(19);
    this.drawHands();
    this.player = this.createPlayer(this.simulation.player_x, this.simulation.player_y);
    this.reactionText = this.add
      .text(WIDTH / 2, HEIGHT / 2, "", {
        color: "#fff1cf",
        fontFamily: "Georgia, serif",
        fontSize: "30px",
        fontStyle: "bold italic",
        stroke: "#38251f",
        strokeThickness: 7,
      })
      .setOrigin(0.5)
      .setDepth(31)
      .setVisible(false);

    const keyboard = this.input.keyboard;
    if (!keyboard) {
      throw new Error("Keyboard input is unavailable in this browser.");
    }

    this.restartKey = keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.R);
    this.debugKey = keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.H);
    this.exportKey = keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.E);
    this.policyKeys = {
      idle: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.ONE),
      chase: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.TWO),
      predictive: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.THREE),
      learned: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.FOUR),
      connectome: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.FIVE),
    };
    this.input.on(Phaser.Input.Events.POINTER_MOVE, (pointer: Phaser.Input.Pointer) => {
      this.pointerDestination.set(pointer.worldX, pointer.worldY);
      this.pointerActive = true;
    });
    this.input.on(Phaser.Input.Events.POINTER_DOWN, (pointer: Phaser.Input.Pointer) => {
      this.pointerDestination.set(pointer.worldX, pointer.worldY);
      this.pointerActive = true;
    });

    this.add
      .text(24, 22, "SURVIVE THE SWAT", {
        color: "#38251f",
        fontFamily: "Georgia, serif",
        fontSize: "22px",
        fontStyle: "bold",
      })
      .setDepth(20);

    this.timerText = this.add
      .text(WIDTH / 2, 22, "30.0", {
        color: "#38251f",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        fontSize: "22px",
        fontStyle: "bold",
      })
      .setOrigin(0.5, 0)
      .setDepth(20);

    this.brainText = this.add
      .text(WIDTH - 24, 24, "FLY BRAIN: LOADING", {
        color: "#6b4033",
        fontFamily: "monospace",
        fontSize: "13px",
      })
      .setOrigin(1, 0)
      .setDepth(20);

    this.statusText = this.add
      .text(WIDTH / 2, HEIGHT / 2, "", {
        align: "center",
        color: "#fff1cf",
        fontFamily: "Georgia, serif",
        fontSize: "42px",
        fontStyle: "bold",
        stroke: "#38251f",
        strokeThickness: 8,
      })
      .setOrigin(0.5)
      .setDepth(30);

    this.debugText = this.add
      .text(18, 58, "", {
        backgroundColor: "rgba(24, 19, 16, 0.78)",
        color: "#fff1cf",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        fontSize: "12px",
        lineSpacing: 3,
        padding: { x: 8, y: 6 },
      })
      .setDepth(25)
      .setVisible(false);

    this.add
      .text(WIDTH - 18, HEIGHT - 16, "MOVE: MOUSE / TOUCH  ·  R RESTART  ·  H DETAILS", {
        color: "#d9b98d",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        fontSize: "11px",
      })
      .setOrigin(1, 1)
      .setDepth(20);

    const connectomeStatusElement = document.querySelector<HTMLElement>("#connectome-status");
    if (!connectomeStatusElement) {
      throw new Error("Connectome status element is missing.");
    }
    this.connectomeStatusElement = connectomeStatusElement;
    this.brainVisualization = new FlyBrainVisualization();
    this.connectomePolicy.start();
    this.syncConnectomeUi();
    this.syncBrainVisualization();

    window.__flybrainGame = {
      snapshot: () => this.browserSnapshot(),
      humanTrajectoryCsv: () => this.humanTrajectoryCsv(),
    };
  }

  update(_time: number, delta: number): void {
    if (Phaser.Input.Keyboard.JustDown(this.restartKey)) {
      this.simulation.restart();
      this.learnedPolicy.reset();
      this.connectomePolicy.reset();
      this.connectomeAction = null;
      this.resetRateSample();
      this.humanTrajectory = [];
      this.statusText.setText("");
      this.resetPresentation();
    }
    if (Phaser.Input.Keyboard.JustDown(this.debugKey)) {
      this.debugEnabled = !this.debugEnabled;
    }
    if (this.debugEnabled && Phaser.Input.Keyboard.JustDown(this.exportKey)) {
      this.downloadHumanTrajectory();
    }
    if (this.debugEnabled) {
      if (Phaser.Input.Keyboard.JustDown(this.policyKeys.idle)) {
        this.setPolicyMode("idle");
      } else if (Phaser.Input.Keyboard.JustDown(this.policyKeys.chase)) {
        this.setPolicyMode("chase");
      } else if (Phaser.Input.Keyboard.JustDown(this.policyKeys.predictive)) {
        this.setPolicyMode("predictive");
      } else if (Phaser.Input.Keyboard.JustDown(this.policyKeys.learned)) {
        this.setPolicyMode("learned");
      } else if (Phaser.Input.Keyboard.JustDown(this.policyKeys.connectome)) {
        this.setPolicyMode("connectome");
      }
    }

    if (
      this.policyMode === "connectome"
      && this.connectomePolicy.snapshot().status === "error"
    ) {
      this.policyMode = "learned";
      this.learnedPolicy.reset();
      this.connectomeAction = null;
      this.brainText.setText("FLY BRAIN: COMPACT FALLBACK");
    }
    this.syncConnectomeUi();

    const previousX = this.simulation.player_x;
    const fixedStepMilliseconds = 1_000 / 60;
    this.accumulatedTime = Math.min(this.accumulatedTime + delta, 250);

    while (this.accumulatedTime >= fixedStepMilliseconds) {
      let action: PolicyAction;
      if (this.policyMode === "connectome") {
        const connectome = this.connectomePolicy.snapshot();
        if (connectome.status === "ready") {
          const connectomeAction = this.connectomePolicy.takeAction(this.simulation);
          if (connectomeAction) {
            this.connectomeAction = connectomeAction;
          }
        }
        action = this.connectomeAction ?? this.learnedPolicy.decide(this.simulation);
      } else if (this.policyMode === "learned") {
        action = this.learnedPolicy.decide(this.simulation);
      } else {
        action = decideScriptedAction(this.policyMode, {
          tick: this.simulation.tick,
          playerXUnits: this.simulation.player_x_units,
          playerYUnits: this.simulation.player_y_units,
          playerVelocityXUnits: this.simulation.player_velocity_x_units,
          playerVelocityYUnits: this.simulation.player_velocity_y_units,
        });
      }
      this.latestAction = action;

      const destinationXUnits = Math.round(this.pointerDestination.x * UNITS_PER_PIXEL);
      const destinationYUnits = Math.round(this.pointerDestination.y * UNITS_PER_PIXEL);
      this.humanTrajectory.push({
        tick: this.simulation.tick,
        destinationXUnits,
        destinationYUnits,
      });

      this.simulation.step_toward_with_action(
        destinationXUnits,
        destinationYUnits,
        action.targetXUnits,
        action.targetYUnits,
        action.strike,
      );
      this.accumulatedTime -= fixedStepMilliseconds;
      if (this.policyMode === "connectome") {
        this.connectomePolicy.request(this.simulation);
      }
    }

    this.sampleSimulationRate();

    this.player.x = this.simulation.player_x;
    this.player.y = this.simulation.player_y;
    this.player.rotation = Phaser.Math.Clamp((this.player.x - previousX) / 24, -0.18, 0.18);
    this.updatePresentation(delta);
    this.drawFlyExpression();
    this.drawSlapTelegraph();
    this.drawHands();
    this.drawPresentationEffects();
    this.drawDebugOverlay();
    this.syncBrainVisualization();
    this.timerText.setText((this.simulation.round_remaining_ticks / 60).toFixed(1));

    if (this.simulation.round_status === ROUND_HIT) {
      this.statusText.setText("SWATTED\nPRESS R TO RETRY");
    } else if (this.simulation.round_status === ROUND_SURVIVED) {
      this.statusText.setText("YOU SURVIVED\nPRESS R TO FLY AGAIN");
    } else if (this.simulation.round_status === ROUND_ACTIVE) {
      this.statusText.setText("");
    }
  }

  private setPolicyMode(mode: PolicyMode): void {
    this.policyMode = mode;
    this.learnedPolicy.reset();
    this.connectomePolicy.reset();
    this.connectomeAction = null;
    this.resetRateSample();
    if (mode === "connectome") {
      this.connectomePolicy.start();
    }
    const label = mode === "idle"
      ? "DEV · IDLE [1]"
      : mode === "chase"
        ? "DEV · CURRENT CHASE [2]"
        : mode === "predictive"
          ? "DEV · PREDICTIVE [3]"
          : mode === "learned"
            ? "COMPACT GRU FALLBACK [4]"
            : "FULL MALECNS · LOADING [5]";
    this.brainText.setText(`FLY BRAIN: ${label}`);
    this.syncConnectomeUi();
  }

  private browserSnapshot(): BrowserSnapshot {
    return {
      tick: this.simulation.tick,
      roundStatus: this.simulation.round_status,
      handPhase: this.simulation.hand_phase(0),
      playerX: this.simulation.player_x,
      playerY: this.simulation.player_y,
      policyMode: this.policyMode,
      pointerActive: this.pointerActive,
      debugEnabled: this.debugEnabled,
      debugPanelVisible: this.debugText.visible,
      reactionText: this.reactionText.visible ? this.reactionText.text : null,
      simulationRateHz: this.simulationRateHz,
      renderRateFps: this.game.loop.actualFps,
      connectome: this.connectomePolicy.snapshot(),
    };
  }

  private syncConnectomeUi(): void {
    const connectome = this.connectomePolicy.snapshot();
    this.connectomeStatusElement.hidden = !this.debugEnabled;
    if (connectome.status === "idle") {
      this.connectomeStatusElement.textContent =
        "Full connectome (WebGPU): idle · 148 MiB local / about 71 MiB compressed";
    } else if (connectome.status === "loading") {
      const loaded = connectome.progress?.loadedBytes ?? 0;
      const total = connectome.progress?.totalBytes ?? 0;
      const percent = total > 0 ? Math.floor((loaded / total) * 100) : 0;
      const detail = connectome.progress?.detail ?? "starting";
      this.connectomeStatusElement.textContent =
        `Full connectome: loading ${percent}% · verifying ${detail}`;
      if (this.policyMode === "connectome") {
        this.brainText.setText(`FLY BRAIN: LOADING ${percent}%`);
      }
    } else if (connectome.status === "ready" && connectome.info) {
      const shortHash = connectome.info.packageSha256.slice(0, 8);
      const timing = connectome.inferenceMedianMs === null
        ? "warming up"
        : `brain ${connectome.inferenceMedianMs.toFixed(1)}/${(connectome.inferenceP95Ms ?? 0).toFixed(1)} ms med/p95 · trip p95 ${(connectome.roundTripP95Ms ?? 0).toFixed(1)} ms · queue ${connectome.pendingSteps}`;
      this.connectomeStatusElement.textContent =
        `Full connectome: ready · WebGPU u16 · ${connectome.info.adapter} · v${connectome.info.formatVersion} ${shortHash} · ${timing} · sim ${this.simulationRateHz.toFixed(0)} Hz · render ${this.game.loop.actualFps.toFixed(0)} fps`;
      if (this.policyMode === "connectome") {
        this.brainText.setText("FLY BRAIN: FULL MALECNS");
      }
    } else {
      this.connectomeStatusElement.textContent =
        `Full connectome unavailable: ${connectome.error ?? "unknown error"} · using GRU [4]`;
    }
  }

  private syncBrainVisualization(): void {
    const connectome = this.connectomePolicy.snapshot();
    if (this.policyMode === "connectome") {
      if (connectome.status === "ready") {
        this.brainVisualization.update({
          mode: "connectome",
          activity: this.connectomePolicy.activity(),
          step: connectome.activityStep,
        });
        return;
      }
      const loaded = connectome.progress?.loadedBytes ?? 0;
      const total = connectome.progress?.totalBytes ?? 0;
      this.brainVisualization.update({
        mode: "loading",
        activity: this.learnedPolicy.activity(),
        loadPercent: total > 0 ? Math.floor((loaded / total) * 100) : undefined,
      });
      return;
    }
    if (this.policyMode === "learned") {
      this.brainVisualization.update({
        mode: "fallback",
        activity: this.learnedPolicy.activity(),
        fallbackReason: connectome.status === "error"
          ? (connectome.error ?? "unknown error")
          : undefined,
      });
      return;
    }
    this.brainVisualization.update({ mode: "development", activity: [] });
  }

  private resetRateSample(): void {
    this.simulationRateHz = 0;
    this.rateSampleStarted = performance.now();
    this.rateSampleTick = this.simulation.tick;
  }

  private sampleSimulationRate(): void {
    const now = performance.now();
    const elapsed = now - this.rateSampleStarted;
    if (elapsed < 500) {
      return;
    }
    this.simulationRateHz = (
      (this.simulation.tick - this.rateSampleTick) * 1_000
    ) / elapsed;
    this.rateSampleStarted = now;
    this.rateSampleTick = this.simulation.tick;
  }

  private humanTrajectoryCsv(): string {
    const header = "tick,destination_x_units,destination_y_units";
    const rows = this.humanTrajectory.map(
      (row) => `${row.tick},${row.destinationXUnits},${row.destinationYUnits}`,
    );
    return [header, ...rows].join("\n") + "\n";
  }

  private downloadHumanTrajectory(): void {
    if (this.humanTrajectory.length === 0) {
      return;
    }
    const blob = new Blob([this.humanTrajectoryCsv()], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `flybrain-human-trajectory-${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  private drawRoom(): void {
    const graphics = this.add.graphics();
    graphics.fillStyle(0xf1d4ab).fillRect(0, 0, WIDTH, HEIGHT);
    graphics.fillStyle(0xc97d65).fillRect(0, 350, WIDTH, TABLE_TOP - 350);
    graphics.fillStyle(0x6b3829).fillRect(0, TABLE_TOP, WIDTH, HEIGHT - TABLE_TOP);
    graphics.fillStyle(0x87503a).fillRect(0, TABLE_TOP, WIDTH, 12);
    graphics.lineStyle(4, 0x38251f, 0.7).lineBetween(0, TABLE_TOP, WIDTH, TABLE_TOP);
  }

  private drawFly(): void {
    this.flyWings = this.add.graphics().setDepth(1);
    this.flyWings.fillStyle(0x799e91, 0.56).fillEllipse(300, 260, 300, 380);
    this.flyWings.fillStyle(0x799e91, 0.56).fillEllipse(660, 260, 300, 380);
    this.flyWings.lineStyle(7, 0x405a53, 0.55).strokeEllipse(300, 260, 300, 380);
    this.flyWings.strokeEllipse(660, 260, 300, 380);

    const face = this.add.graphics();
    face.fillStyle(0x4b4036).fillEllipse(0, 0, 410, 410);
    face.lineStyle(10, 0x2d2521).strokeEllipse(0, 0, 410, 410);
    face.fillStyle(0xb13b34).fillCircle(-90, -45, 108);
    face.fillCircle(90, -45, 108);
    face.lineStyle(9, 0x38251f).strokeCircle(-90, -45, 108);
    face.strokeCircle(90, -45, 108);
    face.lineStyle(2, 0xf2a46f, 0.55);
    for (let offset = -66; offset <= 66; offset += 22) {
      face.lineBetween(-156, -45 + offset, -24, -45 + offset);
      face.lineBetween(24, -45 + offset, 156, -45 + offset);
    }
    this.flyExpression = this.add.graphics();
    this.flyHead = this.add
      .container(WIDTH / 2, 265, [face, this.flyExpression])
      .setDepth(2);
  }

  private updatePresentation(delta: number): void {
    const elapsed = Math.min(delta, 50);
    this.presentationTimeMs += elapsed;
    const phase = this.simulation.hand_phase(0);
    const roundStatus = this.simulation.round_status;

    if (phase === HAND_PHASE_WIND_UP && this.previousHandPhase !== HAND_PHASE_WIND_UP) {
      this.attackTarget.set(
        Phaser.Math.Clamp(
          this.latestAction.targetXUnits / UNITS_PER_PIXEL,
          32,
          WIDTH - 32,
        ),
        Phaser.Math.Clamp(
          this.latestAction.targetYUnits / UNITS_PER_PIXEL,
          32,
          TABLE_TOP - 32,
        ),
      );
    }

    const enteredMissPresentation = (
      phase === HAND_PHASE_IMPACT
      && this.previousHandPhase !== HAND_PHASE_IMPACT
    ) || (
      phase === HAND_PHASE_RECOVER
      && this.previousHandPhase !== HAND_PHASE_RECOVER
      && this.missReactionRemainingMs <= 0
    );
    if (roundStatus === ROUND_ACTIVE && enteredMissPresentation) {
      this.effectTarget.copy(this.attackTarget);
      this.impactEffectRemainingMs = IMPACT_EFFECT_MS;
      this.missReactionRemainingMs = 820;
      this.showReaction("MISS!", "#67d5ff", this.attackTarget.x, this.attackTarget.y - 32);
    }

    if (roundStatus === ROUND_HIT && this.previousRoundStatus !== ROUND_HIT) {
      this.effectTarget.set(this.simulation.player_x, this.simulation.player_y);
      this.impactEffectRemainingMs = IMPACT_EFFECT_MS;
      this.hitReactionRemainingMs = 900;
      this.showReaction(
        "SMACK!",
        "#ffcf6b",
        this.simulation.player_x,
        Math.max(88, this.simulation.player_y - 108),
      );
    } else if (
      roundStatus === ROUND_SURVIVED
      && this.previousRoundStatus !== ROUND_SURVIVED
    ) {
      this.missReactionRemainingMs = 900;
      this.showReaction(
        "ESCAPED!",
        "#95f4b5",
        this.simulation.player_x,
        Math.max(88, this.simulation.player_y - 108),
      );
    }

    this.impactEffectRemainingMs = Math.max(0, this.impactEffectRemainingMs - elapsed);
    this.missReactionRemainingMs = Math.max(0, this.missReactionRemainingMs - elapsed);
    this.hitReactionRemainingMs = Math.max(0, this.hitReactionRemainingMs - elapsed);
    this.reactionTextRemainingMs = Math.max(0, this.reactionTextRemainingMs - elapsed);
    this.previousHandPhase = phase;
    this.previousRoundStatus = roundStatus;

    let headX = WIDTH / 2;
    let headY = 265 + Math.sin(this.presentationTimeMs * 0.0024) * 1.5;
    let headScale = 1;
    let headRotation = Math.sin(this.presentationTimeMs * 0.0017) * 0.004;
    if (phase === HAND_PHASE_WIND_UP) {
      const progress = Phaser.Math.Clamp(
        this.simulation.hand_phase_tick(0) / HAND_WIND_UP_TICKS,
        0,
        1,
      );
      headX += (this.attackTarget.x - WIDTH / 2) * 0.025 * progress;
      headY += (this.attackTarget.y - 265) * 0.018 * progress;
      headScale += progress * 0.025;
    } else if (phase === HAND_PHASE_STRIKE) {
      const progress = Phaser.Math.Clamp(
        this.simulation.hand_phase_tick(0) / HAND_STRIKE_TICKS,
        0,
        1,
      );
      headScale += 0.025 + Math.sin(progress * Math.PI) * 0.018;
      headY += 3;
    }
    if (this.missReactionRemainingMs > 0) {
      const strength = this.missReactionRemainingMs / 820;
      headX += Math.sin(this.presentationTimeMs * 0.07) * 5 * strength;
      headRotation += Math.sin(this.presentationTimeMs * 0.052) * 0.025 * strength;
    }
    if (this.hitReactionRemainingMs > 0) {
      headScale += 0.018 * (this.hitReactionRemainingMs / 900);
    }
    this.flyHead
      .setPosition(headX, headY)
      .setScale(headScale)
      .setRotation(headRotation);
    this.flyWings.setAlpha(0.91 + Math.sin(this.presentationTimeMs * 0.008) * 0.06);

    if (this.hitReactionRemainingMs > 0) {
      const impulse = Phaser.Math.Clamp(this.hitReactionRemainingMs / 900, 0, 1);
      this.player.setScale(1 + impulse * 0.32, 1 - impulse * 0.22);
    } else {
      this.player.setScale(1);
    }

    if (this.reactionTextRemainingMs > 0) {
      const progress = 1 - this.reactionTextRemainingMs / REACTION_TEXT_MS;
      this.reactionText
        .setVisible(true)
        .setY(this.reactionTextOriginY - progress * 30)
        .setAlpha(Math.min(1, this.reactionTextRemainingMs / 180))
        .setScale(1 + progress * 0.18);
    } else {
      this.reactionText.setVisible(false);
    }
  }

  private drawFlyExpression(): void {
    this.flyExpression.clear();
    const phase = this.simulation.hand_phase(0);
    const useAttackTarget = (
      phase === HAND_PHASE_WIND_UP
      || phase === HAND_PHASE_STRIKE
      || phase === HAND_PHASE_IMPACT
    );
    const gazeTargetX = useAttackTarget ? this.attackTarget.x : this.simulation.player_x;
    const gazeTargetY = useAttackTarget ? this.attackTarget.y : this.simulation.player_y;
    const gazeX = Phaser.Math.Clamp((gazeTargetX - WIDTH / 2) / 260, -1, 1) * 22;
    const gazeY = Phaser.Math.Clamp((gazeTargetY - 220) / 210, -1, 1) * 18;

    this.flyExpression.fillStyle(0x2d2521);
    this.flyExpression.fillCircle(-90 + gazeX, -45 + gazeY, 29);
    this.flyExpression.fillCircle(90 + gazeX, -45 + gazeY, 29);
    this.flyExpression.fillStyle(0xfff1cf, 0.78);
    this.flyExpression.fillCircle(-99 + gazeX, -55 + gazeY, 7);
    this.flyExpression.fillCircle(81 + gazeX, -55 + gazeY, 7);

    const isAttacking = phase === HAND_PHASE_WIND_UP || phase === HAND_PHASE_STRIKE;
    const isFrustrated = this.missReactionRemainingMs > 0;
    this.flyExpression.lineStyle(11, 0x2d2521, 0.95);
    if (isFrustrated) {
      this.flyExpression.lineBetween(-150, -91, -48, -111);
      this.flyExpression.lineBetween(48, -111, 150, -91);
    } else if (isAttacking) {
      this.flyExpression.lineBetween(-150, -116, -48, -88);
      this.flyExpression.lineBetween(48, -88, 150, -116);
    } else {
      this.flyExpression.lineBetween(-146, -108, -52, -101);
      this.flyExpression.lineBetween(52, -101, 146, -108);
    }

    this.flyExpression.lineStyle(10, 0x2d2521, 1).beginPath();
    if (isFrustrated) {
      this.flyExpression.moveTo(-48, 112).lineTo(0, 89).lineTo(48, 112);
    } else if (this.hitReactionRemainingMs > 0) {
      this.flyExpression.moveTo(-58, 87).lineTo(0, 119).lineTo(58, 87);
    } else if (isAttacking) {
      this.flyExpression.moveTo(-55, 91).lineTo(0, 121).lineTo(55, 91);
    } else {
      this.flyExpression.moveTo(-50, 90).lineTo(0, 108).lineTo(50, 90);
    }
    this.flyExpression.strokePath();
    if (isFrustrated) {
      this.flyExpression.fillStyle(0x67d5ff, 0.78);
      this.flyExpression.fillTriangle(144, -91, 156, -121, 167, -91);
      this.flyExpression.fillCircle(155, -87, 11);
    } else if (this.hitReactionRemainingMs > 0) {
      this.flyExpression.lineStyle(5, 0xfff1cf, 0.62);
      this.flyExpression.lineBetween(-38, 103, 38, 103);
    }
  }

  private drawSlapTelegraph(): void {
    this.telegraphGraphics.clear();
    const phase = this.simulation.hand_phase(0);
    if (
      phase !== HAND_PHASE_WIND_UP
      && phase !== HAND_PHASE_STRIKE
      && phase !== HAND_PHASE_IMPACT
    ) {
      return;
    }

    const windUpProgress = phase === HAND_PHASE_WIND_UP
      ? Phaser.Math.Clamp(this.simulation.hand_phase_tick(0) / HAND_WIND_UP_TICKS, 0, 1)
      : 1;
    const strikeProgress = phase === HAND_PHASE_STRIKE
      ? Phaser.Math.Clamp(this.simulation.hand_phase_tick(0) / HAND_STRIKE_TICKS, 0, 1)
      : 0;
    const pulse = 0.5 + Math.sin(this.presentationTimeMs * 0.03) * 0.5;
    const warningColor = phase === HAND_PHASE_WIND_UP ? 0xf2cf74 : 0xe8573f;
    const targetRadius = phase === HAND_PHASE_WIND_UP
      ? 52 - windUpProgress * 20
      : 29 + pulse * 4;

    this.telegraphGraphics
      .fillStyle(0xd94b35, 0.045 + windUpProgress * 0.065)
      .fillCircle(this.attackTarget.x, this.attackTarget.y, targetRadius + 11)
      .lineStyle(3 + strikeProgress * 2, warningColor, 0.6 + pulse * 0.35)
      .strokeCircle(this.attackTarget.x, this.attackTarget.y, targetRadius);

    const bracket = 13;
    const gap = targetRadius + 5;
    this.telegraphGraphics.lineStyle(3, warningColor, 0.9);
    this.telegraphGraphics.lineBetween(
      this.attackTarget.x - gap - bracket,
      this.attackTarget.y,
      this.attackTarget.x - gap,
      this.attackTarget.y,
    );
    this.telegraphGraphics.lineBetween(
      this.attackTarget.x + gap,
      this.attackTarget.y,
      this.attackTarget.x + gap + bracket,
      this.attackTarget.y,
    );
    this.telegraphGraphics.lineBetween(
      this.attackTarget.x,
      this.attackTarget.y - gap - bracket,
      this.attackTarget.x,
      this.attackTarget.y - gap,
    );
    this.telegraphGraphics.lineBetween(
      this.attackTarget.x,
      this.attackTarget.y + gap,
      this.attackTarget.x,
      this.attackTarget.y + gap + bracket,
    );

    if (phase === HAND_PHASE_STRIKE) {
      this.telegraphGraphics.lineStyle(4, 0xe8573f, 0.22 + strikeProgress * 0.38);
      for (let index = 0; index < 2; index += 1) {
        this.telegraphGraphics.lineBetween(
          this.simulation.hand_palm_x(index),
          this.simulation.hand_palm_y(index),
          this.attackTarget.x,
          this.attackTarget.y,
        );
      }
    }
  }

  private drawPresentationEffects(): void {
    this.effectsGraphics.clear();
    if (this.impactEffectRemainingMs <= 0) {
      return;
    }
    const progress = 1 - this.impactEffectRemainingMs / IMPACT_EFFECT_MS;
    const alpha = Math.max(0, 1 - progress);
    const color = this.hitReactionRemainingMs > 0 ? 0xffcf6b : 0x67d5ff;
    const innerRadius = 18 + progress * 28;
    const outerRadius = 42 + progress * 72;
    this.effectsGraphics
      .lineStyle(5 - progress * 3, color, alpha * 0.9)
      .strokeCircle(this.effectTarget.x, this.effectTarget.y, innerRadius);
    this.effectsGraphics.lineStyle(3, color, alpha * 0.78);
    for (let ray = 0; ray < 12; ray += 1) {
      const angle = (Math.PI * 2 * ray) / 12;
      this.effectsGraphics.lineBetween(
        this.effectTarget.x + Math.cos(angle) * (innerRadius + 8),
        this.effectTarget.y + Math.sin(angle) * (innerRadius + 8),
        this.effectTarget.x + Math.cos(angle) * outerRadius,
        this.effectTarget.y + Math.sin(angle) * outerRadius,
      );
    }
  }

  private showReaction(text: string, color: string, x: number, y: number): void {
    this.reactionTextRemainingMs = REACTION_TEXT_MS;
    this.reactionTextOriginY = y;
    this.reactionText
      .setText(text)
      .setColor(color)
      .setPosition(x, y)
      .setAlpha(1)
      .setScale(1)
      .setVisible(true);
  }

  private resetPresentation(): void {
    this.presentationTimeMs = 0;
    this.previousHandPhase = HAND_PHASE_TRACK;
    this.previousRoundStatus = ROUND_ACTIVE;
    this.impactEffectRemainingMs = 0;
    this.missReactionRemainingMs = 0;
    this.hitReactionRemainingMs = 0;
    this.reactionTextRemainingMs = 0;
    this.attackTarget.set(WIDTH / 2, HEIGHT / 2);
    this.effectTarget.copy(this.attackTarget);
    this.telegraphGraphics.clear();
    this.effectsGraphics.clear();
    this.reactionText.setVisible(false);
    this.flyHead.setPosition(WIDTH / 2, 265).setScale(1).setRotation(0);
    this.player.setScale(1);
  }

  private drawHands(): void {
    this.hands.clear();
    for (let index = 0; index < 2; index += 1) {
      const palmX = this.simulation.hand_palm_x(index);
      const palmY = this.simulation.hand_palm_y(index);
      const phase = this.simulation.hand_phase(index);
      const activeContact = this.simulation.hand_has_active_contact(index);
      const palmColor = activeContact
        ? 0xd94b35
        : phase === HAND_PHASE_WIND_UP
          ? 0x9d6749
          : phase === HAND_PHASE_STRIKE
            ? 0x744b3b
            : 0x5b4639;
      const fingerDirection = index === 0 ? 1 : -1;
      const palmWidth = phase === HAND_PHASE_STRIKE ? 72 : 64;
      const palmHeight = phase === HAND_PHASE_WIND_UP ? 82 : 76;

      if (phase === HAND_PHASE_STRIKE) {
        const trailDirection = index === 0 ? -1 : 1;
        this.hands.lineStyle(5, 0xe8a36f, 0.25);
        for (const trailY of [-18, 0, 18]) {
          this.hands.lineBetween(
            palmX + trailDirection * 38,
            palmY + trailY,
            palmX + trailDirection * 82,
            palmY + trailY,
          );
        }
      }

      this.hands.fillStyle(palmColor).fillEllipse(palmX, palmY, palmWidth, palmHeight);
      this.hands.lineStyle(5, 0x2d2521).strokeEllipse(palmX, palmY, palmWidth, palmHeight);
      for (const fingerY of [-22, 0, 22]) {
        const fingerX = palmX + fingerDirection * 29;
        this.hands.fillStyle(palmColor).fillCircle(fingerX, palmY + fingerY, 13);
        this.hands.lineStyle(4, 0x2d2521).strokeCircle(fingerX, palmY + fingerY, 13);
      }

      if (phase === HAND_PHASE_WIND_UP) {
        this.hands.lineStyle(3, 0xf2cf74, 0.9).strokeCircle(palmX, palmY, 48);
      }
    }
  }

  private drawDebugOverlay(): void {
    this.debugGraphics.clear();
    if (!this.debugEnabled) {
      this.debugText.setText("");
      this.debugText.setVisible(false);
      return;
    }
    this.debugText.setVisible(true);

    const playerX = this.simulation.player_x;
    const playerY = this.simulation.player_y;
    const velocityX = this.simulation.player_velocity_x;
    const velocityY = this.simulation.player_velocity_y;
    const targetX = this.latestAction.targetXUnits / UNITS_PER_PIXEL;
    const targetY = this.latestAction.targetYUnits / UNITS_PER_PIXEL;
    const mouseX = this.pointerDestination.x;
    const mouseY = this.pointerDestination.y;

    this.debugGraphics.lineStyle(2, 0x35d2ff, 0.9);
    this.debugGraphics.strokeCircle(playerX, playerY, this.simulation.player_radius);
    this.debugGraphics.lineBetween(
      playerX,
      playerY,
      playerX + velocityX * 12,
      playerY + velocityY * 12,
    );

    this.debugGraphics.lineStyle(2, 0xffdf5d, 0.95);
    this.debugGraphics.lineBetween(targetX - 9, targetY, targetX + 9, targetY);
    this.debugGraphics.lineBetween(targetX, targetY - 9, targetX, targetY + 9);
    this.debugGraphics.strokeCircle(targetX, targetY, 13);

    this.debugGraphics.lineStyle(2, 0x71f79f, 0.95);
    this.debugGraphics.lineBetween(mouseX - 8, mouseY - 8, mouseX + 8, mouseY + 8);
    this.debugGraphics.lineBetween(mouseX - 8, mouseY + 8, mouseX + 8, mouseY - 8);
    this.debugGraphics.strokeCircle(mouseX, mouseY, 11);

    const handLines: string[] = [];
    for (let index = 0; index < 2; index += 1) {
      const palmX = this.simulation.hand_palm_x(index);
      const palmY = this.simulation.hand_palm_y(index);
      const phase = this.simulation.hand_phase(index);
      const phaseName = HAND_PHASE_NAMES[phase] ?? `unknown-${phase}`;
      const contact = this.simulation.hand_has_active_contact(index);

      this.debugGraphics.lineStyle(2, contact ? 0xff3f35 : 0x35d2ff, 0.95);
      this.debugGraphics.strokeCircle(
        palmX,
        palmY,
        this.simulation.hand_palm_radius(index),
      );
      handLines.push(
        `${index === 0 ? "L" : "R"} ${phaseName.padEnd(7)} t=${String(this.simulation.hand_phase_tick(index)).padStart(2)} cd=${String(this.simulation.hand_cooldown(index)).padStart(2)} contact=${Number(contact)}`,
      );
    }

    this.debugText.setText([
      `tick=${this.simulation.tick} round=${this.simulation.round_status} attack=${Number(this.simulation.attack_active)}`,
      `player=(${playerX.toFixed(1)}, ${playerY.toFixed(1)}) vel=(${velocityX.toFixed(2)}, ${velocityY.toFixed(2)})`,
      `mouse=(${mouseX.toFixed(1)}, ${mouseY.toFixed(1)}) hand-target=(${targetX.toFixed(1)}, ${targetY.toFixed(1)}) strike=${Number(this.latestAction.strike)}`,
      ...handLines,
      "dev: [1] idle  [2] chase  [3] predictive  [4] GRU  [5] MaleCNS  [E] export",
    ]);
  }

  private createPlayer(x: number, y: number): Phaser.GameObjects.Container {
    const shadow = this.add.ellipse(0, 22, 34, 10, 0x38251f, 0.2);
    const leftWing = this.add.ellipse(-13, -4, 18, 32, 0xe9f6ed, 0.86).setAngle(-24);
    const rightWing = this.add.ellipse(13, -4, 18, 32, 0xe9f6ed, 0.86).setAngle(24);
    const body = this.add.ellipse(0, 3, 21, 35, 0x3f6f82).setStrokeStyle(3, 0x38251f);
    const head = this.add.circle(0, -18, 11, 0xf0b58d).setStrokeStyle(3, 0x38251f);
    const player = this.add.container(x, y, [shadow, leftWing, rightWing, body, head]);
    player.setDepth(10);
    return player;
  }
}

await initWasm();

new Phaser.Game({
  type: Phaser.AUTO,
  parent: "game",
  width: WIDTH,
  height: HEIGHT,
  backgroundColor: "#f1d4ab",
  scene: GrayboxScene,
  render: {
    antialias: true,
    pixelArt: false,
  },
  scale: {
    mode: Phaser.Scale.FIT,
    autoCenter: Phaser.Scale.CENTER_BOTH,
  },
});
