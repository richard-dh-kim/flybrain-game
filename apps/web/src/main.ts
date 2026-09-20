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
import "./style.css";

const WIDTH = 960;
const HEIGHT = 540;
const TABLE_TOP = 438;
const UNITS_PER_PIXEL = 1_024;
const ROUND_ACTIVE = 0;
const ROUND_HIT = 1;
const ROUND_SURVIVED = 2;
const HAND_PHASE_NAMES = ["track", "wind-up", "strike", "impact", "recover"] as const;
type PolicyMode = ScriptedPolicyMode | "learned" | "connectome";

interface BrowserSnapshot {
  tick: number;
  roundStatus: number;
  playerX: number;
  playerY: number;
  policyMode: PolicyMode;
  pointerActive: boolean;
  debugEnabled: boolean;
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
  private hands!: Phaser.GameObjects.Graphics;
  private debugGraphics!: Phaser.GameObjects.Graphics;
  private brainActivityGraphics!: Phaser.GameObjects.Graphics;
  private simulation!: Simulation;
  private restartKey!: Phaser.Input.Keyboard.Key;
  private debugKey!: Phaser.Input.Keyboard.Key;
  private exportKey!: Phaser.Input.Keyboard.Key;
  private timerText!: Phaser.GameObjects.Text;
  private brainText!: Phaser.GameObjects.Text;
  private statusText!: Phaser.GameObjects.Text;
  private debugText!: Phaser.GameObjects.Text;
  private brainActivityText!: Phaser.GameObjects.Text;
  private connectomeStatusElement!: HTMLElement;
  private policyKeys!: Record<PolicyMode, Phaser.Input.Keyboard.Key>;
  private policyMode: PolicyMode = "predictive";
  private learnedPolicy = new LearnedGruPolicy();
  private connectomePolicy = new ConnectomePolicy();
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

  create(): void {
    this.cameras.main.setBackgroundColor("#dca982");
    this.drawRoom();
    this.drawFly();
    this.simulation = new Simulation();
    this.hands = this.add.graphics().setDepth(8);
    this.debugGraphics = this.add.graphics().setDepth(19);
    this.brainActivityGraphics = this.add.graphics().setDepth(24);
    this.drawHands();
    this.player = this.createPlayer(this.simulation.player_x, this.simulation.player_y);

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
      .text(WIDTH - 24, 24, "CONTROLLER: PREDICTIVE [3]", {
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
      .setDepth(25);

    this.brainActivityText = this.add
      .text(WIDTH - 22, 62, "COMPUTED GRU ACTIVITY\n64 hidden units", {
        align: "right",
        color: "#fff1cf",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        fontSize: "11px",
      })
      .setOrigin(1, 0)
      .setDepth(25)
      .setVisible(false);

    this.add
      .text(WIDTH - 18, HEIGHT - 16, "1–5 opponent  ·  R restart  ·  H debug  ·  E export", {
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
    this.syncConnectomeUi();

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
      this.humanTrajectory = [];
      this.statusText.setText("");
    }
    if (Phaser.Input.Keyboard.JustDown(this.debugKey)) {
      this.debugEnabled = !this.debugEnabled;
    }
    if (Phaser.Input.Keyboard.JustDown(this.exportKey)) {
      this.downloadHumanTrajectory();
    }
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

    if (
      this.policyMode === "connectome"
      && this.connectomePolicy.snapshot().status === "error"
    ) {
      this.policyMode = "learned";
      this.learnedPolicy.reset();
      this.brainText.setText("CONTROLLER: GRU FALLBACK [4]");
    }
    this.syncConnectomeUi();

    const previousX = this.simulation.player_x;
    const fixedStepMilliseconds = 1_000 / 60;
    this.accumulatedTime = Math.min(this.accumulatedTime + delta, 250);

    while (this.accumulatedTime >= fixedStepMilliseconds) {
      let action: PolicyAction;
      if (this.policyMode === "connectome") {
        const connectomeAction = this.connectomePolicy.takeAction(this.simulation);
        if (!connectomeAction) {
          this.connectomePolicy.request(this.simulation);
          this.accumulatedTime = Math.min(this.accumulatedTime, fixedStepMilliseconds);
          break;
        }
        action = connectomeAction;
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
        break;
      }
    }

    this.player.x = this.simulation.player_x;
    this.player.y = this.simulation.player_y;
    this.player.rotation = Phaser.Math.Clamp((this.player.x - previousX) / 24, -0.18, 0.18);
    this.drawHands();
    this.drawDebugOverlay();
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
    if (mode === "connectome") {
      this.connectomePolicy.start();
    }
    const label = mode === "idle"
      ? "IDLE [1]"
      : mode === "chase"
        ? "CURRENT CHASE [2]"
        : mode === "predictive"
          ? "PREDICTIVE [3]"
          : mode === "learned"
            ? "LEARNED GRU [4]"
            : "FULL FLY BRAIN · LOADING [5]";
    this.brainText.setText(`CONTROLLER: ${label}`);
    this.syncConnectomeUi();
  }

  private browserSnapshot(): BrowserSnapshot {
    return {
      tick: this.simulation.tick,
      roundStatus: this.simulation.round_status,
      playerX: this.simulation.player_x,
      playerY: this.simulation.player_y,
      policyMode: this.policyMode,
      pointerActive: this.pointerActive,
      debugEnabled: this.debugEnabled,
      connectome: this.connectomePolicy.snapshot(),
    };
  }

  private syncConnectomeUi(): void {
    const connectome = this.connectomePolicy.snapshot();
    if (connectome.status === "idle") {
      this.connectomeStatusElement.textContent =
        "Full connectome (WebGPU): press 5 to load · 148 MiB local / about 71 MiB compressed";
    } else if (connectome.status === "loading") {
      const loaded = connectome.progress?.loadedBytes ?? 0;
      const total = connectome.progress?.totalBytes ?? 0;
      const percent = total > 0 ? Math.floor((loaded / total) * 100) : 0;
      const detail = connectome.progress?.detail ?? "starting";
      this.connectomeStatusElement.textContent =
        `Full connectome: loading ${percent}% · verifying ${detail}`;
      if (this.policyMode === "connectome") {
        this.brainText.setText(`CONTROLLER: FULL FLY BRAIN · LOADING ${percent}% [5]`);
      }
    } else if (connectome.status === "ready" && connectome.info) {
      const shortHash = connectome.info.packageSha256.slice(0, 8);
      this.connectomeStatusElement.textContent =
        `Full connectome: ready · WebGPU u16 · ${connectome.info.adapter} · v${connectome.info.formatVersion} ${shortHash}`;
      if (this.policyMode === "connectome") {
        this.brainText.setText("CONTROLLER: FULL FLY BRAIN [5]");
      }
    } else {
      this.connectomeStatusElement.textContent =
        `Full connectome unavailable: ${connectome.error ?? "unknown error"} · using GRU [4]`;
    }
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
    const body = this.add.graphics();
    body.fillStyle(0x799e91, 0.56).fillEllipse(300, 260, 300, 380);
    body.fillStyle(0x799e91, 0.56).fillEllipse(660, 260, 300, 380);
    body.lineStyle(7, 0x405a53, 0.55).strokeEllipse(300, 260, 300, 380);
    body.strokeEllipse(660, 260, 300, 380);
    body.fillStyle(0x4b4036).fillEllipse(WIDTH / 2, 265, 410, 410);
    body.lineStyle(10, 0x2d2521).strokeEllipse(WIDTH / 2, 265, 410, 410);

    body.fillStyle(0xb13b34).fillCircle(390, 220, 108);
    body.fillCircle(570, 220, 108);
    body.lineStyle(9, 0x38251f).strokeCircle(390, 220, 108);
    body.strokeCircle(570, 220, 108);
    body.lineStyle(2, 0xf2a46f, 0.55);
    for (let offset = -66; offset <= 66; offset += 22) {
      body.lineBetween(324, 220 + offset, 456, 220 + offset);
      body.lineBetween(504, 220 + offset, 636, 220 + offset);
    }

    body
      .lineStyle(10, 0x2d2521)
      .beginPath()
      .moveTo(430, 355)
      .lineTo(480, 380)
      .lineTo(530, 355)
      .strokePath();
  }

  private drawHands(): void {
    this.hands.clear();
    for (let index = 0; index < 2; index += 1) {
      const palmX = this.simulation.hand_palm_x(index);
      const palmY = this.simulation.hand_palm_y(index);
      const phase = this.simulation.hand_phase(index);
      const activeContact = this.simulation.hand_has_active_contact(index);
      const palmColor = activeContact ? 0xd94b35 : phase === 1 ? 0x9d6749 : 0x5b4639;
      const fingerDirection = index === 0 ? 1 : -1;

      this.hands.fillStyle(palmColor).fillEllipse(palmX, palmY, 64, 76);
      this.hands.lineStyle(5, 0x2d2521).strokeEllipse(palmX, palmY, 64, 76);
      for (const fingerY of [-22, 0, 22]) {
        const fingerX = palmX + fingerDirection * 29;
        this.hands.fillStyle(palmColor).fillCircle(fingerX, palmY + fingerY, 13);
        this.hands.lineStyle(4, 0x2d2521).strokeCircle(fingerX, palmY + fingerY, 13);
      }

      if (phase === 1) {
        this.hands.lineStyle(3, 0xf2cf74, 0.9).strokeCircle(palmX, palmY, 48);
      }
    }

    if (this.simulation.attack_active) {
      const targetX = (
        this.simulation.hand_palm_x(0) + this.simulation.hand_palm_x(1)
      ) / 2;
      const targetY = (
        this.simulation.hand_palm_y(0) + this.simulation.hand_palm_y(1)
      ) / 2;
      this.hands.lineStyle(3, 0xf2cf74, 0.65).strokeCircle(targetX, targetY, 24);
    }
  }

  private drawDebugOverlay(): void {
    this.debugGraphics.clear();
    this.brainActivityGraphics.clear();
    this.brainActivityText.setVisible(false);
    if (!this.debugEnabled) {
      this.debugText.setText("");
      return;
    }

    const playerX = this.simulation.player_x;
    const playerY = this.simulation.player_y;
    const velocityX = this.simulation.player_velocity_x;
    const velocityY = this.simulation.player_velocity_y;
    const targetX = this.latestAction.targetXUnits / UNITS_PER_PIXEL;
    const targetY = this.latestAction.targetYUnits / UNITS_PER_PIXEL;

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
      `target=(${targetX.toFixed(1)}, ${targetY.toFixed(1)}) strike=${Number(this.latestAction.strike)}`,
      ...handLines,
    ]);

    if (this.policyMode === "learned") {
      this.drawBrainActivity();
    }
  }

  private drawBrainActivity(): void {
    const panelX = WIDTH - 166;
    const panelY = 56;
    this.brainActivityGraphics
      .fillStyle(0x181310, 0.78)
      .fillRoundedRect(panelX, panelY, 150, 154, 8)
      .lineStyle(1, 0xfff1cf, 0.35)
      .strokeRoundedRect(panelX, panelY, 150, 154, 8);
    this.brainActivityText.setVisible(true);

    const activity = this.learnedPolicy.activity();
    for (let index = 0; index < activity.length; index += 1) {
      const value = activity[index] ?? 0;
      const column = index % 8;
      const row = Math.floor(index / 8);
      const color = value >= 0 ? 0xffbd59 : 0x5fc9ff;
      const alpha = 0.18 + Math.min(1, Math.abs(value)) * 0.82;
      this.brainActivityGraphics
        .fillStyle(color, alpha)
        .fillCircle(panelX + 22 + column * 16, panelY + 52 + row * 12, 4);
    }
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
