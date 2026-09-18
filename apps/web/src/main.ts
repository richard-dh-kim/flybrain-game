import Phaser from "phaser";
import initWasm, { Simulation } from "./generated/wasm/flybrain_game.js";
import "./style.css";

const WIDTH = 960;
const HEIGHT = 540;
const TABLE_TOP = 438;

class GrayboxScene extends Phaser.Scene {
  private player!: Phaser.GameObjects.Container;
  private simulation!: Simulation;
  private keys!: Record<"up" | "down" | "left" | "right", Phaser.Input.Keyboard.Key>;
  private cursors!: Phaser.Types.Input.Keyboard.CursorKeys;
  private accumulatedTime = 0;

  create(): void {
    this.cameras.main.setBackgroundColor("#dca982");
    this.drawRoom();
    this.drawFly();
    this.drawHands();
    this.simulation = new Simulation();
    this.player = this.createPlayer(this.simulation.player_x, this.simulation.player_y);

    const keyboard = this.input.keyboard;
    if (!keyboard) {
      throw new Error("Keyboard input is unavailable in this browser.");
    }

    this.cursors = keyboard.createCursorKeys();
    this.keys = {
      up: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.W),
      down: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.S),
      left: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.A),
      right: keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.D),
    };

    this.add
      .text(24, 22, "SURVIVE THE SWAT", {
        color: "#38251f",
        fontFamily: "Georgia, serif",
        fontSize: "22px",
        fontStyle: "bold",
      })
      .setDepth(20);

    this.add
      .text(WIDTH - 24, 24, "BRAIN: SCRIPTED PLACEHOLDER", {
        color: "#6b4033",
        fontFamily: "monospace",
        fontSize: "13px",
      })
      .setOrigin(1, 0)
      .setDepth(20);
  }

  update(_time: number, delta: number): void {
    const horizontal = Number(this.keys.right.isDown || this.cursors.right.isDown)
      - Number(this.keys.left.isDown || this.cursors.left.isDown);
    const vertical = Number(this.keys.down.isDown || this.cursors.down.isDown)
      - Number(this.keys.up.isDown || this.cursors.up.isDown);
    const previousX = this.simulation.player_x;
    const fixedStepMilliseconds = 1_000 / 60;
    this.accumulatedTime = Math.min(this.accumulatedTime + delta, 250);

    while (this.accumulatedTime >= fixedStepMilliseconds) {
      this.simulation.step(horizontal, vertical);
      this.accumulatedTime -= fixedStepMilliseconds;
    }

    this.player.x = this.simulation.player_x;
    this.player.y = Math.min(this.simulation.player_y, TABLE_TOP - 32);
    this.player.rotation = Phaser.Math.Clamp((this.player.x - previousX) / 24, -0.18, 0.18);
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
    body.fillStyle(0x4b4036).fillEllipse(WIDTH / 2, 286, 250, 280);
    body.lineStyle(8, 0x2d2521).strokeEllipse(WIDTH / 2, 286, 250, 280);
    body.fillStyle(0x799e91, 0.56).fillEllipse(352, 276, 180, 245);
    body.fillStyle(0x799e91, 0.56).fillEllipse(608, 276, 180, 245);
    body.lineStyle(6, 0x405a53, 0.55).strokeEllipse(352, 276, 180, 245);
    body.strokeEllipse(608, 276, 180, 245);

    body.fillStyle(0xb13b34).fillCircle(423, 244, 73);
    body.fillCircle(537, 244, 73);
    body.lineStyle(7, 0x38251f).strokeCircle(423, 244, 73);
    body.strokeCircle(537, 244, 73);
    body.lineStyle(2, 0xf2a46f, 0.55);
    for (let offset = -44; offset <= 44; offset += 22) {
      body.lineBetween(379, 244 + offset, 467, 244 + offset);
      body.lineBetween(493, 244 + offset, 581, 244 + offset);
    }

    body.lineStyle(8, 0x2d2521).beginPath().moveTo(447, 337).lineTo(480, 354).lineTo(513, 337).strokePath();
  }

  private drawHands(): void {
    const hands = this.add.graphics();
    hands.lineStyle(25, 0x43372f, 1);
    hands.beginPath().moveTo(375, 355).lineTo(286, 390).lineTo(210, 344).strokePath();
    hands.beginPath().moveTo(585, 355).lineTo(674, 390).lineTo(750, 344).strokePath();
    hands.fillStyle(0x43372f).fillCircle(203, 339, 30).fillCircle(757, 339, 30);
    hands.lineStyle(5, 0x2d2521).strokeCircle(203, 339, 30).strokeCircle(757, 339, 30);
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
