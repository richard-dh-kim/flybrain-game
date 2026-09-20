import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const projectRoot = fileURLToPath(new URL("..", import.meta.url));
const generatedRoot = new URL("../apps/web/src/generated/wasm/", import.meta.url);
const movementReplayUrl = new URL("../tests/replays/player-movement-v1.csv", import.meta.url);
const actionReplayUrl = new URL("../tests/replays/action-round-v1.csv", import.meta.url);
const moduleUrl = new URL("flybrain_game.js", generatedRoot);
const wasmUrl = new URL("flybrain_game_bg.wasm", generatedRoot);

const [{ default: initWasm, Simulation }, wasmBytes, movementReplay, actionReplay] = await Promise.all([
  import(moduleUrl.href),
  readFile(wasmUrl),
  readFile(movementReplayUrl, "utf8"),
  readFile(actionReplayUrl, "utf8"),
]);

await initWasm({ module_or_path: wasmBytes });
runMovementReplay(Simulation, movementReplay);
const actionTicks = runActionReplay(Simulation, actionReplay);
console.log(`WASM replay parity passed for movement and ${actionTicks} action ticks from ${projectRoot}`);

function runMovementReplay(SimulationClass, replayText) {
  const simulation = new SimulationClass();

  for (const [index, line] of replayText.split("\n").entries()) {
    if (line.length === 0 || line.startsWith("#")) {
      continue;
    }

    const [
      horizontal,
      vertical,
      ticks,
      expectedTick,
      positionX,
      positionY,
      velocityX,
      velocityY,
    ] = line.split(",").map(Number);

    for (let tick = 0; tick < ticks; tick += 1) {
      simulation.step(horizontal, vertical);
    }

    assertEqual(simulation.tick, expectedTick, "tick", index + 1);
    assertEqual(simulation.player_x_units, positionX, "position x", index + 1);
    assertEqual(simulation.player_y_units, positionY, "position y", index + 1);
    assertEqual(simulation.player_velocity_x_units, velocityX, "velocity x", index + 1);
    assertEqual(simulation.player_velocity_y_units, velocityY, "velocity y", index + 1);
  }

  simulation.free();
}

function runActionReplay(SimulationClass, replayText) {
  const simulation = new SimulationClass();
  let currentEpisode;
  let replayTicks = 0;

  for (const [index, line] of replayText.split("\n").entries()) {
    if (line.length === 0 || line.startsWith("#")) {
      continue;
    }

    const values = line.split(",").map(Number);
    assertEqual(values.length, 26, "column count", index + 1);
    const episode = values[0];
    if (currentEpisode !== episode) {
      if (currentEpisode !== undefined) {
        assertEqual(episode, currentEpisode + 1, "episode sequence", index + 1);
        assertEqual(simulation.round_status, 2, "survival episode status", index + 1);
        simulation.restart();
      }
      currentEpisode = episode;
    }

    simulation.step_with_action(values[2], values[3], values[4], values[5], values[6] === 1);
    replayTicks += 1;
    assertEqual(simulation.tick, values[1], "tick", index + 1);
    assertEqual(simulation.player_x_units, values[7], "player x", index + 1);
    assertEqual(simulation.player_y_units, values[8], "player y", index + 1);
    assertEqual(simulation.player_velocity_x_units, values[9], "velocity x", index + 1);
    assertEqual(simulation.player_velocity_y_units, values[10], "velocity y", index + 1);
    assertEqual(simulation.round_status, values[11], "round status", index + 1);
    assertEqual(simulation.round_elapsed_ticks, values[12], "round elapsed ticks", index + 1);
    assertEqual(Number(simulation.attack_active), values[13], "attack active", index + 1);

    for (const [handIndex, offset] of [[0, 14], [1, 20]]) {
      assertEqual(simulation.hand_phase(handIndex), values[offset], `hand ${handIndex} phase`, index + 1);
      assertEqual(simulation.hand_phase_tick(handIndex), values[offset + 1], `hand ${handIndex} phase tick`, index + 1);
      assertEqual(simulation.hand_cooldown(handIndex), values[offset + 2], `hand ${handIndex} cooldown`, index + 1);
      assertEqual(simulation.hand_palm_x_units(handIndex), values[offset + 3], `hand ${handIndex} palm x`, index + 1);
      assertEqual(simulation.hand_palm_y_units(handIndex), values[offset + 4], `hand ${handIndex} palm y`, index + 1);
      assertEqual(Number(simulation.hand_has_active_contact(handIndex)), values[offset + 5], `hand ${handIndex} contact`, index + 1);
    }
  }

  assertEqual(replayTicks, 1868, "action replay tick count", 0);
  assertEqual(currentEpisode, 1, "final episode", 0);
  assertEqual(simulation.round_status, 1, "hit episode status", 0);
  simulation.free();
  return replayTicks;
}

function assertEqual(actual, expected, field, row) {
  if (actual !== expected) {
    throw new Error(`${field} mismatch at replay row ${row}: ${actual} !== ${expected}`);
  }
}
