import type { Simulation } from "./generated/wasm/flybrain_game.js";
import model from "./generated/gru-v1.json";
import { GruRuntime, sigmoid } from "./gruRuntime";
import type { PolicyAction } from "./scriptedPolicy";

const UNITS_PER_PIXEL = 1_024;
const FLIGHT_MIN_X_UNITS = 32 * UNITS_PER_PIXEL;
const FLIGHT_MAX_X_UNITS = (960 - 32) * UNITS_PER_PIXEL;
const FLIGHT_MIN_Y_UNITS = 32 * UNITS_PER_PIXEL;
const FLIGHT_MAX_Y_UNITS = (438 - 32) * UNITS_PER_PIXEL;
const ROUND_TICKS = 1_800;
const PLAYER_SPEED_UNITS = 5 * UNITS_PER_PIXEL;
const HAND_SPEED_SCALE_UNITS = 32 * UNITS_PER_PIXEL;
const ACCELERATION_SCALE_UNITS = 2 * PLAYER_SPEED_UNITS;
const PHASE_TICK_SCALE = 36;
const COOLDOWN_SCALE = 48;

export class LearnedGruPolicy {
  private runtime = new GruRuntime(model);

  reset(): void {
    this.runtime.reset();
  }

  activity(): ArrayLike<number> {
    return this.runtime.activity();
  }

  decide(simulation: Simulation): PolicyAction {
    const features = observationFeatures(simulation);
    const { output } = this.runtime.step(features);
    const targetX = denormalize(sigmoid(value(output, 0)), FLIGHT_MIN_X_UNITS, FLIGHT_MAX_X_UNITS);
    const targetY = denormalize(sigmoid(value(output, 1)), FLIGHT_MIN_Y_UNITS, FLIGHT_MAX_Y_UNITS);
    const strikeProbability = sigmoid(value(output, 2));
    const strikeReady = !simulation.attack_active
      && simulation.hand_phase(0) === 0
      && simulation.hand_phase(1) === 0
      && simulation.hand_cooldown(0) === 0
      && simulation.hand_cooldown(1) === 0;
    return {
      targetXUnits: targetX,
      targetYUnits: targetY,
      strike: strikeReady && strikeProbability >= model.strikeThreshold,
    };
  }
}

function observationFeatures(simulation: Simulation): number[] {
  const features = [
    simulation.tick / ROUND_TICKS,
    normalize(simulation.player_x_units, FLIGHT_MIN_X_UNITS, FLIGHT_MAX_X_UNITS),
    normalize(simulation.player_y_units, FLIGHT_MIN_Y_UNITS, FLIGHT_MAX_Y_UNITS),
    simulation.player_velocity_x_units / PLAYER_SPEED_UNITS,
    simulation.player_velocity_y_units / PLAYER_SPEED_UNITS,
    simulation.player_acceleration_x_units / ACCELERATION_SCALE_UNITS,
    simulation.player_acceleration_y_units / ACCELERATION_SCALE_UNITS,
  ];
  for (let hand = 0; hand < 2; hand += 1) {
    const phase = simulation.hand_phase(hand);
    features.push(
      normalize(simulation.hand_palm_x_units(hand), FLIGHT_MIN_X_UNITS, FLIGHT_MAX_X_UNITS),
      normalize(simulation.hand_palm_y_units(hand), FLIGHT_MIN_Y_UNITS, FLIGHT_MAX_Y_UNITS),
      simulation.hand_palm_velocity_x_units(hand) / HAND_SPEED_SCALE_UNITS,
      simulation.hand_palm_velocity_y_units(hand) / HAND_SPEED_SCALE_UNITS,
      ...[0, 1, 2, 3, 4].map((candidate) => Number(phase === candidate)),
      simulation.hand_phase_tick(hand) / PHASE_TICK_SCALE,
      simulation.hand_cooldown(hand) / COOLDOWN_SCALE,
      Number(simulation.hand_has_active_contact(hand)),
    );
  }
  features.push(
    Number(simulation.attack_active),
    simulation.round_remaining_ticks / ROUND_TICKS,
  );
  if (features.length !== model.inputSize || features.length !== model.featureNames.length) {
    throw new Error(`GRU feature mismatch: ${features.length} != ${model.inputSize}`);
  }
  return features;
}

function normalize(valueToNormalize: number, minimum: number, maximum: number): number {
  return (valueToNormalize - minimum) / (maximum - minimum);
}

function denormalize(valueToDenormalize: number, minimum: number, maximum: number): number {
  const clamped = Math.min(1, Math.max(0, valueToDenormalize));
  return Math.round(minimum + clamped * (maximum - minimum));
}

function value(values: ArrayLike<number>, index: number): number {
  const result = values[index];
  if (result === undefined) {
    throw new Error(`missing model value at index ${index}`);
  }
  return result;
}
