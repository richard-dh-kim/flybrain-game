export type ScriptedPolicyMode = "idle" | "chase" | "predictive";

export interface PolicyObservation {
  tick: number;
  playerXUnits: number;
  playerYUnits: number;
  playerVelocityXUnits: number;
  playerVelocityYUnits: number;
}

export interface PolicyAction {
  targetXUnits: number;
  targetYUnits: number;
  strike: boolean;
}

const FIRST_STRIKE_TICK = 45;
const STRIKE_INTERVAL_TICKS = 72;
const PREDICTION_TICKS = 18;

export function decideScriptedAction(
  mode: ScriptedPolicyMode,
  observation: PolicyObservation,
): PolicyAction {
  const predictionTicks = mode === "predictive" ? PREDICTION_TICKS : 0;
  return {
    targetXUnits:
      observation.playerXUnits + observation.playerVelocityXUnits * predictionTicks,
    targetYUnits:
      observation.playerYUnits + observation.playerVelocityYUnits * predictionTicks,
    strike:
      mode !== "idle"
      && observation.tick >= FIRST_STRIKE_TICK
      && observation.tick % STRIKE_INTERVAL_TICKS === 0,
  };
}
