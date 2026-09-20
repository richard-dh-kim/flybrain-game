import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const generatedRoot = new URL("../apps/web/src/generated/wasm/", import.meta.url);
const moduleUrl = new URL("flybrain_game.js", generatedRoot);
const wasmUrl = new URL("flybrain_game_bg.wasm", generatedRoot);
const policyUrl = new URL("../apps/web/src/scriptedPolicy.ts", import.meta.url);

const [{ default: initWasm, Simulation }, wasmBytes, decideScriptedAction] = await Promise.all([
  import(moduleUrl.href),
  readFile(wasmUrl),
  loadPolicy(),
]);
await initWasm({ module_or_path: wasmBytes });

const directionSets = [
  {
    name: "cardinal",
    directions: [[1, 0], [0, 1], [-1, 0], [0, -1]],
  },
  {
    name: "diagonal",
    directions: [[1, 1], [-1, 1], [-1, -1], [1, -1]],
  },
];
const turnPeriods = [120, 180, 240];
const results = [];

for (const directionSet of directionSets) {
  for (const turnPeriod of turnPeriods) {
    for (let phase = 0; phase < 6; phase += 1) {
      const offset = (turnPeriod / 6) * phase;
      const scenario = { ...directionSet, turnPeriod, offset };
      const chaseTicks = runScenario(Simulation, decideScriptedAction, "chase", scenario);
      const predictiveTicks = runScenario(
        Simulation,
        decideScriptedAction,
        "predictive",
        scenario,
      );
      results.push({
        name: `${directionSet.name}-${turnPeriod}-${offset}`,
        chaseTicks,
        predictiveTicks,
      });
    }
  }
}

const chaseAverage = average(results.map((result) => result.chaseTicks));
const predictiveAverage = average(results.map((result) => result.predictiveTicks));
const predictiveWins = results.filter(
  (result) => result.predictiveTicks < result.chaseTicks,
).length;
const ties = results.filter(
  (result) => result.predictiveTicks === result.chaseTicks,
).length;
const predictiveLosses = results.length - predictiveWins - ties;
const ratio = predictiveAverage / chaseAverage;
const summary = [
  `Policy comparison across ${results.length} turning scenarios.`,
  `Predictive wins: ${predictiveWins}/${results.length}.`,
  `Ties: ${ties}; losses: ${predictiveLosses}.`,
  `Mean time to hit: chase=${chaseAverage.toFixed(1)} ticks, predictive=${predictiveAverage.toFixed(1)} ticks.`,
  `Predictive/chase ratio: ${ratio.toFixed(3)}.`,
].join(" ");
console.log(summary);

if (predictiveLosses > results.length / 6) {
  throw new Error(
    `predictive policy lost ${predictiveLosses}/${results.length} turning scenarios`,
  );
}
if (ratio >= 0.6) {
  throw new Error(
    `predictive mean time-to-hit ratio ${ratio.toFixed(3)} did not beat the 0.600 gate`,
  );
}
console.log("Policy gate passed.");

async function loadPolicy() {
  const temporaryDirectory = await mkdtemp(join(tmpdir(), "flybrain-policy-"));
  try {
    const compiler = fileURLToPath(new URL("../node_modules/.bin/tsc", import.meta.url));
    const result = spawnSync(
      compiler,
      [
        fileURLToPath(policyUrl),
        "--target",
        "ES2022",
        "--module",
        "ES2022",
        "--outDir",
        temporaryDirectory,
        "--skipLibCheck",
      ],
      { encoding: "utf8" },
    );
    if (result.status !== 0) {
      throw new Error(`could not compile scripted policy: ${result.stderr || result.stdout}`);
    }
    const source = await readFile(join(temporaryDirectory, "scriptedPolicy.js"), "utf8");
    const encoded = Buffer.from(source).toString("base64");
    const policyModule = await import(`data:text/javascript;base64,${encoded}`);
    return policyModule.decideScriptedAction;
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
}

function runScenario(SimulationClass, policy, mode, scenario) {
  const simulation = new SimulationClass();

  while (simulation.round_status === 0 && simulation.tick < 1800) {
    const directionIndex = Math.floor(
      ((simulation.tick + scenario.offset) / scenario.turnPeriod)
      % scenario.directions.length,
    );
    const [horizontal, vertical] = scenario.directions[directionIndex];
    const action = policy(mode, {
      tick: simulation.tick,
      playerXUnits: simulation.player_x_units,
      playerYUnits: simulation.player_y_units,
      playerVelocityXUnits: simulation.player_velocity_x_units,
      playerVelocityYUnits: simulation.player_velocity_y_units,
    });
    simulation.step_with_action(
      horizontal,
      vertical,
      action.targetXUnits,
      action.targetYUnits,
      action.strike,
    );
  }

  const ticks = simulation.tick;
  const status = simulation.round_status;
  simulation.free();
  if (status !== 1) {
    throw new Error(`${mode} did not hit the player in scenario ${scenario.name}`);
  }
  return ticks;
}

function average(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}
