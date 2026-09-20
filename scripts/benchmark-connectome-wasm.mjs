import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { performance } from "node:perf_hooks";

const packageDirectory = resolve(process.argv[2] ?? "artifacts/connectome-packed-u16-v2");
const ticks = Number.parseInt(process.argv[3] ?? "300", 10);
if (!Number.isSafeInteger(ticks) || ticks <= 0) {
  throw new Error("ticks must be a positive integer");
}
const require = createRequire(import.meta.url);
const { QuantizedConnectome } = require("../.tools/connectome-wasm-node/flybrain_connectome.js");

const diskStarted = performance.now();
const rowOffsets = typed("row_offsets.u32.bin", Uint32Array);
const columnIndices = typed("column_indices.u32.bin", Uint32Array);
const edgeWeights = typed("edge_weights.u16.bin", Uint16Array);
const edgeScales = typed("edge_scales.f32.bin", Float32Array);
const leak = typed("leak.f32.bin", Float32Array);
const sensoryIndices = typed("sensory_indices.u32.bin", Uint32Array);
const sensoryFeatureIds = typed("sensory_feature_ids.u8.bin", Uint8Array);
const sensorySigns = typed("sensory_signs.i8.bin", Int8Array);
const motorIndices = typed("motor_indices.u32.bin", Uint32Array);
const readoutWeight = typed("readout_weight.f32.bin", Float32Array);
const readoutBias = typed("readout_bias.f32.bin", Float32Array);
const diskLoadMs = performance.now() - diskStarted;

const constructStarted = performance.now();
const model = new QuantizedConnectome(
  rowOffsets,
  columnIndices,
  edgeWeights,
  edgeScales,
  leak,
  sensoryIndices,
  sensoryFeatureIds,
  sensorySigns,
  motorIndices,
  readoutWeight,
  readoutBias,
  33,
  0.5,
);
const constructMs = performance.now() - constructStarted;

let generator = 20_260_920;
const features = new Float32Array(33);
const durations = [];
let decision;
for (let tick = 0; tick < ticks; tick += 1) {
  for (let index = 0; index < features.length; index += 1) {
    generator ^= generator << 13;
    generator ^= generator >>> 17;
    generator ^= generator << 5;
    generator >>>= 0;
    features[index] = ((generator >>> 16) / 65_535) * 2 - 1;
  }
  const started = performance.now();
  decision = model.step(features);
  durations.push(performance.now() - started);
}
durations.sort((left, right) => left - right);
const state = model.state_copy();
console.log(JSON.stringify({
  status: "measured",
  backend: "node-v8-wasm-single-thread",
  weightEncoding: "rowwise-u16",
  neurons: model.neuron_count,
  edges: model.edge_count,
  ticks,
  diskLoadMs,
  wasmConstructMs: constructMs,
  inferenceMedianMs: percentile(durations, 50, 100),
  inferenceP95Ms: percentile(durations, 95, 100),
  inferenceMaximumMs: durations.at(-1),
  within60HzBudget: percentile(durations, 95, 100) <= 1_000 / 60,
  finalTarget: [decision[0], decision[1]],
  finalStrikeLogit: decision[2],
  finalStateNonzero: state.reduce((count, value) => count + Number(value !== 0), 0),
  processRssMiB: process.memoryUsage().rss / (1024 * 1024),
}, null, 2));

function typed(filename, Constructor) {
  const bytes = readFileSync(resolve(packageDirectory, filename));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Constructor(buffer);
}

function percentile(sorted, numerator, denominator) {
  const index = Math.ceil(((sorted.length - 1) * numerator) / denominator);
  return sorted[index];
}
