/// <reference lib="webworker" />

import {
  ConnectomeWebGpu,
  type ConnectomeGpuProgress,
} from "./connectomeWebGpu";

interface BenchmarkRequest {
  type: "benchmark";
  baseUrl: string;
  ticks: number;
}

interface LoadRequest {
  type: "load";
  baseUrl: string;
}

interface StepRequest {
  type: "step";
  requestId: number;
  epoch: number;
  features: Float32Array;
}

interface ResetRequest {
  type: "reset";
  epoch: number;
}

interface DestroyRequest {
  type: "destroy";
}

type WorkerRequest = BenchmarkRequest | LoadRequest | StepRequest | ResetRequest | DestroyRequest;

const worker = self as DedicatedWorkerGlobalScope;
let activeModel: ConnectomeWebGpu | undefined;
let commandChain = Promise.resolve();
let newestEpoch = 0;

worker.onmessage = (event: MessageEvent<WorkerRequest>): void => {
  const request = event.data;
  if (request.type === "reset") {
    newestEpoch = Math.max(newestEpoch, request.epoch);
  }
  commandChain = commandChain
    .then(() => handleRequest(request))
    .catch((error: unknown) => {
      worker.postMessage({
        type: "error",
        requestId: request.type === "step" ? request.requestId : undefined,
        message: error instanceof Error ? error.message : String(error),
      });
    });
};

async function handleRequest(request: WorkerRequest): Promise<void> {
  if (request.type === "benchmark") {
    await benchmark(request);
  } else if (request.type === "load") {
    activeModel?.destroy();
    activeModel = await ConnectomeWebGpu.create(
      request.baseUrl,
      (progress: ConnectomeGpuProgress) => worker.postMessage({ type: "progress", progress }),
    );
    worker.postMessage({ type: "loaded", info: activeModel.info });
  } else if (request.type === "step") {
    if (request.epoch !== newestEpoch) {
      return;
    }
    if (!activeModel) {
      throw new Error("connectome model is not loaded");
    }
    const started = performance.now();
    const decision = await activeModel.step(request.features);
    worker.postMessage({
      type: "decision",
      requestId: request.requestId,
      epoch: request.epoch,
      decision,
      inferenceMs: performance.now() - started,
    });
  } else if (request.type === "reset") {
    activeModel?.reset();
    worker.postMessage({ type: "reset", epoch: request.epoch });
  } else {
    activeModel?.destroy();
    activeModel = undefined;
  }
}

async function benchmark(request: BenchmarkRequest): Promise<void> {
  const loadStarted = performance.now();
  const model = await ConnectomeWebGpu.create(
    request.baseUrl,
    (progress: ConnectomeGpuProgress) => worker.postMessage({ type: "progress", progress }),
  );
  const loadMs = performance.now() - loadStarted;
  const features = new Float32Array(model.featureCount);
  let generator = 20_260_920;
  const durations: number[] = [];
  let finalDecision = undefined;
  for (let tick = 0; tick < request.ticks; tick += 1) {
    for (let index = 0; index < features.length; index += 1) {
      generator ^= generator << 13;
      generator ^= generator >>> 17;
      generator ^= generator << 5;
      generator >>>= 0;
      features[index] = ((generator >>> 16) / 65_535) * 2 - 1;
    }
    const started = performance.now();
    finalDecision = await model.step(features);
    durations.push(performance.now() - started);
  }
  durations.sort((left, right) => left - right);
  worker.postMessage({
    type: "result",
    result: {
      status: "measured",
      ...model.info,
      ticks: request.ticks,
      loadMs,
      inferenceMedianMs: percentile(durations, 50, 100),
      inferenceP95Ms: percentile(durations, 95, 100),
      inferenceMaximumMs: durations.at(-1),
      within60HzBudget: percentile(durations, 95, 100) <= 1_000 / 60,
      finalDecision,
    },
  });
  model.destroy();
}

function percentile(sorted: number[], numerator: number, denominator: number): number {
  const index = Math.ceil(((sorted.length - 1) * numerator) / denominator);
  const value = sorted[index];
  if (value === undefined) {
    throw new Error("cannot take a percentile of an empty sample");
  }
  return value;
}
