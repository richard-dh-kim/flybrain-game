import type { Simulation } from "./generated/wasm/flybrain_game.js";
import {
  actionFromNormalizedDecision,
  observationFeatures,
} from "./learnedPolicy";
import type {
  ConnectomeGpuDecision,
  ConnectomeGpuInfo,
  ConnectomeGpuProgress,
} from "./connectomeWebGpu";
import type { PolicyAction } from "./scriptedPolicy";

const MODEL_BASE_URL = "/models/connectome-u16-v2";
const TIMING_WINDOW = 120;

export type ConnectomePolicyStatus = "idle" | "loading" | "ready" | "error";

export interface ConnectomePolicySnapshot {
  status: ConnectomePolicyStatus;
  progress: ConnectomeGpuProgress | null;
  info: ConnectomeGpuInfo | null;
  error: string | null;
  pending: boolean;
  pendingSteps: number;
  completedSteps: number;
  inferenceMedianMs: number | null;
  inferenceP95Ms: number | null;
  roundTripP95Ms: number | null;
  activitySampleCount: number;
  activityStep: number;
}

type WorkerResponse =
  | { type: "progress"; progress: ConnectomeGpuProgress }
  | { type: "loaded"; info: ConnectomeGpuInfo }
  | {
    type: "decision";
    requestId: number;
    epoch: number;
    decision: ConnectomeGpuDecision;
    inferenceMs: number;
  }
  | { type: "reset"; epoch: number }
  | { type: "error"; message: string; requestId?: number };

export class ConnectomePolicy {
  private worker: Worker | undefined;
  private status: ConnectomePolicyStatus = "idle";
  private progress: ConnectomeGpuProgress | null = null;
  private info: ConnectomeGpuInfo | null = null;
  private error: string | null = null;
  private pendingRequests = new Map<number, { epoch: number; started: number }>();
  private latestDecision: ConnectomeGpuDecision | null = null;
  private latestDecisionStep = 0;
  private displayedActivity: number[] = [];
  private displayedActivityStep = 0;
  private requestId = 0;
  private epoch = 0;
  private completedSteps = 0;
  private inferenceDurations: number[] = [];
  private roundTripDurations: number[] = [];

  start(): void {
    if (this.status === "loading" || this.status === "ready") {
      return;
    }
    this.worker?.terminate();
    this.status = "loading";
    this.progress = {
      stage: "manifest",
      detail: "starting worker",
      loadedBytes: 0,
      totalBytes: 0,
    };
    this.info = null;
    this.error = null;
    this.worker = new Worker(new URL("./connectomeGpuWorker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>): void => {
      this.handleMessage(event.data);
    };
    this.worker.onerror = (event): void => {
      this.fail(event.message || "connectome worker failed");
    };
    this.worker.postMessage({ type: "load", baseUrl: MODEL_BASE_URL });
  }

  reset(): void {
    this.epoch += 1;
    this.pendingRequests.clear();
    this.latestDecision = null;
    this.latestDecisionStep = 0;
    this.displayedActivity = [];
    this.displayedActivityStep = 0;
    this.completedSteps = 0;
    this.inferenceDurations = [];
    this.roundTripDurations = [];
    if (this.status === "ready") {
      this.worker?.postMessage({ type: "reset", epoch: this.epoch });
    }
  }

  request(simulation: Simulation): boolean {
    if (
      this.status !== "ready"
      || !this.worker
      || this.pendingRequests.size > 0
    ) {
      return false;
    }
    const features = new Float32Array(observationFeatures(simulation));
    const requestId = ++this.requestId;
    this.pendingRequests.set(requestId, { epoch: this.epoch, started: performance.now() });
    this.worker.postMessage(
      { type: "step", requestId, epoch: this.epoch, features },
      [features.buffer],
    );
    return true;
  }

  takeAction(simulation: Simulation): PolicyAction | undefined {
    if (!this.latestDecision) {
      return undefined;
    }
    const decision = this.latestDecision;
    this.latestDecision = null;
    this.displayedActivity = decision.activity;
    this.displayedActivityStep = this.latestDecisionStep;
    return actionFromNormalizedDecision(simulation, decision);
  }

  activity(): ArrayLike<number> {
    return this.displayedActivity;
  }

  snapshot(): ConnectomePolicySnapshot {
    return {
      status: this.status,
      progress: this.progress,
      info: this.info,
      error: this.error,
      pending: this.pendingRequests.size > 0,
      pendingSteps: this.pendingRequests.size,
      completedSteps: this.completedSteps,
      inferenceMedianMs: percentile(this.inferenceDurations, 50),
      inferenceP95Ms: percentile(this.inferenceDurations, 95),
      roundTripP95Ms: percentile(this.roundTripDurations, 95),
      activitySampleCount: this.displayedActivity.length,
      activityStep: this.displayedActivityStep,
    };
  }

  private handleMessage(message: WorkerResponse): void {
    if (message.type === "progress") {
      this.progress = message.progress;
    } else if (message.type === "loaded") {
      this.status = "ready";
      this.info = message.info;
      this.progress = {
        stage: "ready",
        detail: message.info.adapter,
        loadedBytes: this.progress?.totalBytes ?? 0,
        totalBytes: this.progress?.totalBytes ?? 0,
      };
      this.reset();
    } else if (message.type === "decision") {
      const pending = this.pendingRequests.get(message.requestId);
      if (!pending || message.epoch !== this.epoch || pending.epoch !== this.epoch) {
        return;
      }
      this.pendingRequests.delete(message.requestId);
      this.latestDecision = message.decision;
      this.completedSteps += 1;
      this.latestDecisionStep = this.completedSteps;
      recordTiming(this.inferenceDurations, message.inferenceMs);
      recordTiming(this.roundTripDurations, performance.now() - pending.started);
    } else if (message.type === "error") {
      this.fail(message.message);
    }
  }

  private fail(message: string): void {
    this.status = "error";
    this.error = message;
    this.pendingRequests.clear();
    this.latestDecision = null;
    this.latestDecisionStep = 0;
    this.displayedActivity = [];
    this.displayedActivityStep = 0;
    this.worker?.terminate();
    this.worker = undefined;
  }
}

function recordTiming(values: number[], value: number): void {
  values.push(value);
  if (values.length > TIMING_WINDOW) {
    values.shift();
  }
}

function percentile(values: number[], percentage: number): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.ceil(((sorted.length - 1) * percentage) / 100);
  return sorted[index] ?? null;
}
