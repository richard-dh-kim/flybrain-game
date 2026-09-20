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

export type ConnectomePolicyStatus = "idle" | "loading" | "ready" | "error";

export interface ConnectomePolicySnapshot {
  status: ConnectomePolicyStatus;
  progress: ConnectomeGpuProgress | null;
  info: ConnectomeGpuInfo | null;
  error: string | null;
  pending: boolean;
  completedSteps: number;
}

type WorkerResponse =
  | { type: "progress"; progress: ConnectomeGpuProgress }
  | { type: "loaded"; info: ConnectomeGpuInfo }
  | {
    type: "decision";
    requestId: number;
    epoch: number;
    decision: ConnectomeGpuDecision;
  }
  | { type: "reset"; epoch: number }
  | { type: "error"; message: string; requestId?: number };

export class ConnectomePolicy {
  private worker: Worker | undefined;
  private status: ConnectomePolicyStatus = "idle";
  private progress: ConnectomeGpuProgress | null = null;
  private info: ConnectomeGpuInfo | null = null;
  private error: string | null = null;
  private pendingRequestId: number | null = null;
  private completedDecision: ConnectomeGpuDecision | null = null;
  private requestId = 0;
  private epoch = 0;
  private completedSteps = 0;

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
    this.pendingRequestId = null;
    this.completedDecision = null;
    this.completedSteps = 0;
    if (this.status === "ready") {
      this.worker?.postMessage({ type: "reset", epoch: this.epoch });
    }
  }

  request(simulation: Simulation): boolean {
    if (
      this.status !== "ready"
      || !this.worker
      || this.pendingRequestId !== null
      || this.completedDecision !== null
    ) {
      return false;
    }
    const features = new Float32Array(observationFeatures(simulation));
    const requestId = ++this.requestId;
    this.pendingRequestId = requestId;
    this.worker.postMessage(
      { type: "step", requestId, epoch: this.epoch, features },
      [features.buffer],
    );
    return true;
  }

  takeAction(simulation: Simulation): PolicyAction | undefined {
    if (!this.completedDecision) {
      return undefined;
    }
    const decision = this.completedDecision;
    this.completedDecision = null;
    return actionFromNormalizedDecision(simulation, decision);
  }

  snapshot(): ConnectomePolicySnapshot {
    return {
      status: this.status,
      progress: this.progress,
      info: this.info,
      error: this.error,
      pending: this.pendingRequestId !== null,
      completedSteps: this.completedSteps,
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
      if (message.epoch !== this.epoch || message.requestId !== this.pendingRequestId) {
        return;
      }
      this.pendingRequestId = null;
      this.completedDecision = message.decision;
      this.completedSteps += 1;
    } else if (message.type === "error") {
      this.fail(message.message);
    }
  }

  private fail(message: string): void {
    this.status = "error";
    this.error = message;
    this.pendingRequestId = null;
    this.completedDecision = null;
    this.worker?.terminate();
    this.worker = undefined;
  }
}
