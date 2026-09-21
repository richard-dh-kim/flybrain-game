interface BenchmarkState {
  status: "loading" | "measured" | "error";
  detail: unknown;
}

declare global {
  interface Window {
    __flybrainConnectomeBenchmark?: {
      snapshot: () => BenchmarkState;
    };
  }
}

const output = document.querySelector("pre");
if (!output) {
  throw new Error("benchmark output element is missing");
}
let state: BenchmarkState = { status: "loading", detail: "starting worker" };
const worker = new Worker(new URL("./connectomeGpuWorker.ts", import.meta.url), {
  type: "module",
});
worker.onmessage = (event: MessageEvent): void => {
  if (event.data.type === "progress") {
    state = { status: "loading", detail: event.data.progress };
  } else if (event.data.type === "result") {
    state = { status: "measured", detail: event.data.result };
  } else if (event.data.type === "error") {
    state = { status: "error", detail: event.data.message };
  }
  output.textContent = JSON.stringify(state, null, 2);
};
worker.onerror = (event): void => {
  state = { status: "error", detail: event.message };
  output.textContent = JSON.stringify(state, null, 2);
};
window.__flybrainConnectomeBenchmark = { snapshot: () => state };
const parameters = new URLSearchParams(location.search);
const ticks = Number.parseInt(parameters.get("ticks") ?? "120", 10);
worker.postMessage({
  type: "benchmark",
  baseUrl: `${import.meta.env.BASE_URL}models/connectome-u16-v2`,
  ticks,
});
