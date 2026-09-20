import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = fileURLToPath(new URL("../", import.meta.url));
const runtimeUrl = new URL("../apps/web/src/gruRuntime.ts", import.meta.url);
const modelUrl = new URL("../apps/web/src/generated/gru-v1.json", import.meta.url);
const fixtureUrl = new URL("../tests/replays/gru-parity-v1.json", import.meta.url);

const [model, fixture, GruRuntime] = await Promise.all([
  readFile(modelUrl, "utf8").then(JSON.parse),
  readFile(fixtureUrl, "utf8").then(JSON.parse),
  loadRuntime(),
]);

if (model.checkpointSha256 !== fixture.checkpointSha256) {
  throw new Error("browser model and PyTorch parity fixture use different checkpoints");
}

const runtime = new GruRuntime(model);
let maximumDifference = 0;
for (const frame of fixture.frames) {
  const result = runtime.step(frame.features);
  maximumDifference = Math.max(
    maximumDifference,
    maximumAbsoluteDifference(result.output, frame.output),
    maximumAbsoluteDifference(result.activity, frame.activity),
  );
  if (maximumDifference > fixture.absoluteTolerance) {
    throw new Error(
      `GRU parity exceeded ${fixture.absoluteTolerance} at tick ${frame.tick}: ${maximumDifference}`,
    );
  }
}

console.log(
  `GRU parity passed for ${fixture.frames.length} recurrent ticks; max |difference|=${maximumDifference.toExponential(3)}.`,
);

async function loadRuntime() {
  const temporaryDirectory = await mkdtemp(join(tmpdir(), "flybrain-gru-runtime-"));
  try {
    const compiler = join(projectRoot, "node_modules", ".bin", "tsc");
    const result = spawnSync(
      compiler,
      [
        fileURLToPath(runtimeUrl),
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
      throw new Error(`could not compile GRU runtime: ${result.stderr || result.stdout}`);
    }
    const source = await readFile(join(temporaryDirectory, "gruRuntime.js"), "utf8");
    const encoded = Buffer.from(source).toString("base64");
    return (await import(`data:text/javascript;base64,${encoded}`)).GruRuntime;
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
}

function maximumAbsoluteDifference(first, second) {
  if (first.length !== second.length) {
    throw new Error(`vector length mismatch: ${first.length} != ${second.length}`);
  }
  let maximum = 0;
  for (let index = 0; index < first.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(first[index] - second[index]));
  }
  return maximum;
}
