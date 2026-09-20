import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDirectory, "..");
const executableSuffix = process.platform === "win32" ? ".exe" : "";
const cargo = userToolOrPath(`cargo${executableSuffix}`);
const wasmBindgen = userToolOrPath(`wasm-bindgen${executableSuffix}`);
const wasmInput = join(
  projectRoot,
  "target",
  "wasm32-unknown-unknown",
  "release",
  "flybrain_connectome_wasm.wasm",
);
const outputDirectory = join(projectRoot, ".tools", "connectome-wasm-node");
mkdirSync(outputDirectory, { recursive: true });
const environment = { ...process.env };
const localCompiler = join(projectRoot, ".tools", "zig-cc");
if (
  process.platform === "linux"
  && !executableIsOnPath("cc")
  && existsSync(localCompiler)
) {
  environment.CC = localCompiler;
  environment.AR = join(projectRoot, ".tools", "zig-ar");
  environment.CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = localCompiler;
}

run(cargo, [
  "build",
  "--release",
  "-p",
  "flybrain-connectome-wasm",
  "--target",
  "wasm32-unknown-unknown",
]);
run(wasmBindgen, [
  wasmInput,
  "--target",
  "nodejs",
  "--out-dir",
  outputDirectory,
  "--out-name",
  "flybrain_connectome",
]);

function userToolOrPath(name) {
  const userTool = join(homedir(), ".cargo", "bin", name);
  return existsSync(userTool) ? userTool : name;
}

function run(command, arguments_) {
  const result = spawnSync(command, arguments_, {
    cwd: projectRoot,
    env: environment,
    shell: false,
    stdio: "inherit",
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function executableIsOnPath(name) {
  return (process.env.PATH ?? "")
    .split(delimiter)
    .some((directory) => existsSync(join(directory, name)));
}
