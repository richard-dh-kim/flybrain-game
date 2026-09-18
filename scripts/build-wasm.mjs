import { mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDirectory, "..");
const executableSuffix = process.platform === "win32" ? ".exe" : "";
const cargo = process.platform === "win32"
  ? join(homedir(), ".cargo", "bin", `cargo${executableSuffix}`)
  : "cargo";
const wasmBindgen = process.platform === "win32"
  ? join(homedir(), ".cargo", "bin", `wasm-bindgen${executableSuffix}`)
  : "wasm-bindgen";
const wasmInput = join(
  projectRoot,
  "target",
  "wasm32-unknown-unknown",
  "release",
  "flybrain_game_wasm.wasm",
);
const outputDirectory = join(projectRoot, "apps", "web", "src", "generated", "wasm");

mkdirSync(outputDirectory, { recursive: true });

run(cargo, [
  "build",
  "--release",
  "-p",
  "flybrain-game-wasm",
  "--target",
  "wasm32-unknown-unknown",
]);
run(wasmBindgen, [
  wasmInput,
  "--target",
  "web",
  "--out-dir",
  outputDirectory,
  "--out-name",
  "flybrain_game",
]);

function run(command, arguments_) {
  const result = spawnSync(command, arguments_, {
    cwd: projectRoot,
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
