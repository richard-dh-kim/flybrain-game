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
  "flybrain_game_wasm.wasm",
);
const outputDirectory = join(projectRoot, "apps", "web", "src", "generated", "wasm");

mkdirSync(outputDirectory, { recursive: true });

const cargoEnvironment = { ...process.env };
const localLinker = join(projectRoot, ".tools", "zig-cc");
if (
  process.platform === "linux"
  && !executableIsOnPath("cc")
  && existsSync(localLinker)
) {
  cargoEnvironment.CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = localLinker;
}

run(cargo, [
  "build",
  "--release",
  "-p",
  "flybrain-game-wasm",
  "--target",
  "wasm32-unknown-unknown",
], cargoEnvironment);
run(wasmBindgen, [
  wasmInput,
  "--target",
  "web",
  "--out-dir",
  outputDirectory,
  "--out-name",
  "flybrain_game",
]);

function userToolOrPath(name) {
  const userTool = join(homedir(), ".cargo", "bin", name);
  return existsSync(userTool) ? userTool : name;
}

function executableIsOnPath(name) {
  return (process.env.PATH ?? "")
    .split(delimiter)
    .some((directory) => existsSync(join(directory, name)));
}

function run(command, arguments_, environment = process.env) {
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
