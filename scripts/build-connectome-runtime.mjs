import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDirectory, "..");
const executableSuffix = process.platform === "win32" ? ".exe" : "";
const cargo = userToolOrPath(`cargo${executableSuffix}`);
const environment = { ...process.env };
const localLinker = join(projectRoot, ".tools", "zig-cc");
if (
  process.platform === "linux"
  && !executableIsOnPath("cc")
  && existsSync(localLinker)
) {
  environment.CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = localLinker;
}

const result = spawnSync(cargo, [
  "build",
  "--release",
  "-p",
  "flybrain-connectome-runtime",
  "--example",
  "benchmark",
], {
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

function userToolOrPath(name) {
  const userTool = join(homedir(), ".cargo", "bin", name);
  return existsSync(userTool) ? userTool : name;
}

function executableIsOnPath(name) {
  return (process.env.PATH ?? "")
    .split(delimiter)
    .some((directory) => existsSync(join(directory, name)));
}
