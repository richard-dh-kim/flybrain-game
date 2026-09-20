import { cpSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(root, process.argv[2] ?? "artifacts/connectome-packed-u16-v2");
const destination = resolve(root, "apps/web/public/models/connectome-u16-v2");
if (!existsSync(resolve(source, "manifest.json"))) {
  throw new Error(`packed model is missing: ${source}`);
}
const manifest = JSON.parse(readFileSync(resolve(source, "manifest.json"), "utf8"));
if (manifest.format_version !== 2 || manifest.quantization?.weight_bits !== 16) {
  throw new Error("web assets require the validated packed-v2 u16 model");
}
mkdirSync(destination, { recursive: true });
cpSync(source, destination, { recursive: true, force: true });
console.log(JSON.stringify({
  source,
  destination,
  packageSha256: manifest.package_sha256,
}, null, 2));
