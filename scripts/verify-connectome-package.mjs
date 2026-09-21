import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { basename, dirname, resolve, sep } from "node:path";

const packageDirectory = resolve(process.argv[2] ?? "artifacts/connectome-packed-u16-v2");
const expectedPackageSha256 = process.argv[3];
const manifestPath = resolve(packageDirectory, "manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));

if (
  manifest.format !== "flybrain-connectome-packed"
  || manifest.format_version !== 2
  || typeof manifest.arrays !== "object"
  || manifest.arrays === null
) {
  throw new Error("packed model manifest format is unsupported or incomplete");
}

const manifestPayload = { ...manifest };
delete manifestPayload.package_sha256;
const actualPackageSha256 = createHash("sha256")
  .update(JSON.stringify(sortJson(manifestPayload)))
  .digest("hex");

if (actualPackageSha256 !== manifest.package_sha256) {
  throw new Error("packed model manifest checksum mismatch");
}
if (expectedPackageSha256 && actualPackageSha256 !== expectedPackageSha256) {
  throw new Error(
    `unexpected packed model checksum: ${actualPackageSha256}`,
  );
}

for (const [arrayName, entry] of Object.entries(manifest.arrays)) {
  if (
    typeof entry !== "object"
    || entry === null
    || typeof entry.file !== "string"
    || basename(entry.file) !== entry.file
  ) {
    throw new Error(`unsafe or missing file name for packed array: ${arrayName}`);
  }

  const arrayPath = resolve(packageDirectory, entry.file);
  if (!arrayPath.startsWith(`${packageDirectory}${sep}`) || dirname(arrayPath) !== packageDirectory) {
    throw new Error(`packed array escapes package directory: ${arrayName}`);
  }

  const metadata = await stat(arrayPath);
  if (!metadata.isFile() || metadata.size !== Number(entry.byte_length)) {
    throw new Error(`packed array byte length mismatch: ${arrayName}`);
  }

  const digest = await sha256File(arrayPath);
  if (digest !== entry.sha256) {
    throw new Error(`packed array checksum mismatch: ${arrayName}`);
  }
}

console.log(
  `Verified ${Object.keys(manifest.arrays).length} arrays in package ${actualPackageSha256}.`,
);

function sortJson(value) {
  if (Array.isArray(value)) {
    return value.map(sortJson);
  }
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, child]) => [key, sortJson(child)]),
    );
  }
  return value;
}

function sha256File(path) {
  return new Promise((resolveDigest, reject) => {
    const digest = createHash("sha256");
    const source = createReadStream(path);
    source.on("data", (chunk) => digest.update(chunk));
    source.on("error", reject);
    source.on("end", () => resolveDigest(digest.digest("hex")));
  });
}
