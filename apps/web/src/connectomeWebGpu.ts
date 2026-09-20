const FORMAT_NAME = "flybrain-connectome-packed";
const FORMAT_VERSION = 2;
const EMPTY_SENSORY = 0xffff_ffff;
const BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  STORAGE: 0x0080,
} as const;
const MAP_READ = 0x0001;

interface ArrayEntry {
  file: string;
  dtype: string;
  shape: number[];
  byte_length: number;
  sha256: string;
}

interface PackedManifest {
  format: string;
  format_version: number;
  package_sha256: string;
  dimensions: {
    neurons: number;
    edges: number;
    features: number;
    motor_neurons: number;
  };
  model: {
    id: string;
  };
  outputs: {
    strike_threshold: number;
  };
  arrays: Record<string, ArrayEntry>;
  quantization: {
    scheme: string;
    weight_bits: number;
  };
  [key: string]: unknown;
}

export interface ConnectomeGpuProgress {
  stage: "manifest" | "device" | "array" | "ready";
  detail: string;
  loadedBytes: number;
  totalBytes: number;
}

export interface ConnectomeGpuDecision {
  targetX: number;
  targetY: number;
  strikeLogit: number;
  strikeProbability: number;
  strike: boolean;
}

export interface ConnectomeGpuInfo {
  backend: "webgpu-rowwise-u16";
  formatVersion: 2;
  modelId: string;
  packageSha256: string;
  neuronCount: number;
  edgeCount: number;
  adapter: string;
}

type ProgressCallback = (progress: ConnectomeGpuProgress) => void;

export class ConnectomeWebGpu {
  readonly info: ConnectomeGpuInfo;
  readonly featureCount: number;
  readonly strikeThreshold: number;

  private readonly device: GPUDevice;
  private readonly featureBuffer: GPUBuffer;
  private readonly stateBuffers: readonly [GPUBuffer, GPUBuffer];
  private readonly recurrencePipeline: GPUComputePipeline;
  private readonly recurrenceBindGroups: readonly [GPUBindGroup, GPUBindGroup];
  private readonly readoutPipeline: GPUComputePipeline;
  private readonly readoutBindGroups: readonly [GPUBindGroup, GPUBindGroup];
  private readonly outputBuffer: GPUBuffer;
  private readonly readbackBuffer: GPUBuffer;
  private readonly ownedBuffers: GPUBuffer[];
  private stateIndex = 0;

  private constructor(
    manifest: PackedManifest,
    device: GPUDevice,
    adapterLabel: string,
    buffers: ModelBuffers,
  ) {
    this.device = device;
    this.featureCount = manifest.dimensions.features;
    this.strikeThreshold = manifest.outputs.strike_threshold;
    this.info = {
      backend: "webgpu-rowwise-u16",
      formatVersion: FORMAT_VERSION,
      modelId: manifest.model.id,
      packageSha256: manifest.package_sha256,
      neuronCount: manifest.dimensions.neurons,
      edgeCount: manifest.dimensions.edges,
      adapter: adapterLabel,
    };

    this.featureBuffer = emptyBuffer(
      device,
      "connectome features",
      this.featureCount * Float32Array.BYTES_PER_ELEMENT,
      BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST,
    );
    const stateBytes = manifest.dimensions.neurons * Float32Array.BYTES_PER_ELEMENT;
    this.stateBuffers = [
      emptyBuffer(
        device,
        "connectome state A",
        stateBytes,
        BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST,
      ),
      emptyBuffer(
        device,
        "connectome state B",
        stateBytes,
        BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST,
      ),
    ];
    this.outputBuffer = emptyBuffer(
      device,
      "connectome output",
      3 * Float32Array.BYTES_PER_ELEMENT,
      BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_SRC,
    );
    this.readbackBuffer = emptyBuffer(
      device,
      "connectome output readback",
      3 * Float32Array.BYTES_PER_ELEMENT,
      BUFFER_USAGE.COPY_DST | BUFFER_USAGE.MAP_READ,
    );

    this.recurrencePipeline = device.createComputePipeline({
      label: "connectome sparse recurrence",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "connectome sparse recurrence shader",
          code: recurrenceShader(manifest.dimensions.neurons),
        }),
        entryPoint: "main",
      },
    });
    const recurrenceLayout = this.recurrencePipeline.getBindGroupLayout(0);
    this.recurrenceBindGroups = [
      recurrenceBindGroup(
        device,
        recurrenceLayout,
        buffers,
        this.featureBuffer,
        this.stateBuffers[0],
        this.stateBuffers[1],
      ),
      recurrenceBindGroup(
        device,
        recurrenceLayout,
        buffers,
        this.featureBuffer,
        this.stateBuffers[1],
        this.stateBuffers[0],
      ),
    ];

    this.readoutPipeline = device.createComputePipeline({
      label: "connectome motor readout",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "connectome motor readout shader",
          code: readoutShader(manifest.dimensions.motor_neurons),
        }),
        entryPoint: "main",
      },
    });
    const readoutLayout = this.readoutPipeline.getBindGroupLayout(0);
    this.readoutBindGroups = [
      readoutBindGroup(
        device,
        readoutLayout,
        this.stateBuffers[0],
        buffers,
        this.outputBuffer,
      ),
      readoutBindGroup(
        device,
        readoutLayout,
        this.stateBuffers[1],
        buffers,
        this.outputBuffer,
      ),
    ];
    this.ownedBuffers = [
      ...buffers.owned,
      this.featureBuffer,
      ...this.stateBuffers,
      this.outputBuffer,
      this.readbackBuffer,
    ];
  }

  static async create(
    baseUrl: string,
    onProgress: ProgressCallback = () => undefined,
  ): Promise<ConnectomeWebGpu> {
    const manifestUrl = `${baseUrl.replace(/\/$/, "")}/manifest.json`;
    onProgress({ stage: "manifest", detail: manifestUrl, loadedBytes: 0, totalBytes: 0 });
    const manifestResponse = await fetch(manifestUrl);
    if (!manifestResponse.ok) {
      throw new Error(`model manifest request failed: ${manifestResponse.status}`);
    }
    const manifest = validateManifest(await manifestResponse.json());
    await verifyManifestDigest(manifest);
    const totalBytes = Object.values(manifest.arrays).reduce(
      (sum, entry) => sum + entry.byte_length,
      0,
    );

    onProgress({ stage: "device", detail: "requesting adapter", loadedBytes: 0, totalBytes });
    if (!navigator.gpu) {
      throw new Error("WebGPU is unavailable");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      throw new Error("WebGPU returned no adapter");
    }
    const largestBinding = Math.max(
      manifest.arrays.column_indices?.byte_length ?? 0,
      manifest.arrays.edge_weights?.byte_length ?? 0,
    );
    if (
      adapter.limits.maxStorageBufferBindingSize < largestBinding
      || adapter.limits.maxBufferSize < largestBinding
    ) {
      throw new Error(
        `WebGPU adapter buffer limit is too small (${adapter.limits.maxStorageBufferBindingSize})`,
      );
    }
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: largestBinding,
        maxBufferSize: largestBinding,
      },
    });
    const adapterInfo = adapter.info;
    const adapterLabel = [adapterInfo.vendor, adapterInfo.architecture, adapterInfo.device]
      .filter((value) => value.length > 0)
      .join(" ") || "WebGPU adapter";
    const loader = new ArrayLoader(baseUrl, manifest, onProgress, totalBytes);
    const buffers = await loadModelBuffers(device, loader, manifest);
    onProgress({ stage: "ready", detail: adapterLabel, loadedBytes: totalBytes, totalBytes });
    return new ConnectomeWebGpu(manifest, device, adapterLabel, buffers);
  }

  async step(features: Float32Array): Promise<ConnectomeGpuDecision> {
    if (features.length !== this.featureCount) {
      throw new Error(`expected ${this.featureCount} features, received ${features.length}`);
    }
    this.device.queue.writeBuffer(this.featureBuffer, 0, features);
    const nextStateIndex = 1 - this.stateIndex;
    const encoder = this.device.createCommandEncoder({ label: "connectome inference tick" });
    const recurrencePass = encoder.beginComputePass({ label: "connectome recurrence pass" });
    recurrencePass.setPipeline(this.recurrencePipeline);
    recurrencePass.setBindGroup(
      0,
      required(this.recurrenceBindGroups[this.stateIndex], "recurrence bind group"),
    );
    recurrencePass.dispatchWorkgroups(Math.ceil(this.info.neuronCount / 64));
    recurrencePass.end();
    const readoutPass = encoder.beginComputePass({ label: "connectome readout pass" });
    readoutPass.setPipeline(this.readoutPipeline);
    readoutPass.setBindGroup(
      0,
      required(this.readoutBindGroups[nextStateIndex], "readout bind group"),
    );
    readoutPass.dispatchWorkgroups(3);
    readoutPass.end();
    encoder.copyBufferToBuffer(this.outputBuffer, 0, this.readbackBuffer, 0, 12);
    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuffer.mapAsync(MAP_READ);
    const raw = new Float32Array(this.readbackBuffer.getMappedRange()).slice();
    this.readbackBuffer.unmap();
    this.stateIndex = nextStateIndex;
    const targetX = required(raw[0], "target x");
    const targetY = required(raw[1], "target y");
    const strikeLogit = required(raw[2], "strike logit");
    const strikeProbability = 1 / (1 + Math.exp(-strikeLogit));
    return {
      targetX,
      targetY,
      strikeLogit,
      strikeProbability,
      strike: strikeProbability >= this.strikeThreshold,
    };
  }

  reset(): void {
    const zeros = new Uint8Array(this.info.neuronCount * Float32Array.BYTES_PER_ELEMENT);
    this.device.queue.writeBuffer(this.stateBuffers[0], 0, zeros);
    this.device.queue.writeBuffer(this.stateBuffers[1], 0, zeros);
    this.stateIndex = 0;
  }

  destroy(): void {
    for (const buffer of this.ownedBuffers) {
      buffer.destroy();
    }
    this.device.destroy();
  }
}

interface ModelBuffers {
  rowOffsets: GPUBuffer;
  columns: GPUBuffer;
  weights: GPUBuffer;
  neuronMeta: GPUBuffer;
  motorIndices: GPUBuffer;
  readoutWeight: GPUBuffer;
  readoutBias: GPUBuffer;
  owned: GPUBuffer[];
}

async function loadModelBuffers(
  device: GPUDevice,
  loader: ArrayLoader,
  manifest: PackedManifest,
): Promise<ModelBuffers> {
  const rowOffsets = bufferFromBytes(
    device,
    "connectome row offsets",
    await loader.bytes("row_offsets"),
    BUFFER_USAGE.STORAGE,
  );
  const columns = bufferFromBytes(
    device,
    "connectome columns",
    await loader.bytes("column_indices"),
    BUFFER_USAGE.STORAGE,
  );
  const weights = bufferFromBytes(
    device,
    "connectome u16 weights",
    await loader.bytes("edge_weights"),
    BUFFER_USAGE.STORAGE,
  );

  const edgeScales = new Float32Array(await loader.bytes("edge_scales"));
  const leak = new Float32Array(await loader.bytes("leak"));
  const sensoryIndices = new Uint32Array(await loader.bytes("sensory_indices"));
  const sensoryFeatureIds = new Uint8Array(await loader.bytes("sensory_feature_ids"));
  const sensorySigns = new Int8Array(await loader.bytes("sensory_signs"));
  const neuronCount = manifest.dimensions.neurons;
  if (edgeScales.length !== neuronCount || leak.length !== neuronCount) {
    throw new Error("neuron metadata length mismatch");
  }
  if (
    sensoryIndices.length !== sensoryFeatureIds.length
    || sensoryIndices.length !== sensorySigns.length
  ) {
    throw new Error("sensory metadata length mismatch");
  }
  const metadataBytes = new ArrayBuffer(neuronCount * 16);
  const metadataFloats = new Float32Array(metadataBytes);
  const metadataU32 = new Uint32Array(metadataBytes);
  for (let neuron = 0; neuron < neuronCount; neuron += 1) {
    metadataFloats[neuron * 4] = required(edgeScales[neuron], "edge scale");
    metadataFloats[neuron * 4 + 1] = required(leak[neuron], "leak");
    metadataU32[neuron * 4 + 2] = EMPTY_SENSORY;
  }
  for (let index = 0; index < sensoryIndices.length; index += 1) {
    const neuron = required(sensoryIndices[index], "sensory index");
    const feature = required(sensoryFeatureIds[index], "sensory feature");
    const sign = required(sensorySigns[index], "sensory sign");
    if (neuron >= neuronCount || feature >= manifest.dimensions.features) {
      throw new Error("sensory metadata index is out of range");
    }
    metadataU32[neuron * 4 + 2] = feature | (sign > 0 ? 0x100 : 0);
  }
  const neuronMeta = bufferFromBytes(
    device,
    "connectome neuron metadata",
    metadataBytes,
    BUFFER_USAGE.STORAGE,
  );
  const motorIndices = bufferFromBytes(
    device,
    "connectome motor indices",
    await loader.bytes("motor_indices"),
    BUFFER_USAGE.STORAGE,
  );
  const readoutWeight = bufferFromBytes(
    device,
    "connectome readout weight",
    await loader.bytes("readout_weight"),
    BUFFER_USAGE.STORAGE,
  );
  const readoutBias = bufferFromBytes(
    device,
    "connectome readout bias",
    await loader.bytes("readout_bias"),
    BUFFER_USAGE.STORAGE,
  );
  const owned = [
    rowOffsets,
    columns,
    weights,
    neuronMeta,
    motorIndices,
    readoutWeight,
    readoutBias,
  ];
  return {
    rowOffsets,
    columns,
    weights,
    neuronMeta,
    motorIndices,
    readoutWeight,
    readoutBias,
    owned,
  };
}

class ArrayLoader {
  private loadedBytes = 0;

  constructor(
    private readonly baseUrl: string,
    private readonly manifest: PackedManifest,
    private readonly onProgress: ProgressCallback,
    private readonly totalBytes: number,
  ) {}

  async bytes(name: string): Promise<ArrayBuffer> {
    const entry = this.manifest.arrays[name];
    if (!entry) {
      throw new Error(`model array is missing from manifest: ${name}`);
    }
    const response = await fetch(`${this.baseUrl.replace(/\/$/, "")}/${entry.file}`);
    if (!response.ok) {
      throw new Error(`model array request failed for ${name}: ${response.status}`);
    }
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength !== entry.byte_length) {
      throw new Error(`model array byte length mismatch: ${name}`);
    }
    const digest = await sha256(bytes);
    if (digest !== entry.sha256) {
      throw new Error(`model array checksum mismatch: ${name}`);
    }
    this.loadedBytes += bytes.byteLength;
    this.onProgress({
      stage: "array",
      detail: name,
      loadedBytes: this.loadedBytes,
      totalBytes: this.totalBytes,
    });
    return bytes;
  }
}

function recurrenceBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  buffers: ModelBuffers,
  features: GPUBuffer,
  stateInput: GPUBuffer,
  stateOutput: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: "connectome recurrence bindings",
    layout,
    entries: [
      { binding: 0, resource: { buffer: buffers.rowOffsets } },
      { binding: 1, resource: { buffer: buffers.columns } },
      { binding: 2, resource: { buffer: buffers.weights } },
      { binding: 3, resource: { buffer: buffers.neuronMeta } },
      { binding: 4, resource: { buffer: features } },
      { binding: 5, resource: { buffer: stateInput } },
      { binding: 6, resource: { buffer: stateOutput } },
    ],
  });
}

function readoutBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  state: GPUBuffer,
  buffers: ModelBuffers,
  output: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    label: "connectome readout bindings",
    layout,
    entries: [
      { binding: 0, resource: { buffer: state } },
      { binding: 1, resource: { buffer: buffers.motorIndices } },
      { binding: 2, resource: { buffer: buffers.readoutWeight } },
      { binding: 3, resource: { buffer: buffers.readoutBias } },
      { binding: 4, resource: { buffer: output } },
    ],
  });
}

function recurrenceShader(neuronCount: number): string {
  return `
struct NeuronMeta {
  edge_scale: f32,
  leak: f32,
  sensory: u32,
  padding: u32,
}

@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> columns: array<u32>;
@group(0) @binding(2) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(3) var<storage, read> metadata: array<NeuronMeta>;
@group(0) @binding(4) var<storage, read> features: array<f32>;
@group(0) @binding(5) var<storage, read> state_in: array<f32>;
@group(0) @binding(6) var<storage, read_write> state_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= ${neuronCount}u) {
    return;
  }
  let start = row_offsets[row];
  let end = row_offsets[row + 1u];
  var weighted_sum = 0.0;
  for (var edge = start; edge < end; edge += 1u) {
    let packed = packed_weights[edge >> 1u];
    let shift = (edge & 1u) * 16u;
    let weight = f32((packed >> shift) & 65535u);
    weighted_sum += weight * state_in[columns[edge]];
  }
  let neuron = metadata[row];
  var signal = weighted_sum * neuron.edge_scale;
  if (neuron.sensory != 0xffffffffu) {
    let feature = neuron.sensory & 255u;
    let sign = select(-1.0, 1.0, (neuron.sensory & 256u) != 0u);
    signal += features[feature] * sign;
  }
  state_out[row] = (1.0 - neuron.leak) * state_in[row]
    + neuron.leak * tanh(signal);
}
`;
}

function readoutShader(motorCount: number): string {
  return `
@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<storage, read> motor_indices: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let channel = id.x;
  if (channel >= 3u) {
    return;
  }
  var value = bias[channel];
  for (var motor = 0u; motor < ${motorCount}u; motor += 1u) {
    value += weights[channel * ${motorCount}u + motor]
      * state[motor_indices[motor]];
  }
  output[channel] = select(1.0 / (1.0 + exp(-value)), value, channel == 2u);
}
`;
}

function bufferFromBytes(
  device: GPUDevice,
  label: string,
  bytes: ArrayBuffer,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const size = alignedSize(bytes.byteLength);
  const buffer = device.createBuffer({ label, size, usage, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(bytes));
  buffer.unmap();
  return buffer;
}

function emptyBuffer(
  device: GPUDevice,
  label: string,
  byteLength: number,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  return device.createBuffer({ label, size: alignedSize(byteLength), usage });
}

function alignedSize(byteLength: number): number {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

function validateManifest(value: unknown): PackedManifest {
  if (typeof value !== "object" || value === null) {
    throw new Error("model manifest is not an object");
  }
  const manifest = value as PackedManifest;
  if (manifest.format !== FORMAT_NAME || manifest.format_version !== FORMAT_VERSION) {
    throw new Error("model manifest format/version is unsupported");
  }
  if (
    manifest.quantization?.scheme !== "rowwise-unsigned-max-v1"
    || manifest.quantization.weight_bits !== 16
  ) {
    throw new Error("model quantization scheme is unsupported");
  }
  if (!manifest.arrays || !manifest.dimensions || !manifest.outputs || !manifest.model?.id) {
    throw new Error("model manifest is incomplete");
  }
  return manifest;
}

async function verifyManifestDigest(manifest: PackedManifest): Promise<void> {
  const payload: Partial<PackedManifest> = { ...manifest };
  delete payload.package_sha256;
  const canonical = JSON.stringify(sortJson(payload));
  const digest = await sha256(new TextEncoder().encode(canonical).buffer);
  if (digest !== manifest.package_sha256) {
    throw new Error("model manifest checksum mismatch");
  }
}

function sortJson(value: unknown): unknown {
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

async function sha256(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function required<T>(value: T | undefined, label: string): T {
  if (value === undefined) {
    throw new Error(`${label} is missing`);
  }
  return value;
}
