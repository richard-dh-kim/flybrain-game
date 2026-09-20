const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

type BrainDisplayMode = "loading" | "connectome" | "fallback" | "development";

export interface BrainVisualizationState {
  mode: BrainDisplayMode;
  activity: ArrayLike<number>;
  step?: number;
  loadPercent?: number;
  fallbackReason?: string;
}

interface Point {
  x: number;
  y: number;
}

const SENSORY_POINTS: Point[] = [
  { x: 71, y: 65 }, { x: 49, y: 84 }, { x: 43, y: 112 }, { x: 52, y: 141 },
  { x: 74, y: 161 }, { x: 101, y: 151 }, { x: 112, y: 121 }, { x: 103, y: 87 },
  { x: 349, y: 65 }, { x: 371, y: 84 }, { x: 377, y: 112 }, { x: 368, y: 141 },
  { x: 346, y: 161 }, { x: 319, y: 151 }, { x: 308, y: 121 }, { x: 317, y: 87 },
];

const GRAPH_POINTS: Point[] = [
  { x: 165, y: 68 }, { x: 190, y: 60 }, { x: 215, y: 60 }, { x: 240, y: 68 },
  { x: 146, y: 91 }, { x: 168, y: 88 }, { x: 190, y: 85 }, { x: 215, y: 85 },
  { x: 240, y: 88 }, { x: 262, y: 91 },
  { x: 140, y: 116 }, { x: 163, y: 113 }, { x: 186, y: 110 }, { x: 210, y: 109 },
  { x: 234, y: 110 }, { x: 257, y: 113 }, { x: 280, y: 116 },
  { x: 146, y: 142 }, { x: 168, y: 139 }, { x: 190, y: 137 }, { x: 215, y: 137 },
  { x: 240, y: 139 }, { x: 262, y: 142 },
  { x: 158, y: 166 }, { x: 181, y: 164 }, { x: 204, y: 162 }, { x: 227, y: 164 },
  { x: 250, y: 166 },
  { x: 177, y: 188 }, { x: 199, y: 187 }, { x: 221, y: 187 }, { x: 243, y: 188 },
];

const MOTOR_POINTS: Point[] = [
  { x: 173, y: 205 }, { x: 198, y: 205 }, { x: 222, y: 205 }, { x: 247, y: 205 },
  { x: 182, y: 222 }, { x: 201, y: 222 }, { x: 219, y: 222 }, { x: 238, y: 222 },
  { x: 188, y: 239 }, { x: 203, y: 239 }, { x: 217, y: 239 }, { x: 232, y: 239 },
  { x: 194, y: 256 }, { x: 205, y: 256 }, { x: 215, y: 256 }, { x: 226, y: 256 },
];

const ALL_POINTS = [...SENSORY_POINTS, ...GRAPH_POINTS, ...MOTOR_POINTS];

export class FlyBrainVisualization {
  private readonly visual: SVGSVGElement;
  private readonly statusElement: HTMLElement;
  private readonly badgeElement: HTMLElement;
  private readonly stepElement: HTMLElement;
  private readonly nodes: SVGCircleElement[];
  private lastStatus = "";

  constructor() {
    this.visual = requiredElement<SVGSVGElement>("#fly-brain-visual");
    this.statusElement = requiredElement<HTMLElement>("#brain-monitor-status");
    this.badgeElement = requiredElement<HTMLElement>("#brain-controller-badge");
    this.stepElement = requiredElement<HTMLElement>("#brain-step");
    const connections = requiredElement<SVGGElement>("#brain-connections");
    const activity = requiredElement<SVGGElement>("#brain-activity");

    drawConnections(connections);
    this.nodes = ALL_POINTS.map((point, index) => {
      const node = createSvgElement("circle");
      node.setAttribute("cx", String(point.x));
      node.setAttribute("cy", String(point.y));
      node.setAttribute("r", "3.2");
      node.setAttribute("class", `brain-node brain-node--${regionForIndex(index)}`);
      activity.append(node);
      return node;
    });
  }

  update(state: BrainVisualizationState): void {
    this.visual.dataset.state = state.mode;
    const presentation = presentationForState(state);
    this.badgeElement.textContent = presentation.badge;
    this.badgeElement.dataset.mode = state.mode;
    if (presentation.status !== this.lastStatus) {
      this.statusElement.textContent = presentation.status;
      this.lastStatus = presentation.status;
    }
    this.stepElement.textContent = state.step === undefined ? "" : `brain step ${state.step}`;

    for (const [index, node] of this.nodes.entries()) {
      const value = state.activity[index] ?? Number.NaN;
      if (!Number.isFinite(value)) {
        node.setAttribute("fill", "#b89072");
        node.setAttribute("fill-opacity", "0.18");
        node.setAttribute("r", "3.2");
        node.removeAttribute("filter");
        continue;
      }

      const magnitude = Math.min(1, Math.abs(value));
      node.setAttribute("fill", value >= 0 ? "#ffbd59" : "#67d5ff");
      node.setAttribute("fill-opacity", String(0.24 + magnitude * 0.76));
      node.setAttribute("r", String(3.2 + magnitude * 3.4));
      if (magnitude > 0.22) {
        node.setAttribute("filter", "url(#brain-node-glow)");
      } else {
        node.removeAttribute("filter");
      }
    }
  }
}

function presentationForState(state: BrainVisualizationState): {
  badge: string;
  status: string;
} {
  if (state.mode === "connectome") {
    return {
      badge: "FULL MALECNS · LIVE",
      status: "64 live samples from the 165,122-neuron controller",
    };
  }
  if (state.mode === "loading") {
    const percent = state.loadPercent === undefined ? "" : ` · ${state.loadPercent}%`;
    return {
      badge: `LOADING FULL BRAIN${percent}`,
      status: "The compact brain controls the hands while MaleCNS loads",
    };
  }
  if (state.mode === "fallback") {
    return {
      badge: "COMPACT BRAIN · FALLBACK",
      status: state.fallbackReason
        ? `Full brain unavailable · ${state.fallbackReason}`
        : "64 live units from the compact recurrent controller",
    };
  }
  return {
    badge: "DEVELOPMENT CONTROLLER",
    status: "Neural activity is paused for this test controller",
  };
}

function drawConnections(group: SVGGElement): void {
  const pairs: Array<readonly [number, number]> = [];
  for (let index = 0; index < 16; index += 1) {
    pairs.push([index, 16 + ((index * 5) % 32)]);
  }
  for (let index = 0; index < 31; index += 1) {
    pairs.push([16 + index, 16 + index + 1]);
  }
  for (let index = 0; index < 24; index += 1) {
    pairs.push([16 + index, 16 + index + 8]);
  }
  for (let index = 0; index < 16; index += 1) {
    pairs.push([32 + index, 48 + index]);
  }
  for (let index = 0; index < 15; index += 1) {
    pairs.push([48 + index, 48 + index + 1]);
  }

  for (const [fromIndex, toIndex] of pairs) {
    const from = ALL_POINTS[fromIndex];
    const to = ALL_POINTS[toIndex];
    if (!from || !to) {
      continue;
    }
    const line = createSvgElement("line");
    line.setAttribute("x1", String(from.x));
    line.setAttribute("y1", String(from.y));
    line.setAttribute("x2", String(to.x));
    line.setAttribute("y2", String(to.y));
    line.setAttribute("class", "brain-synapse");
    group.append(line);
  }
}

function regionForIndex(index: number): "sensory" | "graph" | "motor" {
  if (index < 16) {
    return "sensory";
  }
  return index < 48 ? "graph" : "motor";
}

function createSvgElement<K extends keyof SVGElementTagNameMap>(
  tagName: K,
): SVGElementTagNameMap[K] {
  return document.createElementNS(SVG_NAMESPACE, tagName);
}

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Required brain visualization element is missing: ${selector}`);
  }
  return element;
}
