import { expect, test, type Page } from "@playwright/test";

const enabled = process.env.FLYBRAIN_CONNECTOME_WEBGPU === "1";

test("validated u16 package runs in a WebGPU worker", async ({ page }) => {
  test.skip(!enabled, "set FLYBRAIN_CONNECTOME_WEBGPU=1 for the large optional model");
  test.setTimeout(180_000);
  await page.goto("/connectome-benchmark.html?ticks=60");
  await page.waitForFunction(
    () => window.__flybrainConnectomeBenchmark?.snapshot().status !== "loading",
    undefined,
    { timeout: 170_000 },
  );
  const snapshot = await page.evaluate(
    () => window.__flybrainConnectomeBenchmark?.snapshot(),
  );
  expect(snapshot?.status, JSON.stringify(snapshot)).toBe("measured");
  const result = snapshot?.detail as {
    packageSha256?: string;
    ticks?: number;
    inferenceMedianMs?: number;
    finalDecision?: {
      targetX?: number;
      targetY?: number;
      strikeLogit?: number;
      activity?: number[];
    };
  };
  expect(result.packageSha256).toBe(
    "0f5baf90bf5bed5802931b289d551474524547872ceb3eff657e25d9d8e54a37",
  );
  expect(result.ticks).toBe(60);
  expect(result.inferenceMedianMs).toBeGreaterThan(0);
  expect(result.finalDecision?.targetX).toBeCloseTo(0.3108477, 4);
  expect(result.finalDecision?.targetY).toBeCloseTo(0.5584587, 4);
  expect(result.finalDecision?.strikeLogit).toBeCloseTo(-1.1622446, 4);
  expect(result.finalDecision?.activity).toHaveLength(64);
  expect(result.finalDecision?.activity?.every(Number.isFinite)).toBe(true);
  expect(result.finalDecision?.activity?.some((value) => Math.abs(value) > 1e-6)).toBe(true);
  console.log(`WebGPU benchmark: ${JSON.stringify(result)}`);
});

test("full connectome game mode uses the WebGPU worker", async ({ page }) => {
  test.skip(!enabled, "set FLYBRAIN_CONNECTOME_WEBGPU=1 for the large optional model");
  test.setTimeout(180_000);
  await page.goto("/");
  await page.waitForFunction(() => window.__flybrainGame !== undefined);
  await page.waitForFunction(
    () => window.__flybrainGame?.snapshot().connectome.status === "ready",
    undefined,
    { timeout: 170_000 },
  );
  await page.waitForFunction(
    () => (window.__flybrainGame?.snapshot().connectome.completedSteps ?? 0) > 0,
    undefined,
    { timeout: 20_000 },
  );
  const snapshot = await page.evaluate(() => window.__flybrainGame?.snapshot());
  expect(snapshot?.policyMode).toBe("connectome");
  expect(snapshot?.connectome.status).toBe("ready");
  expect(snapshot?.connectome.completedSteps).toBeGreaterThan(0);
  expect(snapshot?.connectome.inferenceMedianMs).toBeGreaterThan(0);
  await page.waitForFunction(
    () => (window.__flybrainGame?.snapshot().connectome.activityStep ?? 0) > 0,
  );
  const activitySnapshot = await page.evaluate(() => window.__flybrainGame?.snapshot());
  expect(activitySnapshot?.connectome.activitySampleCount).toBe(64);
  await expect(page.locator(".brain-node")).toHaveCount(64);
  await expect(page.locator("#brain-controller-badge")).toHaveText("FULL MALECNS · LIVE");
  await tapKey(page, "h");
  await page.waitForFunction(() => window.__flybrainGame?.snapshot().debugEnabled === true);
  const initialTick = snapshot?.tick ?? 0;
  const initialPlayerX = snapshot?.playerX ?? 0;
  const canvas = page.locator("canvas");
  const bounds = await canvas.boundingBox();
  expect(bounds).not.toBeNull();
  await page.mouse.move(bounds!.x + bounds!.width * 0.8, bounds!.y + bounds!.height * 0.5);
  await page.waitForTimeout(250);
  const later = await page.evaluate(() => window.__flybrainGame?.snapshot());
  expect((later?.tick ?? 0) - initialTick).toBeGreaterThan(1);
  expect(later?.playerX ?? 0).toBeGreaterThan(initialPlayerX);
  await page.waitForTimeout(750);
  const sustained = await page.evaluate(() => window.__flybrainGame?.snapshot());
  expect(sustained?.policyMode).toBe("connectome");
  expect(sustained?.connectome.status).toBe("ready");
  expect(sustained?.connectome.pendingSteps).toBeLessThanOrEqual(1);

  await tapKey(page, "3");
  await page.waitForFunction(() => window.__flybrainGame?.snapshot().policyMode === "predictive");
  await tapKey(page, "5");
  await page.waitForFunction(
    () => (
      window.__flybrainGame?.snapshot().policyMode === "connectome"
      && (window.__flybrainGame?.snapshot().connectome.completedSteps ?? 0) > 0
    ),
    undefined,
    { timeout: 5_000 },
  );
  await tapKey(page, "h");
  await page.waitForFunction(
    () => window.__flybrainGame?.snapshot().debugPanelVisible === false,
  );
  await expect(page.locator("#connectome-status")).toBeHidden();
  if (process.env.FLYBRAIN_CONNECTOME_SCREENSHOT_PATH) {
    await page.screenshot({ path: process.env.FLYBRAIN_CONNECTOME_SCREENSHOT_PATH, fullPage: true });
  }
});

async function tapKey(page: Page, key: string): Promise<void> {
  await page.keyboard.down(key);
  await page.waitForTimeout(100);
  await page.keyboard.up(key);
}

declare global {
  interface Window {
    __flybrainConnectomeBenchmark?: {
      snapshot: () => { status: "loading" | "measured" | "error"; detail: unknown };
    };
    __flybrainGame?: {
      snapshot: () => {
        policyMode: string;
        tick: number;
        handPhase: number;
        playerX: number;
        debugEnabled: boolean;
        debugPanelVisible: boolean;
        reactionText: string | null;
        simulationRateHz: number;
        connectome: {
          status: "idle" | "loading" | "ready" | "error";
          completedSteps: number;
          inferenceMedianMs: number | null;
          pendingSteps: number;
          activitySampleCount: number;
          activityStep: number;
        };
      };
    };
  }
}
