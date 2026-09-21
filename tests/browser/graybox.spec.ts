import { expect, test, type Page } from "@playwright/test";

interface BrowserSnapshot {
  tick: number;
  roundStatus: number;
  handPhase: number;
  playerX: number;
  playerY: number;
  policyMode: "idle" | "chase" | "predictive" | "learned" | "connectome";
  pointerActive: boolean;
  debugEnabled: boolean;
  debugPanelVisible: boolean;
  reactionText: string | null;
  simulationRateHz: number;
  renderRateFps: number;
  connectome: {
    status: "idle" | "loading" | "ready" | "error";
    completedSteps: number;
    activitySampleCount: number;
    activityStep: number;
  };
}

test("loads, follows the pointer, and keeps brain selection automatic", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle("Can't Swat This");
  await expect(page.locator("canvas")).toBeVisible();
  await expect(page.locator("#fly-brain-visual")).toBeVisible();
  await expect(page.locator(".brain-node")).toHaveCount(64);
  expect((await snapshot(page)).connectome.status).not.toBe("idle");
  expect((await snapshot(page)).debugPanelVisible).toBe(false);
  await expect(page.locator("#connectome-status")).toBeHidden();

  await tapGameKey(page, "h");
  await expect.poll(async () => (await snapshot(page)).debugEnabled).toBe(true);
  await expect.poll(async () => (await snapshot(page)).debugPanelVisible).toBe(true);
  await expect(page.locator("#connectome-status")).toBeVisible();
  await tapGameKey(page, "1");
  await expect.poll(async () => (await snapshot(page)).policyMode).toBe("idle");
  const initial = await snapshot(page);

  const canvas = page.locator("canvas");
  const bounds = await canvas.boundingBox();
  expect(bounds).not.toBeNull();
  await page.mouse.move(bounds!.x + bounds!.width * 0.75, bounds!.y + bounds!.height * 0.25);
  await expect.poll(async () => (await snapshot(page)).pointerActive).toBe(true);
  await expect.poll(async () => (await snapshot(page)).playerX).toBeGreaterThan(initial.playerX);
  const trajectoryCsv = await page.evaluate(() => window.__flybrainGame!.humanTrajectoryCsv());
  expect(trajectoryCsv).toMatch(/^tick,destination_x_units,destination_y_units\n0,/);
  expect(trajectoryCsv.trim().split("\n").length).toBeGreaterThan(2);

  await tapGameKey(page, "2");
  await expect.poll(async () => (await snapshot(page)).policyMode).toBe("chase");
  await tapGameKey(page, "3");
  await expect.poll(async () => (await snapshot(page)).policyMode).toBe("predictive");
  await tapGameKey(page, "4");
  await tapGameKey(page, "5");
  await expect.poll(async () => (await snapshot(page)).policyMode).toBe("predictive");

  await tapGameKey(page, "h");
  await expect.poll(async () => (await snapshot(page)).debugPanelVisible).toBe(false);
  await expect(page.locator("#connectome-status")).toBeHidden();
  if (process.env.FLYBRAIN_SCREENSHOT_PATH) {
    await page.screenshot({ path: process.env.FLYBRAIN_SCREENSHOT_PATH, fullPage: true });
  }
});

test("collision ends the round and restart is immediate", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("canvas")).toBeVisible();
  await tapGameKey(page, "h");
  await tapGameKey(page, "3");

  await page.waitForFunction(
    () => window.__flybrainGame?.snapshot().handPhase === 1,
    undefined,
    { timeout: 3_000 },
  );
  await expect
    .poll(async () => (await snapshot(page)).roundStatus, { timeout: 5_000 })
    .toBe(1);
  const hitState = await snapshot(page);
  expect(hitState.tick).toBeGreaterThan(0);

  await tapGameKey(page, "r");
  await tapGameKey(page, "1");
  await expect.poll(async () => (await snapshot(page)).policyMode).toBe("idle");
  const restarted = await snapshot(page);
  expect(restarted.roundStatus).toBe(0);
  expect(restarted.tick).toBeLessThan(hitState.tick);
});

test("CPU fallback activates automatically and completes a slap", async ({ page }) => {
  await page.route("**/models/connectome-u16-v2/**", (route) => route.abort());
  await page.goto("/");
  await expect(page.locator("canvas")).toBeVisible();
  await expect
    .poll(async () => (await snapshot(page)).connectome.status)
    .toBe("error");
  await expect.poll(async () => (await snapshot(page)).policyMode).toBe("learned");
  await expect(page.locator("#brain-controller-badge")).toHaveText(
    "COMPACT GRU · CPU FALLBACK",
  );
  await expect(page.locator("#brain-region-sensory")).toBeHidden();
  await expect(page.locator("#brain-region-network")).toHaveText("64 GRU units");
  await expect(page.locator("#brain-region-motor")).toBeHidden();
  await expect
    .poll(async () => (await snapshot(page)).roundStatus, { timeout: 6_000 })
    .toBe(1);
  expect((await snapshot(page)).tick).toBeLessThan(240);
});

test("a committed slap can miss and shows its reaction", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("canvas")).toBeVisible();
  await tapGameKey(page, "h");
  await tapGameKey(page, "3");

  const canvas = page.locator("canvas");
  const bounds = await canvas.boundingBox();
  expect(bounds).not.toBeNull();
  await page.mouse.move(
    bounds!.x + bounds!.width * 0.1,
    bounds!.y + bounds!.height * 0.8,
  );
  await page.waitForFunction(
    () => window.__flybrainGame?.snapshot().handPhase === 1,
    undefined,
    { timeout: 3_000 },
  );

  await page.mouse.move(
    bounds!.x + bounds!.width * 0.9,
    bounds!.y + bounds!.height * 0.16,
  );
  await page.waitForFunction(
    () => window.__flybrainGame?.snapshot().reactionText === "MISS!",
    undefined,
    { timeout: 2_000 },
  );
  expect((await snapshot(page)).roundStatus).toBe(0);
});

async function snapshot(page: Page): Promise<BrowserSnapshot> {
  await page.waitForFunction(() => window.__flybrainGame !== undefined);
  return page.evaluate(() => window.__flybrainGame!.snapshot());
}

async function tapGameKey(page: Page, key: string): Promise<void> {
  await page.keyboard.down(key);
  await page.waitForTimeout(100);
  await page.keyboard.up(key);
}

declare global {
  interface Window {
    __flybrainGame?: {
      snapshot: () => BrowserSnapshot;
      humanTrajectoryCsv: () => string;
    };
  }
}
