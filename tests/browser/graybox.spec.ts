import { expect, test, type Page } from "@playwright/test";

interface BrowserSnapshot {
  tick: number;
  roundStatus: number;
  playerX: number;
  playerY: number;
  policyMode: "idle" | "chase" | "predictive" | "learned" | "connectome";
  pointerActive: boolean;
  debugEnabled: boolean;
  simulationRateHz: number;
  renderRateFps: number;
  connectome: {
    status: "idle" | "loading" | "ready" | "error";
    completedSteps: number;
  };
}

test("loads, follows the pointer, and switches policy modes", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("canvas")).toBeVisible();
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
  await expect.poll(async () => (await snapshot(page)).policyMode).toBe("learned");

  await tapGameKey(page, "h");
  await expect.poll(async () => (await snapshot(page)).debugEnabled).toBe(true);
  if (process.env.FLYBRAIN_SCREENSHOT_PATH) {
    await page.screenshot({ path: process.env.FLYBRAIN_SCREENSHOT_PATH });
  }
});

test("collision ends the round and restart is immediate", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("canvas")).toBeVisible();

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

test("frozen learned GRU runs locally and completes a slap", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("canvas")).toBeVisible();
  await tapGameKey(page, "4");
  await expect.poll(async () => (await snapshot(page)).policyMode).toBe("learned");
  await expect
    .poll(async () => (await snapshot(page)).roundStatus, { timeout: 6_000 })
    .toBe(1);
  expect((await snapshot(page)).tick).toBeLessThan(240);
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
