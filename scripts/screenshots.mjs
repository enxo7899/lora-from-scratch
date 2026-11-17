// Reproducible README screenshots. Run `npm run dev` in another terminal, then `npm run screenshots`.
import { createRequire } from "node:module";
const { chromium, devices } = createRequire(import.meta.url)("playwright");
const base = process.env.BASE || "http://127.0.0.1:5190/";
const out = process.env.OUT || "docs";
const browser = await chromium.launch();
const errors = [];
async function shot(name, url, { device, vh, viewport = { width: 1280, height: 900 }, full = false, clip } = {}) {
  const ctx = await browser.newContext(device ? { ...devices[device], ...(vh ? { viewport: { width: 390, height: vh } } : {}), deviceScaleFactor: 2 } : { viewport, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  page.on("console", (m) => m.type() === "error" && errors.push(`${name}: ${m.text()}`));
  page.on("pageerror", (e) => errors.push(`${name}: ${e.message}`));
  await page.goto(base + url, { waitUntil: "networkidle" });
  await page.waitForSelector("html[data-ready='1']", { timeout: 120000 });
  await page.waitForTimeout(400);
  const target = clip ? page.locator(clip) : page;
  await target.screenshot({ path: `${out}/${name}.png`, fullPage: full && !clip });
  await ctx.close();
}
await shot("overview", "?run=compare&adapter=code_r4", { viewport: { width: 1280, height: 1900 } });
await shot("compare", "?run=compare&adapter=code_r4", { clip: "#try" });
await shot("charts", "?run=generate", { clip: "#results" });
await shot("break", "?run=scale2,mismatch,merge&adapter=code_r4", { clip: "#break" });
await shot("mobile", "?run=generate&adapter=code_r4", { device: "iPhone 13", vh: 2300 });
await browser.close();
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("no console errors");
