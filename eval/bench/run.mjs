// Time a model on this machine's GPU in headless Chrome: bun run.mjs <model.onnx> <image.png> <out.png>
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import puppeteer from "puppeteer-core";

const [model, image, out] = process.argv.slice(2);
const here = import.meta.dirname;
const ort = join(here, "node_modules/onnxruntime-web/dist");
const files = { "/": join(here, "index.html"), "/model": model, "/image": image };
const types = { mjs: "text/javascript", wasm: "application/wasm", html: "text/html" };

const server = createServer(async (req, res) => {
  const path = new URL(req.url, "http://x").pathname;
  const file = files[path] ?? join(ort, basename(path));
  res.setHeader("content-type", types[file.split(".").pop()] ?? "application/octet-stream");
  res.end(await readFile(file).catch(() => ""));
}).listen(0);

const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: "new",
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU"],
});

const page = await browser.newPage();

await page.goto(`http://localhost:${server.address().port}/?model=/model&image=/image`);

await page.waitForFunction("window.result", { timeout: 600_000 });

const { png, ...result } = await page.evaluate("window.result");

if (png) await writeFile(out, Buffer.from(png.split(",")[1], "base64"));

console.log(JSON.stringify({ model: basename(model), ...result }));

await browser.close();

server.close();
