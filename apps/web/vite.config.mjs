import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? "/",
  build: {
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL("./index.html", import.meta.url)),
        benchmark: fileURLToPath(
          new URL("./connectome-benchmark.html", import.meta.url),
        ),
      },
    },
  },
});
