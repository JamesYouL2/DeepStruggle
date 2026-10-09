import { resolve } from 'node:path';
import { defineConfig } from 'vite';

// TS_WEB_BASE: the path the page is served under -- "/" locally, "/<repo>/" on GitHub Pages
// (set by .github/workflows/pages.yml). The engine and model runtime are loaded relative to it.
export default defineConfig({
  base: process.env.TS_WEB_BASE || '/',
  server: {
    port: 5173,
    // The local files API (web/server/main.py): checkpoints, ONNX exports, replays.
    proxy: {
      '/api': 'http://localhost:8000',
    },
    // rules/map.json and rules/cards.json are bundled from outside web/ui.
    fs: { allow: ['../..'] },
  },
  build: {
    outDir: 'dist',
    // Two pages: the workbench, and the Elo leaderboard (its data, public/leaderboard.json, is
    // fitted from leaderboard/ by tools/scripts/build_web.sh).
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        leaderboard: resolve(__dirname, 'leaderboard.html'),
      },
    },
  },
});
