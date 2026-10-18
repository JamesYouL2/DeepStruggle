import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';

function git(...args: string[]): string {
  try {
    return execFileSync('git', args, { cwd: __dirname, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return '';
  }
}

// The commit the page is built from, for the stats database (src/build_info.ts): a submitted game
// is rebuilt at it. Dirty: tracked files differ from it, so it alone may not rebuild the game.
const commit = git('rev-parse', 'HEAD');
const dirty = commit !== '' && git('status', '--porcelain', '--untracked-files=no') !== '';

// TS_WEB_BASE: the path the page is served under -- "/" locally, "/<repo>/" on GitHub Pages
// (set by .github/workflows/pages.yml). The engine and model runtime are loaded relative to it.
// TS_STATS_ENDPOINT: the stats database's Worker (web/stats_worker/); unset, games can only be
// downloaded.
export default defineConfig({
  base: process.env.TS_WEB_BASE || '/',
  define: {
    'import.meta.env.VITE_TS_BUILD_COMMIT': JSON.stringify(commit),
    'import.meta.env.VITE_TS_BUILD_DIRTY': JSON.stringify(dirty ? '1' : ''),
    'import.meta.env.VITE_TS_STATS_ENDPOINT': JSON.stringify(process.env.TS_STATS_ENDPOINT || ''),
  },
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
