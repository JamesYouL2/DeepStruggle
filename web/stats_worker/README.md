# The stats database (`web/stats_worker/`)

Games people play against the models in the workbench, for estimating the models' strength on
Playdek's rating scale. The page (`web/ui/src/result_dialog.ts`) POSTs a finished game here; this
Cloudflare Worker checks it and keeps it in a D1 (SQLite) table; `tools/game_stats.py` exports the
table and does the rest (`tools/README.md` §2d). Nothing is self-hosted: the Worker and the
database run on Cloudflare's free plan (100k requests and 100k rows written a day, 5 GB).

* `POST /games` -- one submission (`web/ui/src/game/record.ts` `Submission`), at most 64 KB.
  Answers `{"ok": true, "stored": true}`, or `"stored": false` for an id it has already (a
  double click, a retried request). 400 for a malformed game, 403 for an origin not in
  `ALLOWED_ORIGINS`, 429 past `RATE_PER_HOUR` submissions from one address.
* `GET /health` -- `{"ok": true}`.
* Nothing reads games back over HTTP: the Playdek logins stay with the database's owner.

The checks are against junk and accidents, not cheating -- anyone can post anything, and
re-driving a game in the engine (`tools/game_stats.py verify`) is what tells a game from a story.
The client's address is kept only as a salted hash, for the rate limit.

## Setting it up (once)

Needs a Cloudflare account and node. From this directory:

```bash
npm install
npx wrangler login
npx wrangler d1 create ts-game-stats          # put the printed database_id into wrangler.toml
npx wrangler d1 execute ts-game-stats --remote --file schema.sql
npx wrangler secret put IP_SALT               # any random string
npx wrangler deploy                           # prints the Worker's URL
```

Then point the page at it: set the repository variable `TS_STATS_ENDPOINT` (Settings → Secrets and
variables → Actions → Variables) to the Worker's URL, which `.github/workflows/pages.yml` passes
to the build; locally, `TS_STATS_ENDPOINT=<url> tools/scripts/build_web.sh`. A build without it
shows the dialog with *Download record* only. `ALLOWED_ORIGINS` in `wrangler.toml` lists the
page's origins (GitHub Pages and the local servers); add any other place the page is served from.

## Reading it

```bash
PYTHONPATH=. python tools/game_stats.py fetch --output data/game_stats/games.json
npx wrangler d1 execute ts-game-stats --remote --command \
    "SELECT received_at, playdek_login, playdek_rating, human_side, model_label, search, winner, clean FROM games"
```

The columns besides `body` are copies of the submission's fields, for queries like the second;
`body` is the submission exactly as received, and is what the tools read. `src/handler.ts` is
tested under node with a fake D1 by `tests/web/test_game_stats.py`.
