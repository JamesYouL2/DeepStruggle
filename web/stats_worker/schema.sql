-- The stats database (Cloudflare D1): one row per game submitted from the workbench page.
-- `body` is the submission exactly as received (web/ui/src/game/record.ts `Submission`), which is
-- what tools/game_stats.py reads; the other columns are copies of its fields, for queries.
--
--   npx wrangler d1 execute ts-game-stats --remote --file schema.sql
CREATE TABLE IF NOT EXISTS games (
  id             TEXT PRIMARY KEY,
  received_at    TEXT NOT NULL,
  schema         INTEGER NOT NULL,
  commit_sha     TEXT NOT NULL,
  dirty          INTEGER NOT NULL,
  engine_fp      TEXT NOT NULL,
  playdek_login  TEXT NOT NULL,
  playdek_rating INTEGER NOT NULL,
  human_side     TEXT,
  model_sha256   TEXT NOT NULL,
  model_label    TEXT NOT NULL,
  search         TEXT NOT NULL,
  winner         TEXT NOT NULL,
  margin         INTEGER NOT NULL,
  end_turn       INTEGER NOT NULL,
  reason         TEXT NOT NULL,
  n_steps        INTEGER NOT NULL,
  clean          INTEGER NOT NULL,
  ip_hash        TEXT NOT NULL,
  body           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS games_by_ip ON games (ip_hash, received_at);
CREATE INDEX IF NOT EXISTS games_by_model ON games (model_sha256, search);
