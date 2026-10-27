import assert from "node:assert/strict";
import { parseRolloutParam, rolloutParam, specEqual } from "../../../web/ui/src/analysis/rollout_spec";

// keyed2 implies keyed: spelled with or without `:keyed`, it is one search and writes back one way.
const alone = parseRolloutParam("rollout:4:32:4:z2:earlier:own:keyed2")!;
const both = parseRolloutParam("rollout:4:32:4:z2:earlier:own:keyed:keyed2")!;
assert.ok(alone.keyed2 && !alone.keyed);
assert.ok(specEqual(alone, both));
assert.equal(rolloutParam(both), "rollout:4:32:4:z2:earlier:own:keyed2");
const keyed = parseRolloutParam("rollout:4:32:4:z2:earlier:own:keyed")!;
assert.ok(keyed.keyed && !keyed.keyed2 && !specEqual(keyed, alone));
assert.equal(rolloutParam(keyed), "rollout:4:32:4:z2:earlier:own:keyed");

// race<n> races the worlds from the first n; it must be below the worlds, and a spec without it is unraced.
const raced = parseRolloutParam("rollout:4:32:4:z2:earlier:own:keyed:aligned:box4:belief2048:race16")!;
assert.equal(raced.race, 16);
assert.equal(rolloutParam(raced), "rollout:4:32:4:z2:earlier:own:keyed:aligned:box4:belief2048:race16");
assert.ok(!specEqual(raced, { ...raced, race: 0 }) && keyed.race === 0);
assert.equal(parseRolloutParam("rollout:4:32:4:z2:earlier:own:race32"), null);

console.log(JSON.stringify({ canonical: true }));
