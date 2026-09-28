// Bundled by tests/web/test_value_readings.py: which critic reading the page believes, under node.
//   node value_readings.mjs <case.json>   with {steps}: a small replay's steps
// Prints, for every step, the decider trace_view.ts finds, its calibrated P(US wins), and the
// classes/numbers the value display renders -- the Python side checks them against the rules.
import { readFileSync } from "node:fs";
import {
  calibratedSeries, criticTableHtml, policyChipHtml, replayDecider, valueSidesHtml,
} from "../../../web/ui/src/trace_view";
import { ReplayStep } from "../../../web/ui/src/replay_controls";

const { steps } = JSON.parse(readFileSync(process.argv[2], "utf8")) as { steps: ReplayStep[] };

const attr = (html: string, re: RegExp): string[] => [...html.matchAll(re)].map(m => m[1]);

const series = calibratedSeries(steps);
const out = steps.map((s, i) => {
  const decider = replayDecider(steps, i);
  const terminal = s.state_snapshot?.is_terminal ? s.state_snapshot : null;
  const html = s.critic ? valueSidesHtml(s.critic, decider, terminal) : "";
  const table = s.critic ? criticTableHtml(s.critic, decider) : "";
  return {
    decider,
    p_us: series[i]?.pUs ?? null,
    p_us_other: series[i]?.pUsOther ?? null,
    calibrated: attr(html, /class="value-side \w+ calibrated" data-side="(\w+)"/g),
    uncalibrated: attr(html, /class="value-side \w+ uncalibrated" data-side="(\w+)"/g),
    bar_p_us: attr(html, /data-p-us="([\d.]+)"/g).map(Number)[0] ?? null,
    bar_vp: attr(html, /class="value-bar-vp">([^<]*)</g)[0] ?? null,
    side_vp: attr(html, /VP ([+−][\d.]+)<\/div>/g),
    result: attr(html, /class="value-result"[^>]*>([^<]*)</g)[0] ?? null,
    table_calibrated: attr(table, /<tr class="row-\w+ calibrated" data-side="(\w+)"/g),
    chip: policyChipHtml(steps, i).match(/ΔP [+−]\d+%/)?.[0] ?? null,
  };
});
console.log(JSON.stringify(out));
