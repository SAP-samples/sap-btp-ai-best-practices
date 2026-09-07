import assert from "node:assert/strict";
import test from "node:test";

import {
  buildNestedBarChartConfig,
  buildScoreViewModel,
  qualitativeScoreMarkup,
  scoreAriaDescription
} from "../src/pages/score/scoreViewModel.js";

function dimension(index, positioning = "in_line_with_peers") {
  return {
    dimension: `Dimension ${index}`,
    display_name: `Dimension ${index}`,
    score: 62.3456 + index,
    benchmark: {
      company_score: 62.3456 + index,
      peer_average: 70.1234,
      best_peer: 88.9876,
      sample_size: 4,
      positioning,
      commentary: "In line with the peer average (3% below)."
    }
  };
}

test("score view keeps all seven dimensions and only applicable question topics", () => {
  const dimensions = Array.from({ length: 7 }, (_, index) => dimension(index + 1));
  const payload = {
    dimensions,
    questions: [
      {
        question_id: "Q1",
        dimension: "Dimension 1",
        topic_title: "A deliberately very long topic title that must remain intact on mobile",
        applicable: true,
        score: 55.4321,
        benchmark: dimensions[0].benchmark
      },
      {
        question_id: "Q2",
        dimension: "Dimension 1",
        topic_title: "Excluded topic",
        applicable: false,
        score: 10,
        benchmark: dimensions[0].benchmark
      }
    ]
  };

  const view = buildScoreViewModel(payload);
  assert.equal(view.dimensions.length, 7);
  assert.equal(view.dimensions[0].topics.length, 1);
  assert.match(view.dimensions[0].topics[0].label, /very long topic title/);
  assert.doesNotMatch(view.dimensions[0].topics[0].label, /Excluded/);
});

test("nested bar configuration fixes the internal scale and disables numeric disclosure", () => {
  const config = buildNestedBarChartConfig([dimension(1)], {
    company: "Company",
    peerAverage: "Peer average",
    bestPeer: "Best peer"
  });

  assert.equal(config.options.indexAxis, "y");
  assert.equal(config.options.scales.x.min, 0);
  assert.equal(config.options.scales.x.max, 100);
  assert.equal(config.options.scales.x.display, false);
  assert.equal(config.options.scales.x.stacked, false);
  assert.equal(config.options.scales.y.stacked, false);
  assert.equal(config.options.plugins.tooltip.enabled, false);
  assert.equal(config.options.plugins.legend.display, false);
  assert.deepEqual(
    config.data.datasets.map((dataset) => dataset.barThickness),
    [42, 28, 14]
  );
});

test("qualitative score markup and ARIA never include exact scores", () => {
  const item = dimension(1, "above_peers");
  item.label = "Strategy";
  item.display_name = "Strategy";
  const items = [item];
  const translate = (key) => ({
    "score.position.above_peers": "Above peers",
    "score.position.unavailable": "Benchmark unavailable"
  })[key] || key;

  const markup = qualitativeScoreMarkup(items, translate);
  const aria = scoreAriaDescription(items, translate);
  for (const exactValue of ["63.3456", "70.1234", "88.9876"]) {
    assert.doesNotMatch(markup, new RegExp(exactValue.replace(".", "\\.")));
    assert.doesNotMatch(aria, new RegExp(exactValue.replace(".", "\\.")));
  }
  assert.match(markup, /Above peers/);
  assert.match(markup, /3% below/);
  assert.match(aria, /Above peers/);
  assert.doesNotMatch(aria, /\d/);
});

test("unavailable benchmarks remain textual and do not synthesize peer values", () => {
  const item = dimension(1, "unavailable");
  item.benchmark.peer_average = null;
  item.benchmark.best_peer = null;
  item.benchmark.sample_size = 0;
  const view = buildScoreViewModel({ dimensions: [item], questions: [] });

  assert.equal(view.dimensions[0].benchmark.positioning, "unavailable");
  assert.equal(view.dimensions[0].benchmark.peer_average, null);
  assert.equal(view.dimensions[0].benchmark.best_peer, null);
  assert.equal("sample_size" in view.dimensions[0].benchmark, false);
});
