import * as echarts from "echarts/core";
import { PieChart, BarChart, GraphChart, RadarChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  RadarComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  PieChart,
  BarChart,
  GraphChart,
  RadarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  RadarComponent,
  CanvasRenderer,
]);

/** SAP semantic color palette, aligned with --sap* CSS custom properties. */
export const C = Object.freeze({
  positive: "#256f3a",
  positiveLight: "#eaf6ed",
  negative: "#aa0808",
  negativeLight: "#ffeaf4",
  critical: "#8d5e00",
  criticalLight: "#fff3e1",
  information: "#004f9e",
  informationLight: "#e5f0fa",
  neutral: "#475e75",
  neutralLight: "#eef1f4",
  border: "#e5e5e5",
  label: "#556b82",
  // Semantic mapping for part actions
  newPart: "#004f9e",
  phaseOut: "#aa0808",
  revised: "#8d5e00",
  referenced: "#8a95a0",
});

/**
 * Initialize an ECharts instance in a container element, lazily.
 *
 * echarts.init() warns when the container has zero size, which happens for
 * charts inside tabs that start hidden. So we defer the actual init until the
 * container first has real dimensions, driven by a ResizeObserver (which fires
 * when a hidden tab is shown). Once initialized, the observer keeps the chart
 * sized to its container.
 *
 * Returns the chart instance if it could be initialized synchronously (the
 * container was already visible), otherwise null; call disposeChart(container)
 * to clean up regardless.
 */
export function initChart(container, option) {
  let chart = null;
  const hasSize = () => container.clientWidth > 0 && container.clientHeight > 0;
  const ensure = () => {
    if (!chart) {
      if (!hasSize()) return;
      chart = echarts.init(container, null, { renderer: "canvas" });
      chart.setOption(option);
    } else {
      chart.resize();
    }
  };
  const observer = new ResizeObserver(ensure);
  observer.observe(container);
  ensure();
  container._echartsCleanup = () => {
    observer.disconnect();
    chart?.dispose();
  };
  return chart;
}

/** Dispose an ECharts instance and its ResizeObserver by container element. */
export function disposeChart(container) {
  container._echartsCleanup?.();
  delete container._echartsCleanup;
}

/** Return the common shared tooltip formatter. */
function tooltipFormatter(params) {
  const p = Array.isArray(params) ? params[0] : params;
  return `<b>${p.name}</b>: ${p.value}`;
}

/**
 * Build the ECharts option for the parts distribution donut.
 * data: [{ name, value, color }]
 */
export function partsDonutOption(data, total) {
  return {
    tooltip: { trigger: "item", formatter: tooltipFormatter },
    legend: {
      orient: "vertical",
      right: "5%",
      top: "center",
      textStyle: { color: C.neutral, fontSize: 12 },
      icon: "circle",
    },
    graphic: [
      {
        type: "text",
        left: "26%",
        top: "center",
        style: {
          text: String(total),
          textAlign: "center",
          fill: C.neutral,
          fontSize: 22,
          fontWeight: "bold",
        },
      },
      {
        type: "text",
        left: "26%",
        top: "57%",
        style: {
          text: "parts",
          textAlign: "center",
          fill: C.label,
          fontSize: 11,
        },
      },
    ],
    series: [
      {
        type: "pie",
        radius: ["38%", "58%"],
        center: ["28%", "50%"],
        avoidLabelOverlap: false,
        label: { show: false },
        emphasis: {
          label: { show: true, fontSize: 13, fontWeight: "bold" },
        },
        data: data.map((d) => ({
          name: d.name,
          value: d.value,
          itemStyle: { color: d.color },
        })),
      },
    ],
  };
}

/**
 * Build the ECharts option for a plant change-type mini horizontal bar.
 * counts: { REPLACED, ADDED, REMOVED, QUANTITY_CHANGED, VALIDITY_CHANGED }
 */
export function plantMiniBarOption(counts) {
  const labels = ["Replaced", "Added", "Removed", "Qty change"];
  const values = [
    counts?.REPLACED || 0,
    counts?.ADDED || 0,
    counts?.REMOVED || 0,
    (counts?.QUANTITY_CHANGED || 0) + (counts?.VALIDITY_CHANGED || 0),
  ];
  const colors = [C.critical, C.positive, C.negative, C.information];
  return {
    grid: { top: 4, bottom: 4, left: 70, right: 24 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "none" },
      formatter: (params) => params.map((p) => `${p.name}: ${p.value ?? 0}`).join("<br>"),
    },
    xAxis: { type: "value", show: false },
    yAxis: {
      type: "category",
      data: labels,
      axisLabel: { fontSize: 11, color: C.label },
      axisTick: { show: false },
      axisLine: { show: false },
    },
    // One series with per-bar colours; a sparse multi-series chart renders a
    // nonsensical "undefined" for every row it has no value for under an axis tooltip.
    series: [{
      type: "bar",
      data: values.map((value, i) => ({ value, itemStyle: { color: colors[i], borderRadius: 2 } })),
      barMaxWidth: 10,
      label: {
        show: true,
        position: "right",
        fontSize: 10,
        color: C.neutral,
        formatter: (p) => (p.value ? p.value : ""),
      },
    }],
  };
}

/**
 * Build the ECharts option for a procurement readiness stacked horizontal bar.
 * plants: [{ name, priced, missing, notNeeded }]
 */
export function procurementBarOption(plants) {
  const names = plants.map((p) => p.name);
  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: {
      data: ["Has price", "Missing price", "Not needed"],
      textStyle: { color: C.neutral, fontSize: 12 },
      bottom: 0,
    },
    grid: { top: 8, bottom: 40, left: 80, right: 16, containLabel: false },
    xAxis: { type: "value", minInterval: 1, axisLabel: { fontSize: 11 } },
    yAxis: {
      type: "category",
      data: names,
      axisLabel: { fontSize: 11, color: C.label },
      axisTick: { show: false },
      axisLine: { lineStyle: { color: C.border } },
    },
    series: [
      {
        name: "Has price",
        type: "bar",
        stack: "procurement",
        data: plants.map((p) => p.priced),
        itemStyle: { color: C.positive },
        label: { show: true, color: "#fff", fontSize: 10, formatter: (v) => v.value || "" },
      },
      {
        name: "Missing price",
        type: "bar",
        stack: "procurement",
        data: plants.map((p) => p.missing),
        itemStyle: { color: C.negative },
        label: { show: true, color: "#fff", fontSize: 10, formatter: (v) => v.value || "" },
      },
      {
        name: "Not needed",
        type: "bar",
        stack: "procurement",
        data: plants.map((p) => p.notNeeded),
        itemStyle: { color: C.neutralLight, borderColor: C.border },
        label: { show: false },
      },
    ],
  };
}

/**
 * Build the ECharts option for the supply chain flow graph.
 * nodes: [{ id, name, category, value, symbolSize }]
 * edges: [{ source, target, label }]
 * categories: [{ name, itemStyle }]
 */
export function supplyChainGraphOption(nodes, edges, categories) {
  return {
    tooltip: {
      formatter: (p) =>
        p.dataType === "node"
          ? `<b>${p.data.name}</b><br>${p.data.detail || ""}`
          : `${p.data.source} → ${p.data.target}`,
    },
    legend: {
      data: categories.map((c) => c.name),
      bottom: 0,
      textStyle: { color: C.neutral, fontSize: 11 },
    },
    series: [
      {
        type: "graph",
        layout: "force",
        animation: true,
        animationDuration: 600,
        roam: true,
        draggable: true,
        zoom: 0.9,
        force: {
          repulsion: 200,
          gravity: 0.1,
          edgeLength: [80, 160],
        },
        categories,
        nodes: nodes.map((n) => ({
          id: n.id,
          name: n.name,
          category: n.category,
          symbolSize: n.symbolSize || 28,
          detail: n.detail,
          label: { show: true, fontSize: 10, overflow: "break", width: 80 },
          emphasis: { scale: true },
        })),
        edges: edges.map((e) => ({
          source: e.source,
          target: e.target,
          lineStyle: { color: C.border, curveness: 0.1, type: e.dashed ? "dashed" : "solid" },
          label: { show: !!e.label, fontSize: 9, color: C.label, formatter: e.label || "" },
        })),
        edgeSymbol: ["none", "arrow"],
        edgeSymbolSize: 8,
      },
    ],
  };
}

/** Build a five-factor, 1–5 comparison of the two sourcing paths. */
export function makeBuyRadarOption(scores) {
  const indicators = ["Cost", "Lead time", "Impacted plants", "Historical trend", "Quality"];
  return {
    tooltip: {
      formatter: (params) => {
        const point = Array.isArray(params) ? params[0] : params;
        return `<b>${point.name}</b><br>${indicators.map((label, index) => `${label}: ${point.value[index]} / 5`).join("<br>")}`;
      },
    },
    legend: {
      data: ["Make", "Buy"],
      bottom: 0,
      textStyle: { color: C.neutral, fontSize: 12 },
    },
    radar: {
      center: ["50%", "47%"],
      radius: "62%",
      splitNumber: 5,
      indicator: indicators.map((name) => ({ name, max: 5 })),
      axisName: { color: C.neutral, fontSize: 11 },
      splitLine: { lineStyle: { color: C.border } },
      splitArea: { areaStyle: { color: ["#fff", "#f7f8f9"] } },
      axisLine: { lineStyle: { color: C.border } },
    },
    series: [{
      type: "radar",
      symbol: "circle",
      symbolSize: 5,
      data: [
        {
          name: "Make",
          value: scores.make,
          lineStyle: { color: C.positive, width: 2 },
          itemStyle: { color: C.positive },
          areaStyle: { color: "rgb(37 111 58 / 30%)" },
        },
        {
          name: "Buy",
          value: scores.buy,
          lineStyle: { color: C.information, width: 2 },
          itemStyle: { color: C.information },
          areaStyle: { color: "rgb(0 79 158 / 28%)" },
        },
      ],
    }],
  };
}
