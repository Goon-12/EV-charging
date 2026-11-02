"use strict";

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */
const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const FAMILY = {
  metaheuristic: { color: "#1d4ed8", label: "Metaheuristic" },
  rolling: { color: "#0891b2", label: "Rolling horizon" },
  conventional: { color: "#ea580c", label: "Conventional policy" },
  priority: { color: "#64748b", label: "Priority vector" },
};

const METHOD_COLORS = {
  "GA": "#1d4ed8",
  "PSO": "#7c3aed",
  "GA rolling horizon": "#0891b2",
  "PSO rolling horizon": "#0e7490",
  "EDF (chronological)": "#ea580c",
  "FCFS (chronological)": "#b45309",
  "EDF-priority (repair-based)": "#64748b",
  "FCFS-priority (repair-based)": "#94a3b8",
  "Uncontrolled (infeasible reference)": "#dc2626",
};

const state = {
  overview: null,
  tab: "overview",
  schedules: {},
  pending: {},
  selection: new Set(["EDF (chronological)", "FCFS (chronological)", "GA", "PSO"]),
  currentMethod: null,
  heat: null,
  live: null,
  liveMethods: [],
};

const tooltip = $("#tooltip");

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function signed(value, digits = 1) {
  if (value === null || value === undefined) return "—";
  const n = Number(value);
  return `${n >= 0 ? "+" : ""}${fmt(n, digits)}`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function methodStyle(name) {
  const color = METHOD_COLORS[name] || "#334155";
  let dash = null;
  if (name.includes("rolling")) dash = "7 5";
  else if (name.includes("priority")) dash = "2 5";
  else if (name.includes("Uncontrolled")) dash = "2 5";
  return { color, dash };
}

function shortName(name) {
  return name
    .replace(" (chronological)", "")
    .replace(" (repair-based)", " (prio)")
    .replace(" (infeasible reference)", " (ref)")
    .replace(" rolling horizon", " RH");
}

async function api(path, options) {
  const response = await fetch(path, options);
  let body;
  try {
    body = await response.json();
  } catch (error) {
    body = { error: `HTTP ${response.status}` };
  }
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function svgEl(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  if (parent) parent.appendChild(node);
  return node;
}

function ticks(min, max, count = 5) {
  const span = (max - min) / count;
  if (!(span > 0)) return [min];
  const power = Math.pow(10, Math.floor(Math.log10(span)));
  const error = span / power;
  const mult = error >= 7.5 ? 10 : error >= 3.5 ? 5 : error >= 1.5 ? 2 : 1;
  const step = power * mult;
  const start = Math.ceil(min / step) * step;
  const out = [];
  for (let value = start; value <= max + 1e-9; value += step) {
    out.push(Number(value.toFixed(6)));
  }
  return out;
}

function base64Bytes(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function powerColor(t) {
  const stops = [
    [219, 234, 254],
    [59, 130, 246],
    [30, 58, 138],
  ];
  const clamped = Math.max(0, Math.min(1, t));
  const scaled = clamped * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const fraction = scaled - index;
  const a = stops[index];
  const b = stops[index + 1];
  const channel = (i) => Math.round(a[i] + (b[i] - a[i]) * fraction);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

function showTip(html, x, y) {
  tooltip.innerHTML = html;
  tooltip.hidden = false;
  const rect = tooltip.getBoundingClientRect();
  const pad = 14;
  let left = x + pad;
  let top = y + pad;
  if (left + rect.width > window.innerWidth - 8) left = x - rect.width - pad;
  if (top + rect.height > window.innerHeight - 8) top = y - rect.height - pad;
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function hideTip() {
  tooltip.hidden = true;
}

function tipRow(label, value, color) {
  const swatch = color
    ? `<i class="tip-swatch" style="background:${color}"></i>`
    : "";
  return `<div class="tip-row"><span>${swatch}${escapeHtml(label)}</span><b>${value}</b></div>`;
}

function renderLegend(host, entries) {
  host.innerHTML = "";
  for (const entry of entries) {
    const item = document.createElement("span");
    item.className = "legend-item";
    const mark = document.createElement("span");
    if (entry.line) {
      mark.className = "legend-line";
      mark.style.borderTopColor = entry.color;
      mark.style.borderTopStyle = entry.dash ? "dashed" : "solid";
    } else {
      mark.className = "swatch";
      mark.style.background = entry.color;
    }
    item.append(mark, document.createTextNode(entry.label));
    host.appendChild(item);
  }
}

/* ------------------------------------------------------------------ */
/* Tabs                                                                */
/* ------------------------------------------------------------------ */
function initTabs() {
  $$(".tab").forEach((button) => {
    button.addEventListener("click", () => selectTab(button.dataset.tab));
  });
}

function selectTab(name) {
  state.tab = name;
  $$(".tab").forEach((button) => {
    const active = button.dataset.tab === name;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  $$(".panel").forEach((panel) => {
    const active = panel.id === `panel-${name}`;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  });
  if (name === "profiles") renderProfiles();
  if (name === "schedule") renderScheduleTab();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* ------------------------------------------------------------------ */
/* Overview                                                            */
/* ------------------------------------------------------------------ */
function renderOverview() {
  const ov = state.overview;
  const h = ov.headline;

  $("#brand-sub").textContent =
    `${ov.site.n_ev} EVs · ${ov.site.n_chargers} ports · ${fmt(ov.site.max_grid_kw, 0)} kW grid · ${fmt(ov.site.horizon_hours, 0)} h horizon`;

  $("#hero-required").textContent = fmt(ov.site.required_kwh, 1);
  $("#hero-facts").innerHTML = [
    [ov.site.n_ev, "EVs"],
    [ov.site.n_chargers, "ports"],
    [`${fmt(ov.site.max_grid_kw, 0)} kW`, "grid contract"],
    [`${fmt(ov.site.horizon_hours, 0)} h`, "horizon"],
    [`${fmt(ov.site.step_hours * 60, 0)} min`, "per slot"],
  ]
    .map(([value, label]) => `<span class="fact"><b>${value}</b>${label}</span>`)
    .join("");

  const pressure = ov.port_pressure;
  const ratio = pressure.port_hours_needed / pressure.port_hours_available;
  $("#port-gauge").innerHTML = `
    <div class="gauge">
      <div class="gauge-row">
        <span class="gauge-label">Supply</span>
        <div class="gauge-track"><div class="gauge-fill supply" style="width:${(pressure.port_hours_available / pressure.port_hours_needed) * 100}%">${fmt(pressure.port_hours_available, 0)} h</div></div>
      </div>
      <div class="gauge-row">
        <span class="gauge-label">Demand</span>
        <div class="gauge-track"><div class="gauge-fill demand" style="width:100%">${fmt(pressure.port_hours_needed, 0)} h</div></div>
      </div>
    </div>
    <p class="gauge-note" style="margin-top:12px">
      The sessions need <strong>${fmt(ratio, 2)}×</strong> the port-hours the site can supply at rated power.
      Ports bind, not the grid.
    </p>`;

  const cards = [
    {
      label: "PSO delivered",
      value: `${fmt(h.pso.delivered_kwh, 1)}<small>kWh</small>`,
      sub: `${signed(h.pso.delta_vs_conventional_pct)}% vs ${h.best_conventional.method.replace(" (chronological)", "")}`,
      tone: "good",
      accent: "#1d4ed8",
    },
    {
      label: "GA delivered",
      value: `${fmt(h.ga.delivered_kwh, 1)}<small>kWh</small>`,
      sub: `${signed(h.ga.delta_vs_conventional_pct)}% vs the best conventional policy`,
      tone: "good",
      accent: "#7c3aed",
    },
    {
      label: "Share of certified optimum",
      value: `${fmt(h.pso.pct_of_optimum, 1)}<small>%</small>`,
      sub: `MILP ceiling ${fmt(h.reference_kwh, 1)} kWh · gap ${fmt(h.reference_gap_pct, 2)}%`,
      tone: "",
      accent: "#0f766e",
    },
    {
      label: "Seeded runs beating EDF",
      value: `${h.seeded_runs.beating_conventional}<small> / ${h.seeded_runs.total}</small>`,
      sub: `worst run ${signed(h.seeded_runs.min_margin_pct)}% · ${h.violations} constraint violations`,
      tone: "good",
      accent: "#ea580c",
    },
  ];

  $("#kpis").innerHTML = cards
    .map(
      (card) => `
      <article class="kpi" style="--kpi-accent:${card.accent}">
        <span class="kpi-label">${card.label}</span>
        <span class="kpi-value">${card.value}</span>
        <span class="kpi-sub ${card.tone}">${card.sub}</span>
      </article>`
    )
    .join("");

  renderServiceChart();
  renderScatterChart();
  renderTradeoff();
  renderVerdict();
}

function renderVerdict() {
  const ov = state.overview;
  const h = ov.headline;
  $("#verdict").innerHTML = `
    <strong>Verdict.</strong> At the committed budget both optimizers beat every
    conventional policy: PSO delivers ${fmt(h.pso.delivered_kwh, 1)} kWh
    (${signed(h.pso.delta_vs_conventional_pct)}%) and GA ${fmt(h.ga.delivered_kwh, 1)} kWh
    (${signed(h.ga.delta_vs_conventional_pct)}%) against ${fmt(h.best_conventional.delivered_kwh, 1)} kWh
    for ${h.best_conventional.method}. PSO reaches ${fmt(h.pso.pct_of_optimum, 1)}% of the MILP
    optimum (${fmt(h.reference_kwh, 1)} kWh, MIP gap ${fmt(h.reference_gap_pct, 2)}%) — a
    denominator that was solved, not asserted. GA and PSO used
    ${fmt(h.ga.evaluations, 0)} and ${fmt(h.pso.evaluations, 0)} evaluations respectively, and
    ${h.seeded_runs.beating_conventional} of ${h.seeded_runs.total} seeded runs clear EDF.
  `;
}

function renderServiceChart() {
  const ov = state.overview;
  const host = $("#chart-service");
  host.innerHTML = "";
  renderLegend($("#service-legend"), [
    { color: FAMILY.metaheuristic.color, label: "GA / PSO" },
    { color: FAMILY.rolling.color, label: "Rolling horizon" },
    { color: FAMILY.conventional.color, label: "FCFS / EDF" },
    { color: FAMILY.priority.color, label: "Priority vector" },
    { color: "#ea580c", label: `Best conventional (${fmt(ov.headline.best_conventional.delivered_kwh, 1)} kWh)`, line: true, dash: true },
    { color: "#0f172a", label: `MILP optimum (${fmt(ov.headline.reference_kwh, 1)} kWh)`, line: true, dash: true },
  ]);

  const rows = ov.methods
    .filter((row) => row.feasible && !String(row.method).startsWith("Uncontrolled"))
    .sort((a, b) => a.service_delivered_kwh - b.service_delivered_kwh);

  const W = 1000;
  const rowH = 34;
  const M = { l: 218, r: 150, t: 10, b: 34 };
  const H = M.t + M.b + rows.length * rowH;
  const xMax = Math.ceil((ov.headline.reference_kwh * 1.2) / 100) * 100;
  const plotW = W - M.l - M.r;
  const x = (value) => M.l + (value / xMax) * plotW;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Delivered energy by method" });
  host.appendChild(svg);

  for (const tick of ticks(0, xMax, 5)) {
    svgEl("line", { x1: x(tick), y1: M.t, x2: x(tick), y2: H - M.b, stroke: "#eef2f7", "stroke-width": 1 }, svg);
    svgEl("text", { x: x(tick), y: H - M.b + 18, "text-anchor": "middle", "font-size": 11.5, fill: "#64748b" }, svg)
      .textContent = fmt(tick, 0);
  }
  svgEl("text", { x: M.l + plotW / 2, y: H - 2, "text-anchor": "middle", "font-size": 11.5, fill: "#64748b" }, svg)
    .textContent = "delivered energy (kWh)";

  rows.forEach((row, index) => {
    const y = M.t + index * rowH + 5;
    const height = rowH - 10;
    const width = Math.max(2, x(row.service_delivered_kwh) - M.l);
    const family = FAMILY[row.family] || FAMILY.conventional;

    const label = svgEl("text", {
      x: M.l - 12, y: y + height / 2 + 4, "text-anchor": "end",
      "font-size": 12.5, fill: "#0f172a",
    }, svg);
    label.textContent = shortName(row.method);

    svgEl("rect", {
      x: M.l, y, width, height, rx: 4,
      fill: family.color, "fill-opacity": row.method.startsWith("EDF-priority") || row.method.startsWith("FCFS-priority") ? 0.55 : 0.9,
    }, svg);

    svgEl("text", {
      x: M.l + width + 8, y: y + height / 2 + 4,
      "font-size": 11.5, fill: "#334155",
    }, svg).textContent =
      `${fmt(row.service_delivered_kwh, 1)}  ·  ${fmt(row.pct_of_best_service, 1)}%`;

    const hit = svgEl("rect", {
      x: 0, y: M.t + index * rowH, width: W, height: rowH,
      fill: "transparent", "pointer-events": "all",
    }, svg);
    hit.addEventListener("mousemove", (event) => {
      showTip(
        `<div class="tip-title">${escapeHtml(row.method)}</div>` +
          tipRow("Delivered", `${fmt(row.service_delivered_kwh, 1)} kWh`) +
          tipRow("Share of attainable", `${fmt(row.pct_of_best_service, 1)}%`) +
          tipRow("Cost / kWh", fmt(row.cost_per_delivered_kwh, 4)) +
          tipRow("Peak", `${fmt(row.peak_ev_load_kw, 1)} kW`) +
          tipRow("Fully charged EVs", fmt(row.fully_charged_evs, 0)),
        event.clientX,
        event.clientY
      );
    });
    hit.addEventListener("mouseleave", hideTip);
  });

  const bestConventional = ov.headline.best_conventional.delivered_kwh;
  svgEl("line", {
    x1: x(bestConventional), y1: M.t, x2: x(bestConventional), y2: H - M.b,
    stroke: "#ea580c", "stroke-width": 1.6, "stroke-dasharray": "6 4",
  }, svg);
  svgEl("line", {
    x1: x(ov.headline.reference_kwh), y1: M.t, x2: x(ov.headline.reference_kwh), y2: H - M.b,
    stroke: "#0f172a", "stroke-width": 1.6, "stroke-dasharray": "2 4",
  }, svg);
}

function renderScatterChart() {
  const ov = state.overview;
  const host = $("#chart-scatter");
  host.innerHTML = "";

  const rows = ov.methods.filter(
    (row) =>
      row.feasible &&
      !String(row.method).startsWith("Uncontrolled") &&
      Number.isFinite(Number(row.service_pct_of_attainable)) &&
      Number.isFinite(Number(row.cost_per_delivered_kwh))
  );

  const W = 640;
  const H = 400;
  const M = { l: 62, r: 24, t: 16, b: 50 };
  const plotW = W - M.l - M.r;
  const plotH = H - M.t - M.b;

  const xs = rows.map((row) => Number(row.service_pct_of_attainable));
  const ys = rows.map((row) => Number(row.cost_per_delivered_kwh));
  const xMin = Math.max(55, Math.min(...xs) - 4);
  const xMax = Math.max(...xs) + 2;
  const yMin = Math.min(...ys) - 0.18;
  const yMax = Math.max(...ys) + 0.18;

  const sx = (value) => M.l + ((value - xMin) / (xMax - xMin)) * plotW;
  const sy = (value) => M.t + plotH - ((value - yMin) / (yMax - yMin)) * plotH;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Cost per delivered kWh against service attained" });
  host.appendChild(svg);

  for (const tick of ticks(xMin, xMax, 5)) {
    svgEl("line", { x1: sx(tick), y1: M.t, x2: sx(tick), y2: M.t + plotH, stroke: "#eef2f7" }, svg);
    svgEl("text", { x: sx(tick), y: M.t + plotH + 18, "text-anchor": "middle", "font-size": 11, fill: "#64748b" }, svg)
      .textContent = `${fmt(tick, 0)}%`;
  }
  for (const tick of ticks(yMin, yMax, 5)) {
    svgEl("line", { x1: M.l, y1: sy(tick), x2: M.l + plotW, y2: sy(tick), stroke: "#eef2f7" }, svg);
    svgEl("text", { x: M.l - 9, y: sy(tick) + 4, "text-anchor": "end", "font-size": 11, fill: "#64748b" }, svg)
      .textContent = fmt(tick, 2);
  }

  svgEl("text", { x: M.l + plotW / 2, y: H - 8, "text-anchor": "middle", "font-size": 11.5, fill: "#64748b" }, svg)
    .textContent = "service attained (% of attainable energy)";
  svgEl("text", {
    x: 14, y: M.t + plotH / 2, "text-anchor": "middle", "font-size": 11.5, fill: "#64748b",
    transform: `rotate(-90 14 ${M.t + plotH / 2})`,
  }, svg).textContent = "cost per delivered kWh (units/kWh)";

  const frontier = rows
    .filter((row) => Number.isFinite(Number(row.optimal_cost_at_service)))
    .map((row) => ({
      x: Number(row.service_pct_of_attainable),
      y: Number(row.optimal_cost_at_service) / Number(row.service_delivered_kwh),
    }))
    .sort((a, b) => a.x - b.x);

  if (frontier.length > 1) {
    svgEl("polyline", {
      points: frontier.map((point) => `${sx(point.x)},${sy(point.y)}`).join(" "),
      fill: "none", stroke: "#0f172a", "stroke-width": 1.6, "stroke-dasharray": "5 4",
    }, svg);
    const first = frontier[0];
    svgEl("text", {
      x: sx(first.x) + 6, y: sy(first.y) - 8, "font-size": 10.5, fill: "#0f172a",
    }, svg).textContent = "cheapest known at that service";
  }

  const ordered = [...rows].sort(
    (a, b) => Number(a.service_pct_of_attainable) - Number(b.service_pct_of_attainable)
  );
  ordered.forEach((row, index) => {
    const family = FAMILY[row.family] || FAMILY.conventional;
    const isFrontier = row.pareto_status === "frontier";
    const cx = sx(Number(row.service_pct_of_attainable));
    const cy = sy(Number(row.cost_per_delivered_kwh));

    svgEl("circle", {
      cx, cy, r: 6,
      fill: isFrontier ? family.color : "#ffffff",
      stroke: family.color, "stroke-width": 2,
    }, svg);

    svgEl("text", {
      x: cx + (index % 2 === 0 ? 10 : -10),
      y: cy + (index % 2 === 0 ? -6 : 14),
      "text-anchor": index % 2 === 0 ? "start" : "end",
      "font-size": 10.5, fill: "#334155",
    }, svg).textContent = shortName(row.method);

    const hit = svgEl("circle", { cx, cy, r: 14, fill: "transparent", "pointer-events": "all" }, svg);
    hit.addEventListener("mousemove", (event) => {
      showTip(
        `<div class="tip-title">${escapeHtml(row.method)}</div>` +
          tipRow("Delivered", `${fmt(row.service_delivered_kwh, 1)} kWh`) +
          tipRow("Service", `${fmt(row.service_pct_of_attainable, 1)}% of attainable`) +
          tipRow("Cost / kWh", fmt(row.cost_per_delivered_kwh, 4)) +
          tipRow("Gap vs frontier", row.cost_gap_vs_optimal_pct == null ? "—" : `${signed(row.cost_gap_vs_optimal_pct)}%`) +
          tipRow("Peak", `${fmt(row.peak_ev_load_kw, 1)} kW`),
        event.clientX,
        event.clientY
      );
    });
    hit.addEventListener("mouseleave", hideTip);
  });
}

function renderTradeoff() {
  const ov = state.overview;
  const edf = ov.methods.find((row) => row.method === "EDF (chronological)");
  const pso = ov.methods.find((row) => row.method === "PSO");
  if (!edf || !pso) return;

  const rows = [
    ["Delivered energy (kWh)", edf.service_delivered_kwh, pso.service_delivered_kwh, 1, true],
    ["Fully charged EVs", edf.fully_charged_evs, pso.fully_charged_evs, 0, false],
    ["Cost per delivered kWh", edf.cost_per_delivered_kwh, pso.cost_per_delivered_kwh, 4, false],
    ["Peak EV load (kW)", edf.peak_ev_load_kw, pso.peak_ev_load_kw, 1, false],
  ];

  $("#tradeoff").innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr><th>Metric</th><th>EDF</th><th>PSO</th><th>Δ</th></tr></thead>
        <tbody>
          ${rows
            .map(([label, a, b, digits, higherWins]) => {
              const delta = Number(b) - Number(a);
              const good = higherWins ? delta > 0 : delta < 0;
              return `<tr>
                <td>${label}</td>
                <td class="num">${fmt(a, digits)}</td>
                <td class="num">${fmt(b, digits)}</td>
                <td class="num" style="color:${good ? "#047857" : "#b91c1c"}">${signed(delta, digits)}</td>
              </tr>`;
            })
            .join("")}
        </tbody>
      </table>
    </div>
    <p class="hint">
      PSO serves ${fmt(pso.service_delivered_kwh - edf.service_delivered_kwh, 1)} kWh more while
      paying ${fmt(edf.cost_per_delivered_kwh - pso.cost_per_delivered_kwh, 4)} less per kWh, but
      it fully charges ${edf.fully_charged_evs - pso.fully_charged_evs} fewer vehicles and holds a
      ${fmt(pso.peak_ev_load_kw - edf.peak_ev_load_kw, 1)} kW higher peak. The objective says
      service first, not fairness first.
    </p>`;
}

/* ------------------------------------------------------------------ */
/* Line chart                                                          */
/* ------------------------------------------------------------------ */
function renderLineChart(host, config) {
  host.innerHTML = "";
  const {
    series,
    hours,
    solar = null,
    tariff = null,
    gridLimit = null,
    showTariff = false,
    yLabel = "power (kW)",
    xLabel = "hours from simulation start",
  } = config;

  const W = 1000;
  const H = 380;
  const M = { l: 54, r: 24, t: 16, b: 48 };
  const plotW = W - M.l - M.r;
  const plotH = H - M.t - M.b;
  const n = hours.length;

  const values = series.flatMap((entry) => entry.values.map(Number));
  const peak = Math.max(140, ...values, ...(solar || []).map(Number));
  const yMax = Math.ceil(peak / 25) * 25;
  const x = (index) => M.l + (n <= 1 ? 0 : (index / (n - 1)) * plotW);
  const y = (value) => M.t + plotH - (value / yMax) * plotH;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Load profile chart" });
  host.appendChild(svg);

  // evening tariff band
  if (showTariff && tariff) {
    let start = null;
    for (let i = 0; i < n; i += 1) {
      const isPeak = Number(tariff[i]) >= 9.9;
      if (isPeak && start === null) start = i;
      if (!isPeak && start !== null) {
        svgEl("rect", {
          x: x(start), y: M.t, width: Math.max(1, x(i) - x(start)), height: plotH,
          fill: "#fef2f2",
        }, svg);
        svgEl("text", {
          x: x(start) + 8, y: M.t + 16, "font-size": 10.5, fill: "#b91c1c",
        }, svg).textContent = "evening tariff peak";
        start = null;
      }
    }
    if (start !== null) {
      svgEl("rect", {
        x: x(start), y: M.t, width: Math.max(1, x(n - 1) - x(start)), height: plotH,
        fill: "#fef2f2",
      }, svg);
    }
  }

  for (const tick of ticks(0, yMax, 6)) {
    svgEl("line", { x1: M.l, y1: y(tick), x2: M.l + plotW, y2: y(tick), stroke: "#eef2f7" }, svg);
    svgEl("text", { x: M.l - 10, y: y(tick) + 4, "text-anchor": "end", "font-size": 11, fill: "#64748b" }, svg)
      .textContent = fmt(tick, 0);
  }
  for (let hour = 0; hour <= 36; hour += 6) {
    const index = Math.round((hour / 36) * (n - 1));
    svgEl("text", { x: x(index), y: M.t + plotH + 20, "text-anchor": "middle", "font-size": 11, fill: "#64748b" }, svg)
      .textContent = `${hour} h`;
  }
  svgEl("text", { x: M.l + plotW / 2, y: H - 8, "text-anchor": "middle", "font-size": 11.5, fill: "#64748b" }, svg)
    .textContent = xLabel;
  svgEl("text", {
    x: 13, y: M.t + plotH / 2, "text-anchor": "middle", "font-size": 11.5, fill: "#64748b",
    transform: `rotate(-90 13 ${M.t + plotH / 2})`,
  }, svg).textContent = yLabel;

  if (solar && solar.length) {
    const points = solar.map((value, index) => `${x(index)},${y(Number(value))}`).join(" ");
    svgEl("polygon", {
      points: `${M.l},${y(0)} ${points} ${x(n - 1)},${y(0)}`,
      fill: "#dcfce7", opacity: 0.75,
    }, svg);
    svgEl("polyline", {
      points, fill: "none", stroke: "#16a34a", "stroke-width": 1.2, opacity: 0.8,
    }, svg);
  }

  if (gridLimit) {
    svgEl("line", {
      x1: M.l, y1: y(gridLimit), x2: M.l + plotW, y2: y(gridLimit),
      stroke: "#dc2626", "stroke-width": 1.5, "stroke-dasharray": "6 4",
    }, svg);
    svgEl("text", {
      x: M.l + plotW - 4, y: y(gridLimit) - 6, "text-anchor": "end", "font-size": 10.5, fill: "#b91c1c",
    }, svg).textContent = `${fmt(gridLimit, 0)} kW grid limit`;
  }

  for (const entry of series) {
    const points = entry.values.map((value, index) => `${x(index)},${y(Number(value))}`).join(" ");
    svgEl("polyline", {
      points,
      fill: "none",
      stroke: entry.color,
      "stroke-width": entry.width || 2,
      "stroke-dasharray": entry.dash || null,
      "stroke-linejoin": "round",
      "stroke-linecap": "round",
    }, svg);
  }

  // hover
  const hover = svgEl("g", { opacity: 0 }, svg);
  const hoverLine = svgEl("line", {
    y1: M.t, y2: M.t + plotH, stroke: "#94a3b8", "stroke-width": 1, "stroke-dasharray": "3 3",
  }, hover);
  const hoverDots = series.map((entry) =>
    svgEl("circle", { r: 3.5, fill: entry.color, stroke: "#ffffff", "stroke-width": 1.5 }, hover)
  );
  const overlay = svgEl("rect", {
    x: M.l, y: M.t, width: plotW, height: plotH,
    fill: "transparent", "pointer-events": "all",
  }, svg);

  overlay.addEventListener("mousemove", (event) => {
    const rect = svg.getBoundingClientRect();
    const relative = ((event.clientX - rect.left) / rect.width) * W;
    const index = Math.max(0, Math.min(n - 1, Math.round(((relative - M.l) / plotW) * (n - 1))));
    hover.setAttribute("opacity", "1");
    hoverLine.setAttribute("x1", x(index));
    hoverLine.setAttribute("x2", x(index));
    hoverDots.forEach((dot, i) => {
      dot.setAttribute("cx", x(index));
      dot.setAttribute("cy", y(Number(series[i].values[index])));
    });

    const hourValue = Number(hours[index]);
    const rows = series
      .map((entry) => tipRow(shortName(entry.name), `${fmt(entry.values[index], 1)} kW`, entry.color))
      .join("");
    const solarRow = solar ? tipRow("Solar available", `${fmt(solar[index], 1)} kW`) : "";
    const tariffRow = tariff ? tipRow("Tariff", `${fmt(tariff[index], 1)} units/kWh`) : "";
    showTip(
      `<div class="tip-title">${fmt(hourValue, 2)} h · slot ${index}</div>${rows}${solarRow}${tariffRow}`,
      event.clientX,
      event.clientY
    );
  });
  overlay.addEventListener("mouseleave", () => {
    hover.setAttribute("opacity", "0");
    hideTip();
  });
}

/* ------------------------------------------------------------------ */
/* Profiles tab                                                        */
/* ------------------------------------------------------------------ */
function ensureSchedule(method) {
  if (state.schedules[method]) return Promise.resolve(state.schedules[method]);
  if (state.pending[method]) return state.pending[method];
  const promise = api(`/api/schedule?method=${encodeURIComponent(method)}`)
    .then((payload) => {
      state.schedules[method] = payload;
      return payload;
    })
    .finally(() => {
      delete state.pending[method];
    });
  state.pending[method] = promise;
  return promise;
}

async function renderProfiles() {
  const ov = state.overview;
  if (!ov) return;

  const chips = $("#profile-chips");
  const methods = ov.methods.map((row) => row.method);
  chips.innerHTML = "";
  methods.forEach((name) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chip${state.selection.has(name) ? " is-on" : ""}`;
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = methodStyle(name).color;
    button.append(dot, document.createTextNode(shortName(name)));
    button.addEventListener("click", async () => {
      if (state.selection.has(name)) {
        if (state.selection.size === 1) return;
        state.selection.delete(name);
      } else {
        state.selection.add(name);
      }
      button.classList.toggle("is-on", state.selection.has(name));
      await renderProfiles();
    });
    chips.appendChild(button);
  });

  const selected = methods.filter((name) => state.selection.has(name));
  await Promise.all(selected.map((name) => ensureSchedule(name).catch(() => null)));
  const payloads = selected.map((name) => state.schedules[name]).filter(Boolean);

  const series = payloads.map((payload) => {
    const style = methodStyle(payload.method);
    return {
      name: payload.method,
      values: payload.load_kw,
      color: style.color,
      dash: style.dash,
      width: payload.method === "GA" || payload.method === "PSO" ? 2.4 : 1.8,
    };
  });

  renderLineChart($("#chart-profiles"), {
    series,
    hours: ov.series.hours,
    solar: ov.series.solar_kw,
    tariff: ov.series.tariff,
    gridLimit: ov.site.max_grid_kw,
    showTariff: true,
  });

  renderLegend($("#profile-legend"), [
    ...selected.map((name) => {
      const style = methodStyle(name);
      return { color: style.color, label: shortName(name), line: true, dash: Boolean(style.dash) };
    }),
    { color: "#16a34a", label: "Solar available", line: true },
    { color: "#dc2626", label: `${fmt(ov.site.max_grid_kw, 0)} kW grid limit`, line: true, dash: true },
  ]);

  // stats table
  if (!payloads.length) {
    $("#profile-stats").innerHTML = '<p class="muted small">Select at least one method.</p>';
  } else {
    const rows = payloads.map((payload) => {
      const delivered = payload.load_kw.reduce((sum, value) => sum + Number(value), 0) * ov.site.step_hours;
      const peak = Math.max(...payload.load_kw.map(Number));
      const utilisation =
        (payload.occupancy.reduce((sum, value) => sum + Number(value), 0) /
          payload.occupancy.length /
          ov.site.n_chargers) * 100;
      const solarUsed =
        payload.load_kw.reduce(
          (sum, value, index) => sum + Math.min(Number(value), Number(ov.series.solar_kw[index])),
          0
        ) * ov.site.step_hours;
      return { payload, delivered, peak, utilisation, solarUsed };
    });

    $("#profile-stats").innerHTML = `
      <div class="table-wrap">
        <table>
          <thead><tr><th>Method</th><th>Delivered (kWh)</th><th>Peak (kW)</th><th>Port use</th><th>Solar used (kWh)</th></tr></thead>
          <tbody>
            ${rows
              .map(
                ({ payload, delivered, peak, utilisation, solarUsed }) => `
              <tr>
                <td><span class="dot" style="background:${methodStyle(payload.method).color}"></span>${escapeHtml(shortName(payload.method))}</td>
                <td class="num">${fmt(delivered, 1)}</td>
                <td class="num">${fmt(peak, 1)}</td>
                <td class="num">${fmt(utilisation, 1)}%</td>
                <td class="num">${fmt(solarUsed, 1)}</td>
              </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>
      <p class="hint">Delivered energy here is the integral of the plotted load, so it matches the committed table row for row.</p>`;
  }

  $("#port-hours-inline").textContent = `${fmt(ov.port_pressure.port_hours_available, 0)} port-hours`;
  const psoSchedule = state.schedules["PSO"];
  if (psoSchedule) {
    const step = ov.site.step_hours;
    const used =
      psoSchedule.load_kw.reduce(
        (sum, value, index) => sum + Math.min(Number(value), Number(ov.series.solar_kw[index])),
        0
      ) * step;
    $("#solar-note").textContent = `${fmt((used / ov.site.solar_available_kwh) * 100, 1)}%`;
  }
}

/* ------------------------------------------------------------------ */
/* Schedule tab                                                        */
/* ------------------------------------------------------------------ */
function scheduleMetrics(payload) {
  const m = payload.metrics || {};
  return {
    delivered: m.service_delivered_kwh ?? m.delivered_kwh ?? null,
    peak: m.peak_ev_load_kw ?? m.peak_kw ?? null,
    full: m.fully_charged_evs ?? m.fully_charged ?? null,
    cost: m.cost_per_delivered_kwh ?? m.cost_per_kwh ?? null,
    violations: m.constraint_violations ?? m.violations ?? null,
  };
}

async function renderScheduleTab() {
  const ov = state.overview;
  if (!ov) return;
  const select = $("#schedule-select");
  const options = ov.methods.map((row) => row.method).concat(state.liveMethods);
  const current = state.currentMethod || "PSO";

  if (select.options.length !== options.length) {
    select.innerHTML = "";
    options.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = shortName(name);
      select.appendChild(option);
    });
    select.value = options.includes(current) ? current : options[0];
  }

  const method = select.value;
  state.currentMethod = method;
  select.onchange = () => {
    state.currentMethod = select.value;
    renderScheduleTab();
  };

  try {
    const payload = await ensureSchedule(method);
    drawSchedulePills(payload);
    drawHeatmap(payload);
    drawOccupancy(payload);
    drawWorstEvs(payload);
  } catch (error) {
    $("#schedule-pills").innerHTML = `<span class="pill">${escapeHtml(error.message)}</span>`;
  }
}

function drawSchedulePills(payload) {
  const metrics = scheduleMetrics(payload);
  const items = [
    ["Delivered", `${fmt(metrics.delivered, 1)} kWh`],
    ["Peak", `${fmt(metrics.peak, 1)} kW`],
    ["Fully charged", `${fmt(metrics.full, 0)} / ${payload.shape[0]}`],
    ["Cost / kWh", fmt(metrics.cost, 4)],
    ["Violations", fmt(metrics.violations, 0)],
  ];
  $("#schedule-pills").innerHTML = items
    .map(([label, value]) => `<span class="pill">${label} <b>${value}</b></span>`)
    .join("");
}

function drawHeatmap(payload) {
  const canvas = $("#schedule-canvas");
  const wrap = $("#heat-wrap");
  const cssWidth = Math.max(240, wrap.clientWidth - 2);
  const rows = payload.shape[0];
  const cols = payload.shape[1];
  const cellW = cssWidth / cols;
  const cellH = 3;
  const cssHeight = rows * cellH;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const bytes = base64Bytes(payload.cells);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const value = bytes[row * cols + col];
      if (!value) continue;
      ctx.fillStyle = powerColor(value / 255);
      ctx.fillRect(col * cellW, row * cellH, Math.max(1, Math.ceil(cellW)), cellH);
    }
  }

  ctx.strokeStyle = "rgba(15, 23, 42, 0.18)";
  ctx.lineWidth = 1;
  for (let hour = 6; hour < 36; hour += 6) {
    const px = Math.round((hour / 36) * cols) * cellW + 0.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, cssHeight);
    ctx.stroke();
  }

  state.heat = { payload, cellW, cellH, rows, cols };

  const axis = $("#heat-axis");
  axis.innerHTML = "";
  for (const hour of [0, 6, 12, 18, 24, 30, 36]) {
    const label = document.createElement("span");
    label.style.left = `${(hour / 36) * 100}%`;
    label.textContent = `${hour} h`;
    axis.appendChild(label);
  }
}

function drawOccupancy(payload) {
  const canvas = $("#occupancy-canvas");
  const host = canvas.parentElement;
  const cssWidth = Math.max(260, host.clientWidth - 48);
  const cssHeight = 130;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const M = { l: 30, r: 8, t: 10, b: 22 };
  const plotW = cssWidth - M.l - M.r;
  const plotH = cssHeight - M.t - M.b;
  const n = payload.occupancy.length;
  const yMax = state.overview.site.n_chargers;
  const y = (value) => M.t + plotH - (value / yMax) * plotH;

  ctx.strokeStyle = "#eef2f7";
  ctx.fillStyle = "#64748b";
  ctx.font = "10px 'Segoe UI', system-ui, sans-serif";
  ctx.textAlign = "right";
  for (const tick of [0, 10, 20]) {
    ctx.beginPath();
    ctx.moveTo(M.l, y(tick));
    ctx.lineTo(M.l + plotW, y(tick));
    ctx.stroke();
    ctx.fillText(String(tick), M.l - 6, y(tick) + 3);
  }

  const barW = plotW / n;
  payload.occupancy.forEach((count, index) => {
    const value = Number(count);
    ctx.fillStyle = "#bfdbfe";
    ctx.fillRect(M.l + index * barW, y(value), Math.max(1, barW - 0.2), M.t + plotH - y(value));
  });

  ctx.strokeStyle = "#1d4ed8";
  ctx.beginPath();
  payload.occupancy.forEach((count, index) => {
    const cx = M.l + index * barW + barW / 2;
    const cy = y(Number(count));
    if (index === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();

  ctx.strokeStyle = "#dc2626";
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  ctx.moveTo(M.l, y(yMax));
  ctx.lineTo(M.l + plotW, y(yMax));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#64748b";
  ctx.textAlign = "center";
  for (const hour of [0, 12, 24, 36]) {
    const slot = Math.round((hour / 36) * (n - 1));
    ctx.fillText(`${hour} h`, M.l + slot * barW + barW / 2, cssHeight - 6);
  }

  ctx.fillStyle = "#b91c1c";
  ctx.textAlign = "left";
  ctx.fillText(`${yMax} ports`, M.l + 4, y(yMax) - 4);
}

function drawWorstEvs(payload) {
  const rows = payload.evs.id
    .map((id, index) => ({
      id,
      arrival: payload.evs.arrival[index],
      departure: payload.evs.departure[index],
      required: payload.evs.required[index],
      delivered: payload.evs.delivered[index],
    }))
    .filter((row) => row.required > 0)
    .map((row) => ({ ...row, ratio: row.delivered / row.required }))
    .sort((a, b) => a.ratio - b.ratio)
    .slice(0, 6);

  const step = state.overview.site.step_hours;
  $("#worst-ev").innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr><th>EV</th><th>Window</th><th>Needed</th><th>Delivered</th><th>%</th></tr></thead>
        <tbody>
          ${rows
            .map(
              (row) => `<tr>
                <td>#${row.id}</td>
                <td class="num">${fmt(row.arrival * step, 1)}–${fmt(row.departure * step, 1)} h</td>
                <td class="num">${fmt(row.required, 1)}</td>
                <td class="num">${fmt(row.delivered, 1)}</td>
                <td class="num">${fmt(row.ratio * 100, 0)}%</td>
              </tr>`
            )
            .join("")}
        </tbody>
      </table>
    </div>
    <p class="hint">A 100% row means the EV left full; anything less was traded away for another session.</p>`;
}

function initScheduleHover() {
  const wrap = $("#heat-wrap");
  const canvas = $("#schedule-canvas");
  const cursor = $("#heat-cursor");

  wrap.addEventListener("mousemove", (event) => {
    if (!state.heat) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const col = Math.floor(x / state.heat.cellW);
    const row = Math.floor(y / state.heat.cellH);
    if (col < 0 || col >= state.heat.cols || row < 0 || row >= state.heat.rows) {
      cursor.hidden = true;
      hideTip();
      return;
    }

    const { payload } = state.heat;
    const bytes = base64Bytes(payload.cells);
    const quantised = bytes[row * state.heat.cols + col];
    const power = (quantised / 255) * payload.cell_scale_kw;
    const ev = payload.evs;
    const slot = Number(payload.hours[col]);
    const required = Number(ev.required[row]);
    const delivered = Number(ev.delivered[row]);
    const step = state.overview.site.step_hours;

    cursor.hidden = false;
    cursor.style.left = `${canvas.offsetLeft + col * state.heat.cellW}px`;
    cursor.style.top = `${canvas.offsetTop + row * state.heat.cellH}px`;
    cursor.style.width = `${Math.max(2, state.heat.cellW)}px`;
    cursor.style.height = `${Math.max(3, state.heat.cellH)}px`;

    showTip(
      `<div class="tip-title">EV #${ev.id[row]} · slot ${col} (${fmt(slot, 2)} h)</div>` +
        tipRow("Charging", quantised ? `≈ ${fmt(power, 1)} kW` : "idle") +
        tipRow("Connected", `${fmt(ev.arrival[row] * step, 1)}–${fmt(ev.departure[row] * step, 1)} h`) +
        tipRow("Needs", `${fmt(required, 1)} kWh`) +
        tipRow("Gets", `${fmt(delivered, 1)} kWh (${fmt((delivered / required) * 100, 0)}%)`) +
        `<div class="tip-note">One row per EV; empty cells are connected time without a port, or outside the connection window.</div>`,
      event.clientX,
      event.clientY
    );
  });

  wrap.addEventListener("mouseleave", () => {
    cursor.hidden = true;
    hideTip();
  });
}

/* ------------------------------------------------------------------ */
/* Live run                                                            */
/* ------------------------------------------------------------------ */
function initLive() {
  const ov = state.overview;
  const budgets = $("#budget-options");
  budgets.innerHTML = ov.budgets
    .map(
      (budget, index) => `
      <label>
        <input type="radio" name="budget" value="${budget.key}" ${index === 0 ? "checked" : ""}>
        <b>${budget.label}</b>
        <span>${fmt(budget.ga_evaluations, 0)} GA / ${fmt(budget.pso_evaluations, 0)} PSO evals</span>
        <em>${budget.key === "fast" ? "~3 s" : budget.key === "standard" ? "~10 s" : "~20 s"}</em>
      </label>`
    )
    .join("");

  const weights = ov.objective_weights;
  $("#w-unmet").value = weights.unmet;
  $("#w-cost").value = weights.cost;
  $("#w-peak").value = weights.peak;
  updateWeightOutputs();

  $$("#live-form input[type=range]").forEach((input) =>
    input.addEventListener("input", updateWeightOutputs)
  );

  $$("[data-preset]").forEach((button) => {
    button.addEventListener("click", () => {
      const presets = {
        service: { unmet: 0.85, cost: 0.10, peak: 0.05 },
        balanced: { unmet: 0.55, cost: 0.30, peak: 0.08 },
        cost: { unmet: 0.25, cost: 0.65, peak: 0.05 },
      };
      const preset = presets[button.dataset.preset];
      $("#w-unmet").value = preset.unmet;
      $("#w-cost").value = preset.cost;
      $("#w-peak").value = preset.peak;
      $$("[data-preset]").forEach((other) =>
        other.classList.toggle("is-on", other === button)
      );
      updateWeightOutputs();
    });
  });

  $("#live-form").addEventListener("submit", runLive);
  $("#run-hint").textContent =
    `Committed run: ${fmt(ov.full_budget.GA, 0)} GA / ${fmt(ov.full_budget.PSO, 0)} PSO evaluations. Live runs are smaller and will not match the table exactly.`;
}

function currentWeights() {
  return {
    unmet: Number($("#w-unmet").value),
    cost: Number($("#w-cost").value),
    peak: Number($("#w-peak").value),
    battery: 0.02,
    solar: 0.05,
  };
}

function updateWeightOutputs() {
  const weights = currentWeights();
  const total = weights.unmet + weights.cost + weights.peak + weights.battery + weights.solar;
  for (const key of ["unmet", "cost", "peak"]) {
    $(`#w-${key}-out`).textContent = `${fmt((weights[key] / total) * 100, 0)}%`;
  }
}

async function runLive(event) {
  event.preventDefault();
  const button = $("#run-button");
  const status = $("#live-status");
  const algorithms = [
    $("#algo-ga").checked ? "GA" : null,
    $("#algo-pso").checked ? "PSO" : null,
  ].filter(Boolean);

  if (!algorithms.length) {
    status.className = "status error";
    status.textContent = "Select at least one optimizer.";
    return;
  }
  const seed = Number($("#seed-input").value);
  if (!Number.isInteger(seed) || seed < 0) {
    status.className = "status error";
    status.textContent = "Seed must be a non-negative integer.";
    return;
  }

  const budget = ($('input[name="budget"]:checked') || {}).value || "fast";
  const started = performance.now();
  button.disabled = true;
  status.className = "status busy";
  status.textContent = "Running baselines and optimizers…";

  const timer = window.setInterval(() => {
    status.textContent = `Running… ${((performance.now() - started) / 1000).toFixed(1)} s`;
  }, 200);

  try {
    const result = await api("/api/optimise", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        budget,
        seed,
        algorithms,
        weights: currentWeights(),
      }),
    });
    applyLiveResult(result);
    status.className = "status";
    status.textContent = `Done in ${fmt(result.elapsed_seconds, 1)} s.`;
  } catch (error) {
    status.className = "status error";
    status.textContent = error.message;
  } finally {
    window.clearInterval(timer);
    button.disabled = false;
  }
}

function applyLiveResult(result) {
  state.live = result;

  for (const name of ["GA", "PSO"]) {
    if (result.schedules[name]) {
      const key = `${name} (live)`;
      state.schedules[key] = { ...result.schedules[name], method: key };
      if (!state.liveMethods.includes(key)) state.liveMethods.push(key);
    }
  }

  $("#live-placeholder").hidden = true;
  $("#live-output").hidden = false;

  const best = result.best_optimizer;
  $("#live-verdict").innerHTML = `
    <strong>${escapeHtml(best.method)} delivered ${fmt(best.delivered_kwh, 1)} kWh</strong>
    — ${signed(best.delta_vs_conventional_pct)}% versus ${escapeHtml(result.anchor.method)} and
    ${fmt(best.pct_of_optimum, 1)}% of the certified optimum.
    <span class="sub">${escapeHtml(result.budget_label)} budget · ${fmt(
      result.rows.find((row) => row.method === best.method).evaluations,
      0
    )} evaluations · seed ${result.seed} · ${fmt(result.elapsed_seconds, 1)} s</span>`;

  const rows = [...result.rows].sort((a, b) => b.delivered_kwh - a.delivered_kwh);
  $("#live-table").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Method</th><th>Delivered (kWh)</th><th>% of optimum</th><th>vs best conv.</th>
          <th>Full EVs</th><th>Peak (kW)</th><th>Cost/kWh</th><th>Evals</th><th>Time</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map((row) => {
            const isBest = row.method === best.method;
            const family = FAMILY[row.family] || FAMILY.conventional;
            const clickable = ["GA", "PSO"].includes(row.method);
            return `<tr class="${isBest ? "is-best" : ""}" ${clickable ? 'data-schedule="' + row.method + ' (live)" style="cursor:pointer"' : ""}>
              <td><span class="dot" style="background:${family.color}"></span>${escapeHtml(shortName(row.method))}${row.feasible ? "" : '<span class="tag-infeasible">infeasible</span>'}</td>
              <td class="num">${fmt(row.delivered_kwh, 1)}</td>
              <td class="num">${fmt(row.pct_of_optimum, 1)}%</td>
              <td class="num">${row.feasible ? signed(row.delta_vs_conventional_pct) + "%" : "—"}</td>
              <td class="num">${fmt(row.fully_charged, 0)}</td>
              <td class="num">${fmt(row.peak_kw, 1)}</td>
              <td class="num">${fmt(row.cost_per_kwh, 4)}</td>
              <td class="num">${fmt(row.evaluations, 0)}</td>
              <td class="num">${fmt(row.seconds, 1)} s</td>
            </tr>`;
          })
          .join("")}
      </tbody>
    </table>`;

  $$("#live-table tr[data-schedule]").forEach((row) => {
    row.addEventListener("click", () => {
      state.currentMethod = row.dataset.schedule;
      selectTab("schedule");
    });
  });

  const series = [];
  for (const name of ["PSO", "GA"]) {
    if (result.series[name]) {
      const style = methodStyle(name);
      series.push({ name, values: result.series[name], color: style.color, width: 2.4 });
    }
  }
  const anchorName = result.anchor.method;
  if (result.series[anchorName]) {
    series.push({
      name: anchorName,
      values: result.series[anchorName],
      color: "#ea580c",
      dash: "6 4",
      width: 1.8,
    });
  }

  renderLineChart($("#chart-live"), {
    series,
    hours: state.overview.series.hours,
    solar: state.overview.series.solar_kw,
    tariff: state.overview.series.tariff,
    gridLimit: state.overview.site.max_grid_kw,
    showTariff: true,
  });
  renderLegend($("#live-legend"), [
    ...series.map((entry) => ({
      color: entry.color,
      label: shortName(entry.name),
      line: true,
      dash: Boolean(entry.dash),
    })),
    { color: "#16a34a", label: "Solar available", line: true },
    { color: "#dc2626", label: "grid limit", line: true, dash: true },
  ]);

  const histories = {};
  for (const name of ["GA", "PSO"]) {
    const payload = result.schedules[name];
    if (payload && payload.history && payload.history.length) {
      histories[name] = payload.history;
    }
  }
  renderConvergence(histories);
}

function renderConvergence(histories) {
  const host = $("#chart-convergence");
  host.innerHTML = "";
  const entries = Object.entries(histories).filter(([, values]) => values.length);
  renderLegend(
    $("#convergence-legend"),
    entries.map(([name]) => ({ color: methodStyle(name).color, label: name, line: true }))
  );
  if (!entries.length) {
    host.innerHTML = '<p class="muted small">No convergence history in this run.</p>';
    return;
  }

  const W = 1000;
  const H = 300;
  const M = { l: 64, r: 24, t: 16, b: 44 };
  const plotW = W - M.l - M.r;
  const plotH = H - M.t - M.b;
  const maxSteps = Math.max(...entries.map(([, values]) => values.length)) - 1;
  const allValues = entries.flatMap(([, values]) => values.map(Number));
  const rawMin = Math.min(...allValues);
  const rawMax = Math.max(...allValues);
  const padding = Math.max(0.002, (rawMax - rawMin) * 0.12);
  const yMin = rawMin - padding;
  const yMax = rawMax + padding;

  const x = (step) => M.l + (maxSteps === 0 ? 0 : (step / maxSteps) * plotW);
  const y = (value) => M.t + plotH - ((value - yMin) / (yMax - yMin)) * plotH;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Convergence chart" });
  host.appendChild(svg);

  for (const tick of ticks(yMin, yMax, 4)) {
    svgEl("line", { x1: M.l, y1: y(tick), x2: M.l + plotW, y2: y(tick), stroke: "#eef2f7" }, svg);
    svgEl("text", { x: M.l - 10, y: y(tick) + 4, "text-anchor": "end", "font-size": 11, fill: "#64748b" }, svg)
      .textContent = fmt(tick, 3);
  }
  for (let step = 0; step <= maxSteps; step += Math.max(1, Math.round(maxSteps / 5))) {
    svgEl("text", { x: x(step), y: M.t + plotH + 20, "text-anchor": "middle", "font-size": 11, fill: "#64748b" }, svg)
      .textContent = String(step);
  }
  svgEl("text", { x: M.l + plotW / 2, y: H - 8, "text-anchor": "middle", "font-size": 11.5, fill: "#64748b" }, svg)
    .textContent = "generation / iteration";
  svgEl("text", {
    x: 14, y: M.t + plotH / 2, "text-anchor": "middle", "font-size": 11.5, fill: "#64748b",
    transform: `rotate(-90 14 ${M.t + plotH / 2})`,
  }, svg).textContent = "best objective (lower is better)";

  for (const [name, values] of entries) {
    const style = methodStyle(name);
    const points = values.map((value, index) => `${x(index)},${y(Number(value))}`).join(" ");
    svgEl("polyline", {
      points, fill: "none", stroke: style.color, "stroke-width": 2,
    }, svg);
  }
}

/* ------------------------------------------------------------------ */
/* Evidence tab                                                        */
/* ------------------------------------------------------------------ */
function renderEvidence() {
  const ov = state.overview;

  const reference = ov.reference;
  $("#reference-table").innerHTML = `
    <table>
      <thead>
        <tr><th>Reference point</th><th>Delivered (kWh)</th><th>Demand met</th><th>Full EVs</th>
        <th>Peak (kW)</th><th>Cost (units)</th><th>MIP gap</th><th>Solve (s)</th><th>Status</th></tr>
      </thead>
      <tbody>
        ${reference
          .map(
            (row) => `<tr>
              <td>${escapeHtml(row.label)}</td>
              <td class="num">${fmt(row.delivered_energy_kwh, 1)}</td>
              <td class="num">${fmt(row.demand_met_pct, 1)}%</td>
              <td class="num">${fmt(row.fully_charged_evs, 0)}</td>
              <td class="num">${fmt(row.peak_ev_load_kw, 1)}</td>
              <td class="num">${fmt(row.grid_cost_units, 0)}</td>
              <td class="num">${fmt(Number(row.mip_gap) * 100, 2)}%</td>
              <td class="num">${fmt(row.solve_seconds, 1)}</td>
              <td>${escapeHtml(row.status)}</td>
            </tr>`
          )
          .join("")}
      </tbody>
    </table>
    <p class="hint">Inputs hashed as scenario ${escapeHtml(reference[0] ? reference[0].scenario_sha256_16 : "—")} and solar ${escapeHtml(reference[0] ? reference[0].solar_sha256_16 : "—")}. Each row is only valid for that pair.</p>`;

  $("#seed-table").innerHTML = `
    <table>
      <thead><tr><th>Method</th><th>Runs</th><th>Delivered mean</th><th>Std</th><th>Peak mean</th><th>Cost mean</th></tr></thead>
      <tbody>
        ${ov.multi_seed.summary
          .map(
            (row) => `<tr>
              <td><span class="dot" style="background:${methodStyle(row.method).color}"></span>${escapeHtml(row.method)}</td>
              <td class="num">${fmt(row.runs, 0)}</td>
              <td class="num">${fmt(row.delivered_mean, 1)}</td>
              <td class="num">${fmt(row.delivered_std, 1)}</td>
              <td class="num">${fmt(row.peak_mean, 1)}</td>
              <td class="num">${fmt(row.cost_mean, 0)}</td>
            </tr>`
          )
          .join("")}
      </tbody>
    </table>
    <p class="hint">Seeds 42–46. The spread is the honest error bar on a single run.</p>`;

  const manifest = ov.manifest;
  const ga = manifest.ga_settings || {};
  const pso = manifest.pso_settings || {};
  const entries = [
    ["Generated (UTC)", manifest.generated_utc],
    ["Mode", manifest.mode],
    ["Scenario hash", manifest.scenario_sha256_16],
    ["Solar hash", manifest.solar_sha256_16],
    ["GA settings", `${ga.population_size} pop × ${ga.generations} gen + ${ga.polish_evaluations} polish`],
    ["PSO settings", `${pso.particles} particles × ${pso.iterations} it + ${pso.polish_evaluations} polish`],
    ["Seeds", (manifest.multi_seed_seeds || []).join(", ")],
    ["Site", `${manifest.n_ev} EVs · ${manifest.n_chargers} ports · ${manifest.max_grid_kw} kW`],
  ];
  $("#manifest-list").innerHTML = entries
    .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value ?? "—")}</dd>`)
    .join("");

  const count = ov.tests.collected;
  $("#test-count").textContent = count
    ? `pytest collects ${count} tests in this repository. The suite is what asserts the headline claim — both optimizers beat every conventional policy — rather than a slide.`
    : "";

  $("#artifact-list").innerHTML = ov.artifacts
    .map((artifact) => {
      if (["csv", "png"].includes(artifact.suffix)) {
        return `<a class="artifact" href="/artifacts/${encodeURIComponent(artifact.name)}" download>
          ${escapeHtml(artifact.name)}<span>${fmt(artifact.size_kb, 1)} kB</span>
        </a>`;
      }
      return `<span class="artifact disabled">${escapeHtml(artifact.name)}<span>${fmt(artifact.size_kb, 1)} kB</span></span>`;
    })
    .join("");
}

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */
let resizeTimer = null;
window.addEventListener("resize", () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    if (state.tab === "schedule" && state.currentMethod) {
      const payload = state.schedules[state.currentMethod];
      if (payload) {
        drawHeatmap(payload);
        drawOccupancy(payload);
      }
    }
  }, 160);
});

async function boot() {
  initTabs();
  initScheduleHover();
  try {
    state.overview = await api("/api/overview");
  } catch (error) {
    $("#brand-sub").textContent = `Could not load results: ${error.message}`;
    return;
  }
  renderOverview();
  renderEvidence();
  initLive();
  await Promise.all(
    ["EDF (chronological)", "FCFS (chronological)", "GA", "PSO"].map((name) =>
      ensureSchedule(name).catch(() => null)
    )
  );
}

boot();
