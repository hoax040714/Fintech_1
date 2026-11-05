"use strict";

const byId = id => document.getElementById(id);
const choice = id => byId(id).value;
const gold = "#E4B863";
const moon = "#7C9CC4";
const up = "#BB5353";
const down = "#2B9778";
const cache = new Map();
const number = new Intl.NumberFormat("zh-CN");
const finite = value => Number.isFinite(value);
const decimal = (value, digits = 6) => finite(value) ? value.toFixed(digits) : "—";
const percentage = (value, digits = 2) => finite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
const money = value => finite(value) ? `¥ ${decimal(value, 2)}` : "—";
const days = horizon => `${horizon} 个交易日`;
const newestDate = date => date.toISOString().slice(0, 10);
let overview;
let catalog;
let activeView = "prediction";
let request = 0;
let factorScope = "";

function data(file) {
  if (!cache.has(file)) {
    cache.set(file, fetch(`data/${file}.json`).then(response => {
      if (!response.ok) throw new Error(`读取 ${file} 失败（HTTP ${response.status}）`);
      return response.json();
    }).catch(error => {
      cache.delete(file);
      throw error;
    }));
  }
  return cache.get(file);
}

function options(id, values, label = value => value, preferred) {
  const select = byId(id);
  const current = preferred ?? select.value;
  select.replaceChildren(...values.map(value => new Option(label(value), value)));
  if (values.some(value => String(value) === current)) select.value = current;
  select.disabled = values.length === 0;
}

function element(tag, value, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value instanceof Node) node.append(value);
  else node.textContent = value ?? "—";
  return node;
}

function table(id, headings, rows) {
  const header = element("tr");
  for (const title of headings) {
    const cell = element("th", title);
    cell.scope = "col";
    header.append(cell);
  }
  const head = element("thead");
  head.append(header);
  const body = element("tbody");
  for (const values of rows) {
    const row = element("tr");
    row.append(...values.map(value => element("td", value)));
    body.append(row);
  }
  byId(id).replaceChildren(head, body);
}

function cards(id, values) {
  byId(id).replaceChildren(...values.map(([caption, value]) => {
    const card = element("div", "", "metric");
    card.append(element("span", caption), element("strong", value));
    return card;
  }));
}

function plot(id, traces, layout) {
  const defaults = {
    margin: {l: 58, r: 22, t: 32, b: 55},
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,255,255,.035)",
    font: {color: "#DDE7F1", family: "Aptos, Microsoft YaHei UI, sans-serif"},
    legend: {orientation: "h", y: 1.13},
    hovermode: "closest",
  };
  return Plotly.react(byId(id), traces, {...defaults, ...layout},
    {responsive: true, displaylogo: false, scrollZoom: false, locale: "zh-CN"});
}

function messageChart(id, text) {
  const chart = byId(id);
  if (chart.classList.contains("js-plotly-plot")) Plotly.purge(chart);
  chart.textContent = text;
}

const modelName = model => overview.modelLabels[model] || model;
const stockName = code => overview.names[code] || code;
const modelOrder = values => {
  const available = new Set(values);
  return [...Object.keys(overview.modelLabels).filter(model => available.delete(model)), ...available];
};

function futureBusinessDate(start, horizon) {
  const date = new Date(`${start}T00:00:00Z`);
  for (let remaining = horizon; remaining > 0;) {
    date.setUTCDate(date.getUTCDate() + 1);
    if (date.getUTCDay() !== 0 && date.getUTCDay() !== 6) remaining--;
  }
  return newestDate(date);
}

async function prediction(isCurrent) {
  const horizon = Number(choice("forecast-horizon"));
  const groups = await data(`predictions-${horizon}`);
  if (!isCurrent()) return;
  options("forecast-model", modelOrder(Object.keys(groups)), modelName);
  const model = choice("forecast-model");
  const rows = groups[model];
  options("forecast-stock", rows.map(row => row[0]), code => `${code}　${stockName(code)}`);
  const code = choice("forecast-stock");
  const [, date, close, estimate, target] = rows.find(row => row[0] === code);
  byId("forecast-date").textContent = `预测形成日 ${date} · ${modelName(model)} · ${days(horizon)}`;
  cards("forecast-cards", [
    ["预测基准收盘价 / CNY", money(close)],
    [`预测 ${horizon} 日收益率`, percentage(estimate)],
    ["参考目标价 / CNY", money(target)],
    ["预测方向", estimate > 0 ? "上涨" : estimate < 0 ? "下跌" : "持平"],
  ]);
  byId("ranking-info").textContent = `${modelName(model)} · ${days(horizon)} · ${rows.length} 只股票`;
  table("ranking", ["排名", "股票", "基准收盘价", "预测收益率", "参考目标价"],
    [...rows].sort((a, b) => b[3] - a[3]).map(([ticker, , price, predicted, reference], index) => {
      const button = element("button", `${ticker} ${stockName(ticker)}`, "stock-link");
      button.type = "button";
      button.dataset.stock = ticker;
      if (ticker === code) button.setAttribute("aria-current", "true");
      return [index + 1, button, money(price), percentage(predicted), money(reference)];
    }));
  byId("ranking").querySelector('[aria-current="true"]')?.closest("tr").classList.add("is-selected");
  byId("price-title").textContent = `${code} ${stockName(code)} · 历史 K 线与预测终点`;
  messageChart("price-chart", "正在读取所选股票的历史行情…");
  const bars = (await data(`prices/${code}`)).filter(row => row[0] <= date);
  if (!isCurrent()) return;
  const dates = bars.map(row => row[0]);
  if (!dates.length) {
    messageChart("price-chart", "此股票没有可用的历史 K 线。");
    return;
  }
  const future = futureBusinessDate(date, horizon);
  const dateSet = new Set(dates);
  const breaks = [];
  for (const day = new Date(`${dates[0]}T00:00:00Z`); newestDate(day) < date; day.setUTCDate(day.getUTCDate() + 1)) {
    const stamp = newestDate(day);
    if (day.getUTCDay() > 0 && day.getUTCDay() < 6 && !dateSet.has(stamp)) breaks.push(stamp);
  }
  const axis = {
    type: "date", range: [dates.at(-126) || dates[0], futureBusinessDate(future, 5)],
    rangebreaks: [{bounds: ["sat", "mon"]}, {values: breaks}],
    rangeslider: {visible: false}, showgrid: false, showspikes: true, spikemode: "across",
  };
  const recent = bars.slice(-126);
  const low = Math.min(close, target, ...recent.map(row => row[3]).filter(finite));
  const high = Math.max(close, target, ...recent.map(row => row[2]).filter(finite));
  const padding = Math.max((high - low) * .1, high * .02);
  const volume = Math.max(1, ...recent.map(row => row[5]).filter(finite));
  await plot("price-chart", [
    {
      type: "candlestick", x: dates, open: bars.map(row => row[1]), high: bars.map(row => row[2]),
      low: bars.map(row => row[3]), close: bars.map(row => row[4]), name: "历史 K 线",
      increasing: {line: {color: up}}, decreasing: {line: {color: down}},
    },
    {
      type: "scatter", x: [date, future], y: [close, target], mode: "lines+markers",
      name: `${horizon} 日参考目标`, line: {color: gold, width: 3, dash: "dash"},
      hovertemplate: "%{x|%Y-%m-%d}<br>¥ %{y:.2f}<extra></extra>",
    },
    {
      type: "bar", x: dates, y: bars.map(row => row[5]), yaxis: "y2", name: "成交量",
      marker: {color: bars.map(row => row[4] >= row[1] ? up : down)}, hovertemplate: "%{y:,.0f}<extra>成交量</extra>",
    },
  ], {
    height: 535, dragmode: "pan", xaxis: axis,
    yaxis: {domain: [0.25, 1], range: [low - padding, high + padding], title: "价格 / CNY", gridcolor: "#324059"},
    yaxis2: {domain: [0, 0.19], range: [0, volume * 1.15], title: "成交量", tickformat: "~s", showgrid: false},
    annotations: [{x: future, y: target, text: money(target), showarrow: false, yshift: 13, font: {color: gold}}],
  });
}

function signedBars(id, rows, name) {
  if (!rows.length) {
    messageChart(id, "所选条件下没有可用结果。");
    return;
  }
  const ordered = [...rows].reverse();
  return plot(id, [{
    type: "bar", orientation: "h", x: ordered.map(row => row.value), y: ordered.map(row => row.factor),
    marker: {color: ordered.map(row => row.value < 0 ? moon : gold)}, name,
    hovertemplate: "%{y}<br>%{x:.6g}<extra></extra>",
  }], {height: Math.max(270, ordered.length * 21 + 70), showlegend: false,
    xaxis: {title: name, gridcolor: "#324059"}, yaxis: {automargin: true}});
}

async function models() {
  const horizon = Number(choice("metric-horizon"));
  const rows = overview.metrics.filter(row => row.horizon === horizon).sort((a, b) => a.mae - b.mae);
  options("importance-model", rows.map(row => row.model), modelName);
  table("model-table", ["模型", "验证集最优参数", "R²", "MAE", "RMSE", "方向准确率", "预测相关系数", "测试开始", "测试结束", "测试样本"],
    rows.map(row => [modelName(row.model), row.best_params, decimal(row.r2), decimal(row.mae), decimal(row.rmse),
      percentage(row.direction_accuracy, 1), decimal(row.prediction_correlation), row.test_start, row.test_end, number.format(row.n_test)]));
  const modelChart = plot("model-chart", [
    {type: "bar", x: rows.map(row => modelName(row.model)), y: rows.map(row => row.r2), name: "R²", marker: {color: gold}},
    {type: "bar", x: rows.map(row => modelName(row.model)), y: rows.map(row => row.prediction_correlation), name: "预测相关系数", marker: {color: moon}},
  ], {height: 340, barmode: "group", xaxis: {automargin: true}, yaxis: {gridcolor: "#324059"}});
  const model = choice("importance-model");
  const category = choice("importance-category");
  const importance = overview.importance.filter(row => row.horizon === horizon && row.model === model);
  const top = importance.filter(row => !category || catalog.get(row.factor)?.factor_type === category)
    .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance)).slice(0, Number(choice("importance-count")));
  byId("importance-method").textContent = importance.length ? `权重口径 · ${importance[0].importance_type}` : "";
  table("importance-table", ["因子", "类型", "定义", "权重 / 重要性"],
    top.map(row => [row.factor, catalog.get(row.factor)?.factor_type, catalog.get(row.factor)?.definition, decimal(row.importance)]));
  await Promise.all([modelChart, signedBars("importance-chart", top.map(row => ({factor: row.factor, value: row.importance})), "权重 / 重要性")]);
}

function statistics(values, first, last, availableDates) {
  let count = 0;
  let mean = 0;
  let squared = 0;
  for (let i = first; i <= last; i++) {
    const value = values[i];
    if (!finite(value)) continue;
    availableDates.add(i);
    const delta = value - mean;
    count++;
    mean += delta / count;
    squared += delta * (value - mean);
  }
  return {value: count ? mean : null, deviation: count > 1 ? Math.sqrt(Math.max(0, squared / (count - 1))) : null, count};
}

async function factors(isCurrent) {
  const horizon = choice("factor-horizon");
  const category = choice("factor-category");
  const {dates, series} = await data(`factor-ic-${horizon}`);
  if (!isCurrent()) return;
  const selected = Object.entries(series).filter(([factor]) => !category || catalog.get(factor)?.factor_type === category);
  const start = byId("factor-start");
  const end = byId("factor-end");
  start.min = end.min = dates[0];
  start.max = end.max = dates.at(-1);
  const scope = `${horizon}/${category}`;
  if (factorScope !== scope) {
    const twoYearsAgo = new Date(`${end.max}T00:00:00Z`);
    twoYearsAgo.setUTCFullYear(twoYearsAgo.getUTCFullYear() - 2);
    const defaultStart = newestDate(twoYearsAgo);
    start.value = defaultStart < start.min ? start.min : defaultStart;
    end.value = end.max;
    factorScope = scope;
  }
  const first = dates.findIndex(date => date >= start.value);
  const last = dates.findLastIndex(date => date <= end.value);
  const available = new Set();
  const result = first >= 0 && last >= first && start.value && end.value
    ? selected.map(([factor, values]) => ({factor, ...statistics(values, first, last, available)}))
      .filter(row => row.count).sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    : [];
  byId("factor-info").textContent = result.length
    ? `${start.value} 至 ${end.value} · ${available.size} 个有效日期。每日计算横截面 Pearson IC，再对日期等权平均。图中显示前 20 名。`
    : "所选日期范围内没有可用的因子 IC。";
  table("factor-table", ["因子", "平均 IC", "IC 波动", "有效日期"],
    result.map(row => [row.factor, decimal(row.value, 4), decimal(row.deviation, 4), row.count]));
  table("factor-definitions", ["因子", "类型", "定义"],
    overview.catalog.filter(row => !category || row.factor_type === category)
      .map(row => [row.factor, row.factor_type, row.definition]));
  await signedBars("factor-chart", result.slice(0, 20), "平均 IC");
}

async function backtest(isCurrent) {
  const horizon = Number(choice("backtest-horizon"));
  const available = overview.backtests.filter(row => row.horizon === horizon);
  options("backtest-model", modelOrder(available.map(row => row.model)), modelName);
  const model = choice("backtest-model");
  const summary = available.find(row => row.model === model);
  const series = await data("backtest-nav");
  if (!isCurrent()) return;
  const rows = series[`${horizon}/${model}`];
  cards("backtest-cards", [
    ["策略累计收益", percentage(summary.total_return)], ["等权基准收益", percentage(summary.benchmark_total_return)],
    ["年化 Sharpe", decimal(summary.annualized_sharpe, 2)], ["最大回撤", percentage(summary.max_drawdown)],
  ]);
  const dates = rows.map(row => row[0]);
  await plot("backtest-chart", [
    {type: "scatter", mode: "lines", x: dates, y: rows.map(row => row[1]), name: "策略净值", line: {color: gold}},
    {type: "scatter", mode: "lines", x: dates, y: rows.map(row => row[2]), name: "等权基准", line: {color: moon}},
  ], {height: 390, xaxis: {type: "date"}, yaxis: {title: "累计净值", gridcolor: "#324059"}});
}

const views = {prediction, models, factors, backtest};

async function refresh() {
  const id = ++request;
  const isCurrent = () => id === request;
  byId("error").hidden = true;
  byId("status").hidden = false;
  byId("status").textContent = "正在读取结果…";
  try {
    await views[activeView](isCurrent);
  } catch (error) {
    if (isCurrent()) {
      byId("error").textContent = `加载失败。请重试或刷新页面。 ${error.message}`;
      byId("error").hidden = false;
    }
  } finally {
    if (isCurrent()) byId("status").hidden = true;
  }
}

async function init() {
  try {
    overview = await data("overview");
    catalog = new Map(overview.catalog.map(row => [row.factor, row]));
    const categories = [...new Set(overview.catalog.map(row => row.factor_type))];
    byId("run-info").textContent = `预测数据截至 ${overview.forecastDate} · ${Object.keys(overview.names).length} 只股票 · ${new Set(overview.metrics.map(row => row.model)).size} 个模型 · ${overview.forecastHorizons.length} 个预测期限`;
    options("forecast-horizon", overview.forecastHorizons, days);
    options("metric-horizon", [...new Set(overview.metrics.map(row => row.horizon))].sort((a, b) => a - b), days);
    options("factor-horizon", overview.factorHorizons, days);
    options("backtest-horizon", [...new Set(overview.backtests.map(row => row.horizon))].sort((a, b) => a - b), days);
    for (const id of ["importance-category", "factor-category"]) options(id, ["", ...categories], category => category || "全部因子");
    document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => {
      activeView = button.dataset.view;
      document.querySelectorAll(".tab").forEach(tab => {
        const selected = tab === button;
        tab.classList.toggle("active", selected);
        tab.setAttribute("aria-pressed", String(selected));
      });
      document.querySelectorAll(".panel").forEach(panel => panel.classList.toggle("active", panel.id === activeView));
      refresh();
    }));
    document.querySelectorAll(".controls select, .controls input").forEach(control => control.addEventListener("change", refresh));
    byId("ranking").addEventListener("click", event => {
      const stock = event.target.closest("[data-stock]");
      if (!stock) return;
      byId("forecast-stock").value = stock.dataset.stock;
      byId("forecast-stock").focus();
      refresh();
    });
    await refresh();
  } catch (error) {
    byId("status").hidden = true;
    byId("error").textContent = `无法加载项目数据。请刷新页面。 ${error.message}`;
    byId("error").hidden = false;
  }
}

init();
