"""Streamlit 展示页：Nocturne Glassmorphism 夜航玻璃拟态。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from price_source import load_prices, FACTOR_CATALOG_FILE
from report_data import (
    ReportError, load_report, report_cache_key, model_label, strategy_label, nav_label,
    model_capabilities, select_backtest, importance_view, existing_columns,
    COLUMN_LABELS, PERCENT_COLUMNS, STRATEGY_NOTES, COST_NOTE, IC_NOTE,
)
from report_charts import price_chart, backtest_figure, model_figure, ic_figure


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results"

st.set_page_config(page_title="股票收益预测 · Stock Return Lab", page_icon="◈", layout="wide")

st.markdown(
    """
<style>
:root {
  --night: #0B1322;
  --night-deep: #060A13;
  --moon: #7C9CC4;
  --gold: #E4B863;
  --gold-light: #F3DCA8;
  --text: rgba(255,255,255,.94);
  --text-soft: rgba(255,255,255,.76);
  --text-muted: rgba(255,255,255,.54);
  --glass: rgba(255,255,255,.08);
  --glass-soft: rgba(255,255,255,.055);
  --glass-hover: rgba(255,255,255,.12);
  --line: rgba(255,255,255,.15);
  --line-hover: rgba(255,255,255,.30);
  --radius: 24px;
  --spring: cubic-bezier(.16,1,.3,1);
  --glass-shadow: 0 16px 40px rgba(3,7,18,.50), inset 0 1px 0 rgba(255,255,255,.22), inset 0 -1px 0 rgba(2,6,16,.35);
}
html, body, [class*="css"] {
  font-family: "Aptos", "Segoe UI Variable", "Microsoft YaHei UI", sans-serif;
}
[data-testid="stMain"] {
  overflow-y: scroll !important;
  scrollbar-gutter: stable;
  overscroll-behavior-y: auto;
}
[data-testid="stMain"]::-webkit-scrollbar { width: 12px; }
[data-testid="stMain"]::-webkit-scrollbar-track { background: rgba(255,255,255,.04); }
[data-testid="stMain"]::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,.28);
  border: 3px solid transparent;
  border-radius: 12px;
  background-clip: padding-box;
}
[data-testid="stMain"]::-webkit-scrollbar-thumb:hover { background-color: rgba(255,255,255,.42); }
[data-testid="stAppViewContainer"] {
  color: var(--text);
  background:
    radial-gradient(circle at 88% 4%, rgba(124,156,196,.28) 0, rgba(124,156,196,.12) 24%, transparent 57%),
    radial-gradient(circle at 7% 88%, rgba(51,81,122,.38) 0, rgba(51,81,122,.14) 28%, transparent 62%),
    radial-gradient(circle at 54% 42%, rgba(228,184,99,.11) 0, transparent 34%),
    linear-gradient(145deg, var(--night-deep) 0%, var(--night) 48%, #0A101C 100%);
  background-attachment: fixed;
}
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: .025;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.95'/%3E%3C/svg%3E");
}
[data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
[data-testid="stHeader"] {
  background: rgba(6,10,19,.18);
  backdrop-filter: blur(60px) saturate(180%);
  -webkit-backdrop-filter: blur(60px) saturate(180%);
  border-bottom: 1px solid rgba(255,255,255,.08);
}
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.block-container {
  width: 100%;
  max-width: none;
  padding: 1.5rem clamp(1rem, 3vw, 3.5rem) 4rem;
}
.block-container > div { width: 100%; min-width: 0; }
h1, h2, h3, h4 {
  color: var(--text);
  letter-spacing: -.02em;
  font-family: "Iowan Old Style", "Noto Serif SC", "Songti SC", serif;
}
p, label, [data-testid="stMarkdownContainer"] { color: var(--text-soft); }
p { line-height: 1.72; }
.project-title {
  color: var(--text);
  font-family: "Iowan Old Style", "Noto Serif SC", "Songti SC", serif;
  font-size: 34px;
  font-weight: 650;
  letter-spacing: -.035em;
  margin: 10px 0 28px;
  text-shadow: 0 8px 30px rgba(3,7,18,.65);
}
.section-kicker {
  color: var(--gold);
  font: 650 12px ui-monospace, "Cascadia Mono", Consolas, monospace;
  letter-spacing: .07em;
}
.section-title {
  color: var(--text);
  font-family: "Iowan Old Style", "Noto Serif SC", "Songti SC", serif;
  font-size: 25px;
  font-weight: 650;
  margin: 3px 0 4px;
}
.section-desc { color: var(--text-muted); font-size: 13px; margin-bottom: 18px; }
.note-box {
  color: var(--text-soft);
  background: var(--glass-soft);
  backdrop-filter: blur(40px) saturate(180%);
  -webkit-backdrop-filter: blur(40px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 13px 16px;
  font-size: 12px;
  margin-top: 14px;
  box-shadow: 0 8px 24px rgba(3,7,18,.45), inset 0 1px 0 rgba(255,255,255,.18), inset 0 -1px 0 rgba(2,6,16,.30);
}
.footer-note {
  border-top: 1px solid rgba(255,255,255,.11);
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.8;
  padding-top: 20px;
  margin-top: 34px;
}
.footer-note b { color: var(--text-soft); }

/* Glass navigation sits beside the title; the content area stays unboxed. */
.stTabs { position: relative; min-width: 0; }
.stTabs [data-baseweb="tab-list"] {
  position: absolute;
  z-index: 5;
  right: 0;
  top: -83px;
  width: max-content;
  max-width: min(62vw, 720px);
  gap: 6px;
  background: var(--glass-soft);
  backdrop-filter: blur(60px) saturate(180%);
  -webkit-backdrop-filter: blur(60px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 6px;
  box-shadow: var(--glass-shadow);
}
.stTabs [data-baseweb="tab"] {
  color: var(--text-muted);
  border: 1px solid transparent;
  border-radius: 16px;
  padding: 12px 20px;
  font-size: 13px;
  transition: all 500ms var(--spring);
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text);
  background: rgba(255,255,255,.08);
  border-color: var(--line);
}
.stTabs [aria-selected="true"] {
  color: var(--gold-light) !important;
  font-weight: 700;
  background: rgba(255,255,255,.10) !important;
  border-color: rgba(228,184,99,.40) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.20), inset 0 -1px 0 rgba(2,6,16,.28);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding: 28px 2px 8px; }

/* Inputs */
div[data-baseweb="select"] > div,
[data-testid="stDateInput"] input {
  color: var(--text);
  background: rgba(255,255,255,.06);
  backdrop-filter: blur(40px) saturate(180%);
  -webkit-backdrop-filter: blur(40px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.15), inset 0 -1px 0 rgba(2,6,16,.30);
  transition: all 500ms var(--spring);
}
div[data-baseweb="select"] > div:hover,
[data-testid="stDateInput"] input:hover {
  background: rgba(255,255,255,.10);
  border-color: var(--line-hover);
}
div[data-baseweb="select"] > div:focus-within,
[data-testid="stDateInput"] input:focus {
  outline: none;
  background: rgba(255,255,255,.10);
  border-color: rgba(255,255,255,.35);
  box-shadow: 0 0 0 3px rgba(228,184,99,.15), inset 0 1px 0 rgba(255,255,255,.25), inset 0 -1px 0 rgba(2,6,16,.30);
}
div[data-baseweb="select"] * { color: var(--text-soft); }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="calendar"] {
  color: var(--text);
  background: rgba(255,255,255,.10);
  backdrop-filter: blur(60px) saturate(180%);
  -webkit-backdrop-filter: blur(60px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: 20px;
  box-shadow: var(--glass-shadow);
}
[data-baseweb="popover"] [role="option"] { color: var(--text-soft); }
[data-baseweb="popover"] [role="option"]:hover { background: rgba(255,255,255,.10); color: var(--text); }
[data-testid="stWidgetLabel"] p { color: var(--text-soft); }
[data-testid="stForm"] { border: 0; padding: 0; }

/* Champagne is reserved for the primary action and key figures. */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  min-height: 42px;
  color: var(--gold-light);
  background: rgba(228,184,99,.15);
  backdrop-filter: blur(40px) saturate(180%);
  -webkit-backdrop-filter: blur(40px) saturate(180%);
  border: 1px solid rgba(228,184,99,.40);
  border-radius: 16px;
  font-weight: 700;
  box-shadow: 0 4px 20px rgba(228,184,99,.15), inset 0 1px 0 rgba(243,220,168,.35), inset 0 -1px 0 rgba(2,6,16,.30);
  transition: all 500ms var(--spring);
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  color: var(--gold-light);
  background: rgba(228,184,99,.22);
  border-color: rgba(228,184,99,.55);
  box-shadow: 0 10px 32px rgba(3,7,18,.55), 0 4px 24px rgba(228,184,99,.18), inset 0 1px 0 rgba(255,255,255,.35);
  transform: translateY(-2px);
}
.stButton > button:active, [data-testid="stFormSubmitButton"] > button:active { transform: scale(.97); }
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible {
  outline: none;
  box-shadow: 0 0 0 3px rgba(228,184,99,.18), 0 8px 24px rgba(3,7,18,.45);
}

/* Neutral colorless glass surfaces. */
div[data-testid="stMetric"] {
  position: relative;
  overflow: hidden;
  background: var(--glass);
  background-image: linear-gradient(to bottom, rgba(255,255,255,.12), transparent 50%);
  backdrop-filter: blur(60px) saturate(180%);
  -webkit-backdrop-filter: blur(60px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 17px 18px;
  box-shadow: var(--glass-shadow);
  transition: all 500ms var(--spring);
}
div[data-testid="stMetric"]:hover {
  background-color: var(--glass-hover);
  border-color: var(--line-hover);
  transform: translateY(-4px);
  box-shadow: 0 24px 64px rgba(3,7,18,.60), inset 0 1px 0 rgba(255,255,255,.32), inset 0 -1px 0 rgba(2,6,16,.35);
}
[data-testid="stMetricLabel"] { color: var(--text-muted); font-size: 12px; }
[data-testid="stMetricValue"] { color: var(--gold-light); font-size: 25px; }
[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
[data-testid="stPlotlyChart"],
[data-testid="stExpander"] {
  overflow: hidden;
  background: var(--glass-soft);
  background-image: linear-gradient(to bottom, rgba(255,255,255,.10), transparent 50%);
  backdrop-filter: blur(60px) saturate(180%);
  -webkit-backdrop-filter: blur(60px) saturate(180%);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--glass-shadow);
}
[data-testid="stExpander"] summary { color: var(--text-soft); }
[data-testid="stDataFrame"] { color: var(--text-soft); }
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container {
  overscroll-behavior: auto;
}
hr { border-color: rgba(255,255,255,.11); }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; }
  div[data-testid="stMetric"]:hover, .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { transform: none; }
}
@media(max-width:820px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .project-title { font-size: 28px; }
  .stTabs [data-baseweb="tab-list"] {
    position: relative;
    top: auto;
    right: auto;
    width: 100%;
    max-width: none;
    gap: 2px;
    padding: 4px;
    border-radius: 18px;
  }
  .stTabs [data-baseweb="tab"] { padding: 10px 8px; font-size: 12px; border-radius: 14px; }
  .stTabs [data-baseweb="tab-panel"] { padding: 22px 0 8px; }
  div[data-testid="stMetric"] { backdrop-filter: blur(30px) saturate(180%); -webkit-backdrop-filter: blur(30px) saturate(180%); }
}
@media(min-width:821px) and (max-width:1080px) {
  .project-title { font-size: 29px; }
  .stTabs [data-baseweb="tab-list"] { top: -79px; max-width: 66vw; }
  .stTabs [data-baseweb="tab"] { padding-left: 14px; padding-right: 14px; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_report(cache_key):
    return load_report(RESULT_DIR)


@st.cache_data(show_spinner=False)
def read_prices(source_sha256):
    return load_prices(columns=[])


def pct(value):
    return f"{value:+.2%}" if pd.notna(value) else "—"


def section_header(number, title, description):
    st.markdown(
        f'<div class="section-kicker">{number}</div><div class="section-title">{title}</div>'
        f'<div class="section-desc">{description}</div>', unsafe_allow_html=True,
    )


def show_table(frame, columns=None, height=None):
    if frame.empty:
        st.info("本次实验没有生成此项结果。")
        return
    view = frame[existing_columns(frame, columns)].copy() if columns else frame.drop(
        columns=["experiment_id", "run_id"], errors="ignore"
    ).copy()
    if "model" in view:
        view["model"] = view.model.map(model_label)
    if "strategy" in view:
        view["strategy"] = view.strategy.map(strategy_label)
    for column in PERCENT_COLUMNS.intersection(view.columns):
        view[column] = view[column].map(pct)
    if "cost_bps" in view:
        view["cost_bps"] = view.cost_bps.map(lambda value: f"{value:g} bp")
    options = {"width": "stretch", "hide_index": True}
    if height:
        options["height"] = height
    st.dataframe(view.rename(columns=COLUMN_LABELS), **options)


def choose_model_period(frame, key):
    left, right = st.columns(2)
    horizon = left.selectbox("预测周期", sorted(int(h) for h in frame.horizon.unique()),
                            format_func=lambda h: f"{h} 个交易观察", key=key + "_h")
    available = frame[frame.horizon == horizon]
    model = right.selectbox("模型", sorted(available.model.unique(), key=model_label),
                            format_func=model_label, key=key + "_m")
    return available[available.model == model].copy(), model, horizon


def plot(figure):
    st.plotly_chart(figure, width="stretch",
                    config={"displaylogo": False, "responsive": True, "scrollZoom": False})


def render_predictions(report):
    section_header("01 / PREDICTION", "最新收益预测", "只展示本实验实际生成的生产模型预测。")
    latest = report.table("latest_predictions.csv")
    if latest.empty:
        st.info("本次实验没有生成最新预测；测试评价仍可在其他页查看。")
        return
    subset, model, horizon = choose_model_period(latest, "prediction")
    names = dict(zip(subset.code, subset["name"]))
    code = st.selectbox("股票", sorted(subset.code.unique()),
                        format_func=lambda value: f"{value}　{names.get(value, '')}", key="prediction_stock")
    row = subset[subset.code == code].iloc[0]
    cols = st.columns(4)
    cols[0].metric("参考收盘价", f"¥ {row['close']:.2f}")
    cols[1].metric(f"预测{horizon}日收益率", pct(row.predicted_return))
    cols[2].metric("参考目标价", f"¥ {row.reference_target_price:.2f}")
    cols[3].metric("本组合可用预测股票数", subset.code.nunique())
    st.caption(f"预测形成日期：{row.date:%Y-%m-%d}。实验股票池共{report.stock_count}只；最新日期缺记录时不会伪造预测。")
    try:
        prices = read_prices(report.metadata["experiment_identity"]["input_sha256"]["prices"])
        history = prices[prices.code == code]
        plot(price_chart(history, row, horizon))
        st.caption("预测终点日期使用工作日近似，不是交易所精确日历；虚线仅连接预测终点，不代表逐日价格路径。")
    except (OSError, ValueError) as error:
        st.warning(f"历史行情图暂不可用：{error}")
    st.markdown("#### 当前模型与周期的全部预测排名")
    show_table(subset.sort_values("predicted_return", ascending=False),
               ["code", "name", "date", "close", "predicted_return", "reference_target_price"], height=380)


def render_metrics(report):
    section_header("02 / MODELS", "模型测试评价", "按实际完成的模型与周期比较测试集指标，不固定为三类模型。")
    metrics = report.table("model_metrics.csv")
    horizon = st.selectbox("评价周期", sorted(int(h) for h in metrics.horizon.unique()),
                           format_func=lambda h: f"{h} 个交易观察", key="metrics_h")
    view = metrics[metrics.horizon == horizon].sort_values("mae")
    plot(model_figure(view))
    show_table(view, [
        "model", "best_params", "validation_mae", "mae", "rmse", "r2", "r2_oos_zero",
        "direction_accuracy", "prediction_correlation", "n_train", "n_validation", "n_test",
        "n_train_stocks", "n_validation_stocks", "n_test_stocks", "test_start", "test_end",
    ])
    st.caption("R²相对测试均值基线；相对零收益基线R²采用零预测。模型参数由验证集选择，测试集不参与调参。")
    with st.expander("各模型输出覆盖情况"):
        show_table(model_capabilities(report))


def render_ic(report):
    section_header("03 / PREDICTION IC", "预测IC与逐股误差", IC_NOTE)
    daily = report.table("ic_daily.csv")
    summary = report.table("ic_summary.csv")
    if daily.empty and summary.empty:
        st.info("本次实验没有可用的预测IC结果。历史单因子IC文件不会作为替代。")
        return
    source = daily if not daily.empty else summary
    _, model, horizon = choose_model_period(source, "ic")
    if not daily.empty:
        selected = daily[(daily.model == model) & (daily.horizon == horizon)]
        plot(ic_figure(selected))
    if not summary.empty:
        show_table(summary[(summary.model == model) & (summary.horizon == horizon)],
                   ["metric", "mean", "std", "icir", "t_stat", "positive_ratio", "n_days"])
    stock = report.table("per_stock_metrics.csv")
    if not stock.empty:
        st.markdown("#### 逐股测试误差")
        show_table(stock[(stock.model == model) & (stock.horizon == horizon)].sort_values("mae"),
                   ["code", "n_test", "mae", "correlation"], height=350)
    st.caption("单因子IC与共同因子时间序列IC未由当前训练流程生成，本页不读取旧版因子IC排行榜。")


def render_backtest(report):
    section_header("04 / BACKTEST", "样本外组合评价", "分别查看本实验实际输出的策略与费用设置。")
    summary = report.table("backtest_summary.csv")
    if summary.empty:
        st.info("本次实验没有可用组合回测结果，模型评价仍可查看。")
        return
    selected, model, horizon = choose_model_period(summary, "backtest")
    left, right = st.columns(2)
    strategy = left.selectbox("组合策略", sorted(selected.strategy.unique()), format_func=strategy_label, key="bt_strategy")
    selected = selected[selected.strategy == strategy]
    cost = right.selectbox("费用设置", sorted(selected.cost_bps.unique()),
                           format_func=lambda value: f"{value:g} bp", key="bt_cost")
    nav, selected_summary = select_backtest(report, model, horizon, strategy, cost)
    if selected_summary.empty:
        st.info("当前选择没有回测摘要。")
        return
    row = selected_summary.iloc[0]
    cols = st.columns(4)
    label = "累计相对收益" if strategy == "long_top5_vs_equal_weight" else "累计净收益"
    cols[0].metric(label, pct(row.cumulative_return))
    cols[1].metric("年化Sharpe", f"{row.annualized_sharpe:.2f}" if pd.notna(row.annualized_sharpe) else "—")
    cols[2].metric("最大回撤", pct(row.max_drawdown))
    cols[3].metric("平均换手率", pct(row.average_turnover))
    st.caption(STRATEGY_NOTES.get(strategy, "按当前实验记录的组合口径展示。") + " " + COST_NOTE)
    if nav.empty:
        st.info("当前选择没有净值序列。")
    else:
        st.markdown("#### " + nav_label(strategy))
        plot(backtest_figure(nav))
        with st.expander("逐期收益、换手和费用"):
            show_table(nav, ["date", "gross_return", "turnover", "cost", "net_return", "nav"])
    st.caption(f"调仓期数：{int(row.periods)}。最大回撤计入初始净值1，沿用负数表示。")


def render_factors(report):
    section_header("05 / FACTORS", "因子与预处理", "展示模型使用的因子、预处理维数和消融评价。")
    metrics = report.table("model_metrics.csv")
    _, model, horizon = choose_model_period(metrics, "factors")
    preprocessing = report.table("preprocessing_summary.csv")
    if not preprocessing.empty:
        show_table(preprocessing[(preprocessing.model == model) & (preprocessing.horizon == horizon)],
                   ["stage", "selection_config", "selected_factor_count", "model_input_dimensions", "pca_explained_variance"])
    top = st.selectbox("显示重要性数量", [10, 20, 30], index=1, key="importance_top")
    importance = importance_view(report, model, horizon, top)
    if importance.empty:
        st.info("该模型没有提供可用的内置因子重要性，不将占位零值解释为真实重要性。")
    else:
        plot(go.Figure(go.Bar(x=importance.importance, y=importance.factor, orientation="h",
                             marker_color="#E4B863")).update_layout(height=420))
        show_table(importance, ["factor", "importance", "importance_method", "stage"])
        if "importance_method" in importance and importance.importance_method.eq("pca_squared_loading_proxy").any():
            st.caption("PCA树模型的重要性按平方载荷分摊，是解释代理；线性模型系数对应标准化后的输入。")
    ablation = report.table("factor_group_ablation.csv")
    with st.expander("因子组消融结果"):
        show_table(ablation, ["horizon", "group", "n_factors", "validation_mae", "test_mae", "r2_oos_zero", "direction_accuracy"])
    with st.expander("当前因子目录与定义"):
        if FACTOR_CATALOG_FILE.is_file():
            try:
                show_table(pd.read_csv(FACTOR_CATALOG_FILE),
                           ["factor", "factor_type", "definition", "ic_scope"], height=380)
            except (OSError, ValueError):
                st.info("因子目录暂不可读，模型评价不受影响。")
        else:
            st.info("当前没有因子目录文件。")


def render_dashboard():
    st.markdown('<div class="project-title">股票收益预测 · 当前实验</div>', unsafe_allow_html=True)
    try:
        report = read_report(report_cache_key(RESULT_DIR))
    except ReportError as error:
        st.warning(str(error))
        st.info("这里只展示成功发布的当前实验。修改代码不会自动训练，旧版静态展示样板仍保留。")
        st.stop()
    cols = st.columns(4)
    cols[0].metric("实验股票池", report.stock_count)
    cols[1].metric("候选因子", report.feature_count)
    cols[2].metric("完成模型数", len(report.models))
    cols[3].metric("完成周期", " / ".join(str(h) for h in report.metadata["horizons"]))
    st.caption(f"实验 {report.experiment_id[:12]} · 结果生成于 {report.metadata.get('generated_at', '—')} · "
               f"行情截止 {report.metadata.get('latest_market_date', '—')}")
    for warning in report.warnings:
        st.info(warning)
    tabs = st.tabs(["最新预测", "模型评价", "预测IC与逐股误差", "组合回测", "因子与预处理"])
    for tab, render in zip(tabs, [render_predictions, render_metrics, render_ic, render_backtest, render_factors]):
        with tab:
            render(report)
    st.caption("结果固定于当前成功实验；参考目标价由预测收益直接换算。历史回测和预测仅用于研究与课程展示。")


if __name__ == "__main__":
    render_dashboard()
