"""把当前训练结果导出为可直接分享的单文件 HTML 样板。"""

from pathlib import Path
import json

import pandas as pd
from price_source import load_prices, source_signature, FACTOR_CATALOG_FILE
from factors import FEATURES
from factor_selection import PREPROCESSING_VERSION
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results"
OUTPUT_FILE = BASE_DIR / "股票收益预测项目_展示样板.html"

GOLD = "#E4B863"
GOLD_LIGHT = "#F3DCA8"
MOON = "#7C9CC4"
RED = "#A83A3A"
GREEN = "#1F6E55"
TEXT = "rgba(255,255,255,0.76)"
GRID = "rgba(255,255,255,0.09)"


def chart_html(figure: go.Figure, include_plotly: bool = False) -> str:
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.025)",
        font={"color": TEXT, "family": "Aptos, Microsoft YaHei UI, sans-serif"},
        margin={"l": 55, "r": 25, "t": 35, "b": 45},
        hoverlabel={"bgcolor": "#111A28", "font": {"color": "#F2F5F9"}},
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return figure.to_html(
        full_html=False,
        include_plotlyjs=True if include_plotly else False,
        config={"displaylogo": False, "responsive": True, "scrollZoom": False},
    )


def price_figure(prices: pd.DataFrame, prediction: pd.Series) -> go.Figure:
    code = prediction["code"]
    history = prices[prices["code"] == code].sort_values("date").copy()
    horizon = int(prediction["horizon"])
    last_date = history["date"].iloc[-1]
    last_close = history["close"].iloc[-1]
    future_date = last_date + pd.offsets.BDay(horizon)
    target = prediction["reference_target_price"]

    trading_dates = pd.DatetimeIndex(history["date"].dt.normalize())
    missing_dates = pd.date_range(trading_dates.min(), trading_dates.max(), freq="B").difference(trading_dates)
    breaks = [{"bounds": ["sat", "mon"]}]
    if len(missing_dates):
        breaks.append({"values": missing_dates.strftime("%Y-%m-%d").tolist()})

    volume_colors = [RED if close >= open_price else GREEN for open_price, close in zip(history["open"], history["close"])]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.04)
    figure.add_trace(
        go.Candlestick(
            x=history["date"], open=history["open"], high=history["high"],
            low=history["low"], close=history["close"], name="历史 K 线",
            increasing={"line": {"color": RED}, "fillcolor": RED},
            decreasing={"line": {"color": GREEN}, "fillcolor": GREEN},
        ), row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[last_date, future_date], y=[last_close, target], mode="lines+markers",
            name=f"{horizon} 日预测", line={"color": GOLD, "width": 4, "dash": "dash"},
            marker={"color": GOLD, "size": [6, 11]},
        ), row=1, col=1,
    )
    figure.add_trace(
        go.Bar(x=history["date"], y=history["volume"], name="成交量", marker_color=volume_colors),
        row=2, col=1,
    )
    start = last_date - pd.DateOffset(months=6)
    recent = history[history["date"] >= start]
    low = min(recent["low"].min(), target)
    high = max(recent["high"].max(), target)
    padding = max((high - low) * 0.1, high * 0.02)
    figure.update_layout(height=620, hovermode="closest", dragmode="pan", legend={"orientation": "h", "y": 1.04})
    figure.update_xaxes(range=[start, future_date + pd.offsets.BDay(5)], rangebreaks=breaks, rangeslider_visible=False)
    figure.update_yaxes(title="价格 / CNY", range=[low - padding, high + padding], row=1, col=1)
    figure.update_yaxes(title="成交量", rangemode="tozero", tickformat="~s", row=2, col=1)
    return figure


def dataframe_html(data: pd.DataFrame) -> str:
    return data.to_html(index=False, border=0, classes="data-table", escape=True)


def metric(label: str, value: str) -> str:
    return f'<div class="metric"><span>{label}</span><strong>{value}</strong></div>'


def main() -> None:
    latest = pd.read_csv(RESULT_DIR / "latest_predictions.csv", dtype={"code": str})
    prices = load_prices(columns=[])
    metrics = pd.read_csv(RESULT_DIR / "model_metrics.csv")
    factor_ic = pd.read_csv(RESULT_DIR / "factor_ic_summary.csv")
    factor_catalog = pd.read_csv(FACTOR_CATALOG_FILE)
    backtest = pd.read_csv(RESULT_DIR / "backtest_nav.csv", parse_dates=["date"])
    backtest_summary = pd.read_csv(RESULT_DIR / "backtest_summary.csv")
    metadata = json.loads((RESULT_DIR / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("features") != FEATURES or metadata.get("source_sha256") != source_signature()
            or metadata.get("preprocessing_version") != PREPROCESSING_VERSION):
        raise ValueError("现有训练结果与Parquet/因子版本不一致，请先运行 python main.py")

    latest["code"] = latest["code"].str.zfill(6)
    prices["code"] = prices["code"].str.zfill(6)
    default_set = latest[(latest["horizon"] == 1) & (latest["model"] == "random_forest")].sort_values("code")
    prediction = default_set.iloc[0]

    price = price_figure(prices, prediction)
    model_view = metrics[metrics["horizon"] == 1].sort_values("mae")
    model_chart = go.Figure()
    model_chart.add_bar(x=model_view["model"], y=model_view["r2"], name="R²", marker_color=GOLD)
    model_chart.add_bar(x=model_view["model"], y=model_view["prediction_correlation"], name="预测相关系数", marker_color=MOON)
    model_chart.update_layout(height=360, barmode="group", legend={"orientation": "h"})

    factor_view = factor_ic[factor_ic["horizon"] == 1].nlargest(20, "abs_mean_ic").sort_values("mean_ic")
    factor_view = factor_view.merge(factor_catalog, on="factor", how="left")
    factor_chart = go.Figure(go.Bar(
        x=factor_view["mean_ic"], y=factor_view["factor"], orientation="h", marker_color=GOLD,
        customdata=factor_view[["factor_type", "definition"]],
        hovertemplate="%{y}<br>平均 IC：%{x:.4f}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
    ))
    factor_chart.update_layout(height=560)

    nav = backtest[(backtest["horizon"] == 1) & (backtest["model"] == "lasso")]
    summary = backtest_summary[(backtest_summary["horizon"] == 1) & (backtest_summary["model"] == "lasso")].iloc[0]
    backtest_chart = go.Figure()
    backtest_chart.add_scatter(x=nav["date"], y=nav["strategy_nav"], mode="lines", name="策略净值", line={"color": GOLD, "width": 3})
    backtest_chart.add_scatter(x=nav["date"], y=nav["benchmark_nav"], mode="lines", name="等权基准", line={"color": MOON, "width": 2})
    backtest_chart.update_layout(height=430, legend={"orientation": "h"})

    ranking = default_set.nlargest(20, "predicted_return").copy()
    ranking["预测收益率"] = ranking["predicted_return"].map(lambda value: f"{value:+.2%}")
    ranking["参考目标价"] = ranking["reference_target_price"].map(lambda value: f"¥ {value:.2f}")
    ranking["最新收盘价"] = ranking["close"].map(lambda value: f"¥ {value:.2f}")
    ranking_table = dataframe_html(ranking[["code", "name", "最新收盘价", "预测收益率", "参考目标价"]].rename(columns={"code": "代码", "name": "股票"}))

    model_table = model_view.copy()
    model_table["MAE"] = model_table["mae"].map(lambda value: f"{value:.2%}")
    model_table["RMSE"] = model_table["rmse"].map(lambda value: f"{value:.2%}")
    model_table["方向准确率"] = model_table["direction_accuracy"].map(lambda value: f"{value:.1%}")
    model_table = dataframe_html(model_table[["model", "best_params", "r2", "MAE", "RMSE", "方向准确率", "prediction_correlation"]].rename(columns={
        "model": "模型", "best_params": "验证集最优参数", "r2": "R²", "prediction_correlation": "预测相关系数"
    }))

    price_fragment = chart_html(price, include_plotly=True)
    model_fragment = chart_html(model_chart)
    factor_fragment = chart_html(factor_chart)
    backtest_fragment = chart_html(backtest_chart)
    generated_at = metadata.get("generated_at", "—")
    latest_date = metadata.get("latest_market_date", "—")

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>股票收益预测课程项目 · 展示样板</title>
<style>
:root{{--night:#0B1322;--deep:#060A13;--gold:{GOLD};--gold-light:{GOLD_LIGHT};--moon:{MOON};--text:rgba(255,255,255,.94);--soft:rgba(255,255,255,.76);--muted:rgba(255,255,255,.54);--line:rgba(255,255,255,.15);--glass:rgba(255,255,255,.08);--shadow:0 16px 40px rgba(3,7,18,.5),inset 0 1px 0 rgba(255,255,255,.22),inset 0 -1px 0 rgba(2,6,16,.35)}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;color:var(--text);font-family:Aptos,"Microsoft YaHei UI",sans-serif;background:radial-gradient(circle at 88% 4%,rgba(124,156,196,.28),transparent 55%),radial-gradient(circle at 7% 88%,rgba(51,81,122,.38),transparent 60%),radial-gradient(circle at 54% 42%,rgba(228,184,99,.11),transparent 34%),linear-gradient(145deg,var(--deep),var(--night) 48%,#0A101C);background-attachment:fixed;min-height:100vh}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;opacity:.025;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}}
.shell{{width:100%;padding:28px clamp(18px,3vw,56px) 60px}} .top{{display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:26px}} h1,h2,h3{{font-family:"Iowan Old Style","Noto Serif SC","Songti SC",serif}} h1{{font-size:34px;margin:0}} .sample{{color:var(--gold);font-size:12px;margin-top:5px}}
.tabs{{display:flex;gap:6px;padding:6px;background:rgba(255,255,255,.055);backdrop-filter:blur(60px) saturate(180%);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow)}} .tab{{appearance:none;border:1px solid transparent;border-radius:16px;background:transparent;color:var(--muted);padding:12px 20px;cursor:pointer;transition:.5s cubic-bezier(.16,1,.3,1)}} .tab:hover{{background:rgba(255,255,255,.08);color:var(--text)}} .tab.active{{color:var(--gold-light);background:rgba(255,255,255,.1);border-color:rgba(228,184,99,.4)}}
.panel{{display:none}} .panel.active{{display:block}} .kicker{{color:var(--gold);font:650 12px Consolas,monospace}} .section-title{{font-size:26px;margin:3px 0 4px}} .desc{{color:var(--muted);font-size:13px;margin-bottom:18px}} .grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}} .metric,.glass{{background:var(--glass);background-image:linear-gradient(to bottom,rgba(255,255,255,.12),transparent 50%);backdrop-filter:blur(60px) saturate(180%);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow)}} .metric{{padding:17px 18px}} .metric span{{display:block;color:var(--muted);font-size:12px}} .metric strong{{display:block;color:var(--gold-light);font-size:25px;margin-top:7px}} .glass{{padding:18px;margin:18px 0;overflow:hidden}} .chart-title{{font-size:19px;margin:4px 0 12px}}
.data-table{{border-collapse:collapse;width:100%;font-size:13px;color:var(--soft)}} .data-table th,.data-table td{{padding:11px 12px;border-bottom:1px solid rgba(255,255,255,.09);text-align:left}} .data-table th{{color:var(--gold-light);font-weight:650}} .note{{color:var(--muted);font-size:12px;line-height:1.7;margin-top:12px}} .footer{{color:var(--muted);font-size:12px;border-top:1px solid rgba(255,255,255,.1);padding-top:18px;margin-top:34px}}
@media(max-width:850px){{.top{{align-items:flex-start;flex-direction:column}}.tabs{{width:100%;overflow:auto}}.tab{{padding:10px 12px;white-space:nowrap}}.grid4{{grid-template-columns:repeat(2,1fr)}}}} @media(max-width:520px){{.grid4{{grid-template-columns:1fr}}h1{{font-size:28px}}}}
</style>
</head>
<body><main class="shell">
<header class="top"><div><h1>股票收益预测课程项目</h1><div class="sample">静态展示样板 · 数据截止 {latest_date} · 生成于 {generated_at}</div></div><nav class="tabs">
<button class="tab active" data-target="prediction">首页预测</button><button class="tab" data-target="models">模型表现</button><button class="tab" data-target="factors">因子表现</button><button class="tab" data-target="backtest">策略回测</button></nav></header>

<section id="prediction" class="panel active"><div class="kicker">01 / PREDICTION</div><h2 class="section-title">最新收益预测</h2><div class="desc">静态样板口径：1个交易日 · Random Forest · {prediction['code']} {prediction['name']}</div>
<div class="grid4">{metric('最近收盘价 / CNY', f"¥ {prediction['close']:.2f}")}{metric('预测 1 日收益率', f"{prediction['predicted_return']:+.2%}")}{metric('参考目标价 / CNY', f"¥ {prediction['reference_target_price']:.2f}")}{metric('预测方向', '上涨' if prediction['predicted_return'] > 0 else '下跌或持平')}</div>
<div class="glass"><h3 class="chart-title">{prediction['code']} {prediction['name']} · 历史 K 线与预测终点</h3>{price_fragment}<div class="note">深红K线表示上涨，深绿K线表示下跌；香槟金虚线连接最新收盘价与预测终点。图表可悬停、拖动和使用工具栏缩放。</div></div>
<div class="glass"><h3 class="chart-title">股票池最新预测排名 · 前20名</h3>{ranking_table}</div></section>

<section id="models" class="panel"><div class="kicker">02 / MODELS</div><h2 class="section-title">模型表现比较</h2><div class="desc">1个交易日预测周期的测试集结果。</div><div class="glass">{model_fragment}</div><div class="glass">{model_table}</div></section>

<section id="factors" class="panel"><div class="kicker">03 / FACTORS</div><h2 class="section-title">因子预测能力</h2><div class="desc">1日未来收益的 |平均 IC| 前20个因子，悬停可查看类型和定义。</div><div class="glass">{factor_fragment}</div></section>

<section id="backtest" class="panel"><div class="kicker">04 / BACKTEST</div><h2 class="section-title">样本外策略回测</h2><div class="desc">静态样板口径：1个交易日 · Lasso。</div><div class="grid4">{metric('策略累计收益', f"{summary['total_return']:+.2%}")}{metric('等权基准收益', f"{summary['benchmark_total_return']:+.2%}")}{metric('年化 Sharpe', f"{summary['annualized_sharpe']:.2f}")}{metric('最大回撤', f"{summary['max_drawdown']:.2%}")}</div><div class="glass">{backtest_fragment}</div></section>

<div class="footer"><b>方法与风险提示</b><br>本文件使用项目当前训练结果制作，仅作为课程项目与界面展示样板，不构成投资建议。样板数据固定在导出时点，不会自动更新。</div>
</main>
<script>
document.querySelectorAll('.tab').forEach(button=>button.addEventListener('click',()=>{{
  document.querySelectorAll('.tab').forEach(item=>item.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(item=>item.classList.remove('active'));
  button.classList.add('active'); document.getElementById(button.dataset.target).classList.add('active');
  setTimeout(()=>document.querySelectorAll('#'+button.dataset.target+' .js-plotly-plot').forEach(plot=>Plotly.Plots.resize(plot)),30);
}}));
</script></body></html>"""
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"已生成：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
