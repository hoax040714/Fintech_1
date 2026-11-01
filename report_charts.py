"""网页和单文件报告共用的图表，不计算或更改模型预测。"""

from html import escape

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from report_data import model_label, strategy_label, nav_label


def price_chart(history: pd.DataFrame, prediction: pd.Series, horizon: int) -> go.Figure:
    """股票交易图：K 线、成交量、非连续交易日和预测终点。"""
    history = history[["date", "open", "high", "low", "close", "volume"]].copy()
    history = history[history["date"] <= pd.Timestamp(prediction["date"])].sort_values("date")
    if history.empty:
        return go.Figure().add_annotation(text="没有可用历史行情", showarrow=False)
    last_date = history["date"].iloc[-1]
    last_close = history["close"].iloc[-1]
    future_date = last_date + pd.offsets.BDay(horizon)
    target_price = prediction["reference_target_price"]
    initial_start = last_date - pd.DateOffset(months=6)
    initial_end = future_date + pd.offsets.BDay(5)
    initial_history = history[history["date"] >= initial_start]
    visible_low = min(initial_history["low"].min(), last_close, target_price)
    visible_high = max(initial_history["high"].max(), last_close, target_price)
    price_padding = max((visible_high - visible_low) * 0.10, visible_high * 0.02)
    visible_volume = initial_history["volume"].max() * 1.15

    trading_dates = pd.DatetimeIndex(history["date"].dt.normalize())
    weekdays = pd.date_range(trading_dates.min(), trading_dates.max(), freq="B")
    missing_trade_dates = weekdays.difference(trading_dates)
    date_breaks = [{"bounds": ["sat", "mon"]}]
    if len(missing_trade_dates):
        date_breaks.append({"values": missing_trade_dates.strftime("%Y-%m-%d").tolist()})

    volume_colors = [
        "rgba(168,58,58,0.76)" if close >= open_price else "rgba(31,110,85,0.76)"
        for open_price, close in zip(history["open"], history["close"])
    ]

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.03,
    )
    figure.add_trace(
        go.Candlestick(
            x=history["date"],
            open=history["open"],
            high=history["high"],
            low=history["low"],
            close=history["close"],
            name="历史 K 线",
            increasing={"line": {"color": "#A83A3A", "width": 1.4}, "fillcolor": "#A83A3A"},
            decreasing={"line": {"color": "#1F6E55", "width": 1.4}, "fillcolor": "#1F6E55"},
            hovertext=[
                f"日期：{r.date:%Y-%m-%d}<br>开盘：¥ {r.open:.2f}<br>最高：¥ {r.high:.2f}"
                f"<br>最低：¥ {r.low:.2f}<br>收盘：¥ {r.close:.2f}"
                for r in history.itertuples()
            ],
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[last_date, future_date],
            y=[last_close, target_price],
            mode="lines+markers",
            name=f"{horizon} 日预测（日期为工作日近似）",
            line={"color": "#E4B863", "width": 4, "dash": "dash"},
            marker={"color": "#E4B863", "size": [6, 11]},
            hovertemplate="日期：%{x|%Y-%m-%d}<br>预测价格：¥ %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=history["date"],
            y=history["volume"],
            name="成交量",
            marker={"color": volume_colors, "line": {"width": 0}},
            hovertemplate="日期：%{x|%Y-%m-%d}<br>成交量：%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_annotation(
        x=future_date,
        y=target_price,
        text=f"¥ {target_price:.2f}",
        showarrow=False,
        xanchor="left",
        xshift=9,
        yshift=12,
        font={"color": "#F3DCA8", "size": 12},
        row=1,
        col=1,
    )
    figure.update_layout(
        height=620,
        margin={"l": 60, "r": 40, "t": 18, "b": 55},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.035)",
        hovermode="closest",
        dragmode="pan",
        legend={"orientation": "h", "y": 1.03, "x": 0, "font": {"color": "rgba(255,255,255,0.72)"}},
        bargap=0.08,
        font={"color": "rgba(255,255,255,0.72)", "family": "Aptos, Segoe UI, sans-serif"},
        hoverlabel={"bgcolor": "#111A28", "bordercolor": "rgba(255,255,255,0.20)", "font": {"color": "#F3F6FA"}},
    )
    figure.update_xaxes(
        range=[initial_start, initial_end],
        rangebreaks=date_breaks,
        rangeslider_visible=False,
        showgrid=False,
        showspikes=True,
        spikecolor="rgba(255,255,255,0.38)",
        spikedash="dot",
        spikemode="across",
        spikesnap="cursor",
    )
    figure.update_xaxes(title="交易日", row=2, col=1)
    figure.update_yaxes(
        title="价格 / CNY",
        range=[visible_low - price_padding, visible_high + price_padding],
        fixedrange=True,
        gridcolor="rgba(255,255,255,0.09)",
        showspikes=True,
        spikecolor="rgba(255,255,255,0.38)",
        spikedash="dot",
        spikemode="across",
        spikesnap="cursor",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title="成交量",
        range=[0, visible_volume],
        fixedrange=True,
        showgrid=False,
        rangemode="tozero",
        tickformat="~s",
        row=2,
        col=1,
    )
    return figure
def backtest_figure(frame, selectable=False):
    figure = go.Figure()
    combinations = list(frame.groupby(["model", "horizon", "strategy", "cost_bps"], sort=True))
    labels = []
    for index, ((model, horizon, strategy, cost), part) in enumerate(combinations):
        label = f"{model_label(model)} · {int(horizon)}日 · {strategy_label(strategy)} · {cost:g} bp"
        labels.append(escape(label))
        part = part.sort_values("date")
        figure.add_scatter(x=part.date, y=part.nav, mode="lines", name=escape(nav_label(strategy)),
                           line={"color": "#E4B863", "width": 3},
                           meta={"model": model, "horizon": int(horizon), "strategy": strategy,
                                 "cost_bps": float(cost), "label": escape(label)},
                           visible=(not selectable or index == 0))
    figure.update_layout(height=430, title=labels[0] if labels else "组合回测", showlegend=False)
    figure.update_xaxes(title="调仓日期")
    figure.update_yaxes(title="净值 / 相对收益复利值")
    if selectable and labels:
        figure.update_layout(updatemenus=[{
            "buttons": [{"label": label, "method": "update",
                         "args": [{"visible": [i == selected for i in range(len(labels))]},
                                  {"title": label}]} for selected, label in enumerate(labels)],
            "direction": "down", "x": 0, "y": 1.20, "xanchor": "left",
        }])
    return figure


def model_figure(metrics):
    figure = go.Figure()
    horizons = sorted(metrics.horizon.unique())
    fields = [("r2", "R²"), ("r2_oos_zero", "相对零收益基线R²"), ("prediction_correlation", "预测相关系数")]
    owners = []
    for horizon in horizons:
        view = metrics[metrics.horizon == horizon].sort_values("mae")
        for field, label in fields:
            if field not in view:
                continue
            figure.add_bar(x=[escape(model_label(name)) for name in view.model],
                           y=view[field], name=label, visible=(horizon == horizons[0]))
            owners.append(horizon)
    figure.update_layout(height=370, barmode="group", legend={"orientation": "h"},
                         title=f"测试集指标 · {int(horizons[0])}日" if horizons else "模型评价")
    if len(horizons) > 1:
        figure.update_layout(updatemenus=[{"buttons": [
            {"label": f"{int(horizon)}日", "method": "update",
             "args": [{"visible": [owner == horizon for owner in owners]},
                      {"title": f"测试集指标 · {int(horizon)}日"}]} for horizon in horizons
        ], "direction": "down", "x": 0, "y": 1.2}])
    return figure


def ic_figure(frame, selectable=False):
    figure = go.Figure()
    groups = list(frame.groupby(["model", "horizon"], sort=True))
    owners, labels = [], []
    for index, ((model, horizon), view) in enumerate(groups):
        labels.append(escape(f"{model_label(model)} · {int(horizon)}日"))
        view = view.sort_values("date")
        for field, label, color in [("pearson_ic", "Pearson IC", "#E4B863"), ("rank_ic", "Rank IC", "#7C9CC4")]:
            if field in view:
                figure.add_scatter(x=view.date, y=view[field], mode="lines", name=label,
                                   line={"color": color, "width": 1.5}, visible=(not selectable or index == 0))
                owners.append(index)
    figure.update_layout(height=350, legend={"orientation": "h"})
    figure.update_yaxes(title="截面相关系数")
    if selectable and labels:
        figure.update_layout(title=labels[0], updatemenus=[{"buttons": [
            {"label": label, "method": "update",
             "args": [{"visible": [owner == index for owner in owners]}, {"title": label}]}
            for index, label in enumerate(labels)
        ], "direction": "down", "x": 0, "y": 1.2}])
    return figure
