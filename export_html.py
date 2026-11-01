"""导出当前成功实验的单文件HTML；不训练、不读取未登记的历史结果。"""

import argparse
from html import escape
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd

from price_source import RESULT_DIR, PRICE_FILE, FEATURE_FILE, PRICE_COLUMNS, validate_prices
from report_data import (
    ReportError, load_report, model_label, strategy_label, model_capabilities,
    existing_columns, importance_view, MODEL_LABELS, STRATEGY_LABELS,
    COLUMN_LABELS, PERCENT_COLUMNS, IC_NOTE, COST_NOTE, STRATEGY_NOTES,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "股票收益预测_当前实验报告.html"

CSS = r"""
:root{--night:#0B1322;--deep:#060A13;--gold:#E4B863;--gold-light:#F3DCA8;--moon:#7C9CC4;--text:rgba(255,255,255,.94);--soft:rgba(255,255,255,.76);--muted:rgba(255,255,255,.54);--line:rgba(255,255,255,.15);--glass:rgba(255,255,255,.08);--shadow:0 16px 40px rgba(3,7,18,.5),inset 0 1px 0 rgba(255,255,255,.22),inset 0 -1px 0 rgba(2,6,16,.35)}
*{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;color:var(--text);font-family:Aptos,"Microsoft YaHei UI",sans-serif;background:radial-gradient(circle at 88% 4%,rgba(124,156,196,.28),transparent 55%),radial-gradient(circle at 7% 88%,rgba(51,81,122,.38),transparent 60%),radial-gradient(circle at 54% 42%,rgba(228,184,99,.11),transparent 34%),linear-gradient(145deg,var(--deep),var(--night) 48%,#0A101C);background-attachment:fixed;min-height:100vh}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.025;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}
.shell{width:100%;padding:28px clamp(18px,3vw,56px) 60px} .top{display:flex;align-items:center;justify-content:space-between;gap:24px;margin-bottom:26px} h1,h2,h3{font-family:"Iowan Old Style","Noto Serif SC","Songti SC",serif} h1{font-size:34px;margin:0} .sample{color:var(--gold);font-size:12px;margin-top:5px}
.tabs{display:flex;gap:6px;padding:6px;background:rgba(255,255,255,.055);backdrop-filter:blur(60px) saturate(180%);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow)} .tab{appearance:none;border:1px solid transparent;border-radius:16px;background:transparent;color:var(--muted);padding:12px 20px;cursor:pointer;transition:.5s cubic-bezier(.16,1,.3,1)} .tab:hover{background:rgba(255,255,255,.08);color:var(--text)} .tab.active{color:var(--gold-light);background:rgba(255,255,255,.1);border-color:rgba(228,184,99,.4)}
.panel{display:none} .panel.active{display:block} .kicker{color:var(--gold);font:650 12px Consolas,monospace} .section-title{font-size:26px;margin:3px 0 4px} .desc{color:var(--muted);font-size:13px;margin-bottom:18px} .grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0} .metric,.glass{background:var(--glass);background-image:linear-gradient(to bottom,rgba(255,255,255,.12),transparent 50%);backdrop-filter:blur(60px) saturate(180%);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow)} .metric{padding:17px 18px} .metric span{display:block;color:var(--muted);font-size:12px} .metric strong{display:block;color:var(--gold-light);font-size:25px;margin-top:7px} .glass{padding:18px;margin:18px 0;overflow:hidden} .chart-title{font-size:19px;margin:4px 0 12px}
.data-table{border-collapse:collapse;width:100%;font-size:13px;color:var(--soft)} .data-table th,.data-table td{padding:11px 12px;border-bottom:1px solid rgba(255,255,255,.09);text-align:left} .data-table th{color:var(--gold-light);font-weight:650} .note{color:var(--muted);font-size:12px;line-height:1.7;margin-top:12px} .footer{color:var(--muted);font-size:12px;border-top:1px solid rgba(255,255,255,.1);padding-top:18px;margin-top:34px}
@media(max-width:850px){.top{align-items:flex-start;flex-direction:column}.tabs{width:100%;overflow:auto}.tab{padding:10px 12px;white-space:nowrap}.grid4{grid-template-columns:repeat(2,1fr)}} @media(max-width:520px){.grid4{grid-template-columns:1fr}h1{font-size:28px}}

.controls{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}select,input,button{font:inherit}select,input{background:#111A28;color:#F2F5F9;border:1px solid #465269;border-radius:10px;padding:8px}button.page{background:#202C40;color:#F3DCA8;border:1px solid #465269;border-radius:9px;padding:6px 13px;cursor:pointer}.table-scroll{overflow:auto;max-height:550px}.empty{color:#9EABBC;padding:18px 0}.data-table td{white-space:nowrap}.data-table th{white-space:nowrap;position:sticky;top:0;background:#111A28}.data-table td.params{white-space:normal;min-width:220px}.pager{display:flex;align-items:center;gap:12px;margin:12px 0}.badge{color:#F3DCA8}.panel{min-height:200px}
"""

JAVASCRIPT = r"""
document.querySelectorAll('.tab').forEach(function(button){
  button.addEventListener('click',function(){
    document.querySelectorAll('.tab,.panel').forEach(function(item){item.classList.remove('active');});
    button.classList.add('active');
    document.getElementById(button.dataset.target).classList.add('active');
    if(window.Plotly){setTimeout(function(){
      document.querySelectorAll('.panel.active .js-plotly-plot').forEach(function(plot){Plotly.Plots.resize(plot);});
    },30);}
  });
});
document.querySelectorAll('.table-widget').forEach(function(widget){
  const data=JSON.parse(document.getElementById(widget.dataset.source).textContent);
  const controls=widget.querySelector('.controls');
  const filters=[];
  const title=function(key,value){
    if(key==='model'){return data.modelLabels[value]||value;}
    if(key==='strategy'){return data.strategyLabels[value]||value;}
    if(key==='horizon'){return value+'日';}
    if(key==='cost_bps'){return value+' bp';}
    return String(value);
  };
  data.filters.forEach(function(key){
    const label=document.createElement('label');
    label.textContent=(data.labels[key]||key)+' ';
    const select=document.createElement('select');
    const all=document.createElement('option');
    all.value='';all.textContent='全部';select.appendChild(all);
    const values=Array.from(new Set(data.rows.map(function(row){return row[key];}).filter(function(x){return x!==null;})));
    values.sort(function(a,b){return typeof a==='number'&&typeof b==='number'?a-b:String(a).localeCompare(String(b));});
    values.forEach(function(value){
      const option=document.createElement('option');option.value=String(value);option.textContent=title(key,value);select.appendChild(option);
    });
    select.dataset.filterKey=key;
    select.setAttribute('aria-label',data.labels[key]||key);
    label.appendChild(select);controls.appendChild(label);filters.push([key,select]);
  });
  const search=document.createElement('input');search.placeholder='搜索股票、模型或因子';
  controls.appendChild(search);
  const head=widget.querySelector('thead');
  const header=document.createElement('tr');
  data.columns.forEach(function(key){const th=document.createElement('th');th.textContent=data.labels[key]||key;header.appendChild(th);});
  head.appendChild(header);
  const prev=widget.querySelector('.prev'),next=widget.querySelector('.next'),status=widget.querySelector('.page-status');
  const body=widget.querySelector('tbody');
  let page=0;
  function show(){
    const query=search.value.trim().toLowerCase();
    const rows=data.rows.filter(function(row){
      return filters.every(function(pair){return pair[1].value===''||String(row[pair[0]])===pair[1].value;})&&
        (!query||data.columns.some(function(key){return String(row[key]===null?'':row[key]).toLowerCase().includes(query);}));
    });
    const pages=Math.max(1,Math.ceil(rows.length/100));page=Math.max(0,Math.min(page,pages-1));
    body.replaceChildren();
    rows.slice(page*100,(page+1)*100).forEach(function(row){
      const tr=document.createElement('tr');
      data.columns.forEach(function(key){
        const td=document.createElement('td');let value=row[key];
        if(value===null||value===undefined){value='—';}
        else if(['model','strategy','horizon','cost_bps'].includes(key)){value=title(key,value);}
        else if(data.percentColumns.includes(key)&&typeof value==='number'){value=(value>=0?'+':'')+(value*100).toFixed(2)+'%';}
        else if(['date','test_start','test_end'].includes(key)){value=String(value).slice(0,10);}
        else if(typeof value==='number'){value=Number.isInteger(value)?value.toLocaleString():value.toFixed(4);}
        td.textContent=String(value);if(key==='best_params'){td.classList.add('params');}tr.appendChild(td);
      });body.appendChild(tr);
    });
    status.textContent='共 '+rows.length+' 条 · 第 '+(page+1)+' / '+pages+' 页';
    prev.disabled=page===0;next.disabled=page===pages-1;
    document.dispatchEvent(new CustomEvent('report-table-filter', {detail:{
      source:widget.dataset.source,
      values:Object.fromEntries(filters.map(function(pair){return [pair[0],pair[1].value];}))
    }}));
  }
  filters.forEach(function(pair){pair[1].addEventListener('change',function(){page=0;show();});});
  search.addEventListener('input',function(){page=0;show();});
  prev.addEventListener('click',function(){page--;show();});
  next.addEventListener('click',function(){page++;show();});
  show();
});
let syncingBacktest=false;
document.addEventListener('report-table-filter',function(event){
  const source=event.detail.source;
  if(syncingBacktest||!['data-backtest_summary','data-backtest_nav'].includes(source)){return;}
  const values=event.detail.values;
  const target=source==='data-backtest_summary'?'data-backtest_nav':'data-backtest_summary';
  syncingBacktest=true;
  document.querySelectorAll('[data-source="'+target+'"] select[data-filter-key]').forEach(function(select){
    const value=values[select.dataset.filterKey]||'';
    if(select.value!==value){select.value=value;select.dispatchEvent(new Event('change'));}
  });
  syncingBacktest=false;
  const plot=document.getElementById('chart-backtest');
  if(!plot||!plot.data||!window.Plotly){return;}
  const selected=plot.data.findIndex(function(trace){
    return trace.meta&&Object.entries(values).every(function(pair){
      return pair[1]===''||String(trace.meta[pair[0]])===pair[1];
    });
  });
  const title=selected<0?'当前筛选没有可用净值':plot.data[selected].meta.label;
  Plotly.update(plot,{visible:plot.data.map(function(_,index){return index===selected;})},{title:{text:title}});
});
"""


def script_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def table_widget(key, title, frame, columns, filters=("model", "horizon")):
    if frame.empty:
        return f'<div class="glass"><h3>{escape(title)}</h3><p class="empty">本次实验没有生成此项结果。</p></div>'
    columns = existing_columns(frame, columns)
    view = frame[columns].copy()
    payload = {
        "rows": json.loads(view.to_json(orient="records", date_format="iso", force_ascii=False)),
        "columns": columns, "filters": [name for name in filters if name in columns],
        "labels": COLUMN_LABELS, "modelLabels": MODEL_LABELS, "strategyLabels": STRATEGY_LABELS,
        "percentColumns": sorted(PERCENT_COLUMNS),
    }
    return (
        f'<div class="glass table-widget" data-source="data-{key}"><h3>{escape(title)}</h3>'
        '<div class="controls"></div><div class="table-scroll"><table class="data-table"><thead></thead><tbody></tbody></table></div>'
        '<div class="pager"><button class="page prev">上一页</button><span class="page-status"></span>'
        f'<button class="page next">下一页</button></div></div><script type="application/json" id="data-{key}">'
        + script_json(payload) + '</script>'
    )


def metric(label, value):
    return f'<div class="metric"><span>{escape(str(label))}</span><strong>{escape(str(value))}</strong></div>'


def build_html(report, prices=None, *, include_charts=True):
    charts_loaded = False
    def chart(figure, key=None):
        nonlocal charts_loaded
        figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.025)",
                             font={"color": "#CED6E2", "family": "Microsoft YaHei UI, sans-serif"},
                             margin={"l": 55, "r": 25, "t": 110, "b": 50})
        figure.update_layout(title={"x": .04, "y": .99, "yanchor": "top"})
        for menu in figure.layout.updatemenus:
            menu.update(y=1.12, yanchor="top")
        figure.update_xaxes(gridcolor="rgba(255,255,255,.09)")
        figure.update_yaxes(gridcolor="rgba(255,255,255,.09)")
        result = figure.to_html(full_html=False, include_plotlyjs=not charts_loaded,
                               div_id="chart-" + key if key else None,
                               config={"displaylogo": False, "responsive": True, "scrollZoom": False})
        charts_loaded = True
        return '<div class="glass">' + result + '</div>'

    if include_charts:
        from report_charts import price_chart, model_figure, backtest_figure, ic_figure
    latest = report.table("latest_predictions.csv")
    metrics = report.table("model_metrics.csv").sort_values(["horizon", "mae"])
    summary = report.table("backtest_summary.csv")
    nav = report.table("backtest_nav.csv")
    importance = report.table("feature_importance.csv")
    preprocessing = report.table("preprocessing_summary.csv")
    sections = []
    example = ""
    if include_charts and prices is not None and not latest.empty:
        prediction = latest.sort_values(["horizon", "model", "code"]).iloc[0]
        history = prices[prices.code == prediction.code]
        title = f"预测图示例：{model_label(prediction.model)} · {int(prediction.horizon)}日 · {prediction.code} {prediction['name']}"
        example = '<p class="desc">' + escape(title) + '</p>' + chart(price_chart(history, prediction, int(prediction.horizon)))
        example += '<p class="note">图示例使用首个可用组合；全部模型与周期的预测见下表。预测终点日期为工作日近似。</p>'
    sections.append((
        "prediction", "最新收益预测",
        '<p class="desc">仅展示本实验实际生成的最新预测；序列模型没有生产预测时不会伪造记录。</p>' + example +
        table_widget("latest", "全部可用预测", latest, [
            "model", "horizon", "code", "name", "date", "close", "predicted_return", "reference_target_price"
        ])
    ))
    sections.append((
        "models", "模型测试评价",
        '<p class="desc">R²、相对零收益基线R²、误差、方向与股票覆盖数分别展示。参数由验证集选择。</p>' +
        (chart(model_figure(metrics)) if include_charts else "") +
        table_widget("metrics", "全部已完成模型与周期", metrics, [
            "model", "horizon", "best_params", "validation_mae", "mae", "rmse", "r2", "r2_oos_zero",
            "direction_accuracy", "prediction_correlation", "n_train", "n_validation", "n_test",
            "n_train_stocks", "n_validation_stocks", "n_test_stocks", "test_start", "test_end",
        ]) +
        table_widget("capabilities", "各模型输出覆盖情况", model_capabilities(report), [
            "model", "horizon", "latest_prediction", "feature_importance"
        ])
    ))
    daily = report.table("ic_daily.csv")
    sections.append((
        "ic", "预测IC与逐股误差",
        '<p class="desc">' + escape(IC_NOTE) + '</p>' +
        (chart(ic_figure(daily, selectable=True)) if include_charts and not daily.empty else "") +
        table_widget("ic_summary", "预测相关性汇总", report.table("ic_summary.csv"), [
            "model", "horizon", "metric", "mean", "std", "icir", "t_stat", "positive_ratio", "n_days"
        ]) +
        table_widget("per_stock", "逐股测试误差", report.table("per_stock_metrics.csv"), [
            "model", "horizon", "code", "n_test", "mae", "correlation"
        ]) +
        '<p class="note">当前训练流程没有生成单因子IC，报告不读取历史因子IC文件作为替代。</p>'
    ))
    strategy_notes = " ".join(STRATEGY_NOTES.values())
    backtest_chart = ""
    if include_charts and not nav.empty:
        figure = backtest_figure(nav, selectable=True)
        figure.layout.updatemenus = ()
        backtest_chart = chart(figure, "backtest")
    sections.append((
        "backtest", "样本外组合评价",
        '<p class="desc">' + escape(strategy_notes + " " + COST_NOTE) + '</p>' +
        '<p class="note">下方筛选会同步更新净值图、摘要和逐期明细；筛选匹配多组时，图表显示首个匹配组合。</p>' +
        backtest_chart +
        table_widget("backtest_summary", "按模型、周期、策略与费用查看", summary, [
            "model", "horizon", "strategy", "cost_bps", "periods", "cumulative_return",
            "annualized_sharpe", "max_drawdown", "average_turnover"
        ], ("model", "horizon", "strategy", "cost_bps")) +
        table_widget("backtest_nav", "逐期收益、换手与费用", nav, [
            "model", "horizon", "strategy", "cost_bps", "date", "gross_return", "turnover", "cost", "net_return", "nav"
        ], ("model", "horizon", "strategy", "cost_bps")) +
        '<p class="note">最大回撤包含初始净值1并沿用负数表示。前5名相对等权曲线是相对收益复利值，未虚构单独的基准净值。</p>'
    ))
    top_frames = []
    if not importance.empty:
        for item in importance[["model", "horizon"]].drop_duplicates().itertuples(index=False):
            selected = importance_view(report, item.model, item.horizon, 20)
            if not selected.empty:
                top_frames.append(selected)
    top = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    sections.append((
        "factors", "因子与预处理",
        '<p class="desc">重要性只列各模型/周期真实提供的前20项；未提供内置重要性的模型见输出覆盖情况。PCA树模型的平方载荷分摊是解释代理。</p>' +
        table_widget("preprocessing", "各阶段的因子数量与输入维数", preprocessing, [
            "model", "horizon", "stage", "selection_config", "selected_factor_count", "model_input_dimensions", "pca_explained_variance"
        ]) +
        table_widget("importance", "模型权重与重要性", top, [
            "model", "horizon", "factor", "importance", "importance_method", "stage"
        ]) +
        table_widget("ablation", "因子组消融", report.table("factor_group_ablation.csv"), [
            "horizon", "group", "n_factors", "validation_mae", "test_mae", "r2_oos_zero", "direction_accuracy"
        ], ("horizon",))
    ))
    tabs = "".join(f'<button class="tab{" active" if i == 0 else ""}" data-target="{key}">{escape(title)}</button>'
                   for i, (key, title, _) in enumerate(sections))
    panels = "".join(f'<section id="{key}" class="panel{" active" if i == 0 else ""}">'
                     f'<h2 class="section-title">{escape(title)}</h2>{content}</section>'
                     for i, (key, title, content) in enumerate(sections))
    metadata = report.metadata
    subtitle = f"实验 {report.experiment_id[:12]} · 行情截止 {metadata.get('latest_market_date', '—')} · 生成于 {metadata.get('generated_at', '—')}"
    cards = (metric("实验股票池", report.stock_count) + metric("候选因子", report.feature_count) +
             metric("完成模型数", len(report.models)) + metric("完成周期", " / ".join(str(h) for h in metadata["horizons"])))
    notes = "".join('<p class="note">' + escape(note) + '</p>' for note in report.warnings)
    if not include_charts:
        notes += '<p class="note">本文件使用纯表格模式，完整数据可筛选、搜索和分页；未包含图表。</p>'
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>股票收益预测 · 当前实验报告</title><style>' + CSS + '</style></head><body><main class="shell">'
        '<header class="top"><div><h1>股票收益预测 · 当前实验</h1><div class="sample">' + escape(subtitle) +
        '</div></div><nav class="tabs">' + tabs + '</nav></header><div class="grid4">' + cards + '</div>' + notes +
        '<details class="note"><summary>完整实验编号</summary>' + escape(report.experiment_id) + '</details>' + panels +
        '<div class="footer">报告固定于导出时点，所有表格来自同一成功实验；仅用于研究与课程展示。</div></main>'
        '<script>' + JAVASCRIPT + '</script></body></html>'
    )


def write_html(destination, content):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + destination.stem + "_", suffix=".html", dir=destination.parent)
    temporary = Path(temporary_name).resolve()
    if temporary.parent != destination.parent:
        os.close(descriptor)
        raise ValueError("临时报告路径不正确")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="导出已成功发布的当前实验HTML报告")
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR, help="成功实验的结果目录")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="输出HTML文件，默认保留旧展示样板")
    parser.add_argument("--price-file", type=Path, default=PRICE_FILE, help="实验对应的行情快照")
    parser.add_argument("--feature-file", type=Path, default=FEATURE_FILE, help="实验对应的因子快照")
    parser.add_argument("--tables-only", action="store_true", help="仅导出可筛选数据表，不加载绘图库或历史行情")
    args = parser.parse_args(argv)
    try:
        report = load_report(args.result_dir, args.price_file, args.feature_file)
        prices = None
        if not args.tables_only and not report.table("latest_predictions.csv").empty:
            prices = validate_prices(pd.read_parquet(args.price_file, columns=PRICE_COLUMNS))
        content = build_html(report, prices, include_charts=not args.tables_only)
        write_html(args.output, content)
    except (ReportError, OSError, ValueError, ImportError) as error:
        print(f"未导出报告：{error}", file=sys.stderr)
        return 2
    print(f"已导出实验 {report.experiment_id[:12]}：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
