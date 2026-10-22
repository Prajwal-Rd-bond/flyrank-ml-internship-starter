"""Render all capstone narratives from the notebook's aggregate JSON."""
import html
import json
import re
from pathlib import Path


def render(root):
    m = json.loads((root / 'work/outputs/capstone_model_metrics.json').read_text())
    rf, base = m['random_forest'], m['baseline']
    summary = (f"Random Forest Precision@50 is {rf['precision_at_50']:.0%}, versus "
               f"{base['precision_at_50']:.0%} for the baseline "
               f"({100*(rf['precision_at_50']-base['precision_at_50']):.0f} percentage points higher). "
               f"The seed-42 holdout contains {m['test_rows']:,} pages from {m['n_holdout_clients']} clients, "
               f"with a declining-label base rate of {m['test_base_rate']:.1%}.")
    md = ['# Content refresh prioritisation', '', 'Prajwal Ravi Devadiga · FlyRank ML Internship', '']
    body = ['<h1>Content refresh prioritisation</h1><p class="author">Prajwal Ravi Devadiga · FlyRank ML Internship</p>']

    def section(title, paragraphs):
        md.extend(['## ' + title, ''])
        body.append('<h2>' + html.escape(title) + '</h2>')
        for p in paragraphs:
            md.extend([p, ''])
            body.append('<p>' + html.escape(p) + '</p>')

    def table(headers, rows):
        md.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'])
        md.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
        md.append('')
        body.append('<div style="overflow-x:auto"><table><thead><tr>' + ''.join('<th scope="col">'+html.escape(h)+'</th>' for h in headers) + '</tr></thead><tbody>')
        for row in rows:
            body.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>')
        body.append('</tbody></table></div>')

    section('Measured result', [summary, 'This exploratory study ranks pages by observed impression decline. It does not predict future traffic or demonstrate that refreshing a page improves its performance.'])
    section('Question and data', [
        'Can content and search signals help an editor prioritise a hypothetical budget of 50 page reviews? Precision@50 measures how many selected pages carry the observed declining label.',
        f"The bundled FlyRank sample contains {sum(m['label_distribution'].values()):,} retained pages across {m['n_clients_total']} pseudonymous clients. Keep pages with positive 90-day impressions and content age of at least 90 days; deduplicate content IDs. The target is trend_direction == down.",
        'Direct trend variables, last/previous 30-day ingredients and IDs are excluded from features. IDs are used only for deduplication and grouped splitting. Published outputs are aggregate summaries; no action queue or additional dataset is committed.'
    ])
    section('Baseline and models', [
        'Baseline score = 0.45 × visibility percentile + 0.35 × staleness percentile + 0.20 × positive CTR-gap percentile. Expected CTR is the median for each position tier among pages with positions above zero and at most 50; unmapped tiers fall back to observed CTR. Nonpositive positions receive zero CTR-gap contribution.',
        'Baseline ranks, CTR medians and categorical vocabulary are computed from the full unlabelled snapshot. This transductive setup is retained to reproduce the original experiment. Model fitting uses training clients only.',
        'The Random Forest uses 200 trees, maximum depth 10, minimum leaf size 25, balanced_subsample weights and random_state=42. Comparators are scaled balanced Logistic Regression (max_iter=1000) and a balanced depth-5 Decision Tree (minimum leaf size 50). Inputs are the 18 numeric and 8 categorical features listed in scripts/ml_utils.py.'
    ])
    section('Client-grouped holdout', [
        f"Seed 42 selects {m['n_holdout_clients']} of {m['n_clients_total']} clients for testing, leaving {m['train_rows']:,} training pages. Assertions verify no client overlap and exclude forbidden features. Models share the same holdout.",
        'These models have been compared using this holdout, so it is development evidence rather than an untouched final test. The table reports observed point estimates, without confidence intervals.'
    ])
    table(['Method','ROC-AUC','Precision@20','Precision@50'], [[name.replace('_',' ').title(),*[f'{m[name][k]:.3f}' for k in ['roc_auc','precision_at_20','precision_at_50']]] for name in ['baseline','logistic_regression','decision_tree','random_forest']])
    section('Actually executed split sensitivity', ['For each split seed, refit the Random Forest with its model seed fixed at 42. Recompute metrics against that split’s baseline and base rate. These three chosen splits are a sensitivity check, not a confidence interval or evidence that every unseen client will benefit.'])
    table(['Split seed','Test pages','Base rate','RF P@50','Baseline P@50'], [[r['split_seed'],r['test_rows'],*[f'{r[k]:.3f}' for k in ['test_base_rate','rf_precision_at_50','baseline_precision_at_50']]] for r in m['seed_sensitivity']])
    section('Interpretation', [f"In the canonical holdout, {round(50*rf['precision_at_50'])} of the top 50 pages carry the declining label; {50-round(50*rf['precision_at_50'])} do not. This is a measured label count, not a forecast of time saved or traffic recovered.", 'The following impurity-based feature importances describe the fitted forest. Correlated features and high-cardinality inputs can affect these values; they do not establish causes or the direction of effects.'])
    table(['Feature','Importance'], [[r['feature'],f"{r['importance']:.4f}"] for r in m['feature_importance']])
    section('Practical use and limitations', [
        'The local demonstration queue includes training pages, and its action rules use the observed trend label. It is not validation evidence. Review the trend in the source analytics before deciding whether any content needs work.',
        'The 90-day aggregates overlap the window used to define decline. Client grouping does not remove this temporal overlap. A future-prediction claim requires non-overlapping outcome windows.',
        'Only six clients are held out in the canonical split; client mix can materially change results. Full-snapshot preprocessing also limits claims about completely unseen data.',
        'Next experiment: fit preprocessing only on training clients, evaluate on future non-overlapping outcomes and reserve a new untouched set after model selection. Earlier weekly notebooks and supplied starter outputs are historical experiments, not the canonical capstone results.'
    ])
    section('Reproduce and audit', [f"Executed with Python {m['provenance']['python']}. Exact package versions are recorded in requirements-capstone.lock.txt; input and code hashes are in work/outputs/capstone_model_metrics.json. The notebook generates this page and work/capstone_report.md from that same JSON. Numerical results can change with library versions."])
    command='python3.14 -m venv .venv\n.venv/bin/pip install -r requirements-capstone.lock.txt\n.venv/bin/python scripts/run_capstone.py'
    md.extend(['```bash',command,'```',''])
    body.append('<pre>'+html.escape(command)+'</pre>')
    section('Acknowledgments and data credit', ['Dataset and starter foundation courtesy of FlyRank. Original starter scripts and teaching materials remain credited to their contributors.'])
    md.extend(['[FlyRank](https://flyrank.ai) · [Executed notebook](notebooks/capstone.ipynb) · [Aggregate metrics](outputs/capstone_model_metrics.json)',''])
    body.append('<p><a href="https://flyrank.ai">FlyRank</a> · <a href="https://github.com/Prajwal-Rd-bond/flyrank-ml-internship-starter/blob/main/work/notebooks/capstone.ipynb">Executed notebook</a> · <a href="https://github.com/Prajwal-Rd-bond/flyrank-ml-internship-starter/blob/main/work/outputs/capstone_model_metrics.json">Aggregate metrics</a></p>')
    (root/'work/capstone_report.md').write_text('\n'.join(md))
    page=root/'index.html'
    head=page.read_text().split('<body>')[0]
    page.write_text(head+'<body>\n<div class="container">\n'+'\n'.join(body)+'\n</div>\n</body>\n</html>\n')
    readme=root/'README.md'
    status='**Evaluation status:** '+summary+' Regenerate the notebook, metrics and reports with `python scripts/run_capstone.py` using the pinned capstone environment. These are exploratory current-state results with overlapping windows and full-snapshot preprocessing; see the report for limitations.'
    readme.write_text(re.sub(r'^\*\*Evaluation status:\*\*.*$',lambda _:status,readme.read_text(),flags=re.M))


if __name__ == '__main__':
    render(Path(__file__).resolve().parents[1])
