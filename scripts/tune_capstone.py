"""Bounded client-grouped model search; historical test split is scored only after selection."""
import hashlib,json,platform
import importlib.metadata
import joblib
from sklearn.inspection import permutation_importance
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from ml_utils import MODEL_NUMERIC_FEATURES,MODEL_CATEGORICAL_FEATURES
ROOT=Path(__file__).resolve().parents[1]

def measure(y,p):
    order=np.argsort(-np.asarray(p),kind='stable')
    return {'precision_at_20':float(np.asarray(y)[order[:20]].mean()),'precision_at_50':float(np.asarray(y)[order[:50]].mean()),'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'base_rate':float(np.mean(y)),'rows':len(y)}

def features(d,expanded=False):
    x=d.copy()
    for c in ['impressions_90d','clicks_90d','sessions_90d','ai_sessions_90d']:
        x['log_'+c]=np.log1p(x[c])
    nums=list(MODEL_NUMERIC_FEATURES)
    if expanded:
        for a,b,name in [('impressions_90d','days_with_impressions','impressions_per_active_day'),('sessions_90d','days_with_sessions','sessions_per_active_day'),('days_since_last_update','content_age_days','relative_staleness'),('char_count','word_count','chars_per_word')]:
            x[name]=x[a]/x[b].replace(0,np.nan);nums.append(name)
        x['position_missing']=x.avg_position.eq(0).astype(int);nums.append('position_missing')
        x.loc[x.avg_position.eq(0),'avg_position']=np.nan
    x=x[nums+MODEL_CATEGORICAL_FEATURES].copy()
    x[nums]=x[nums].replace([np.inf,-np.inf],np.nan)
    for c in MODEL_CATEGORICAL_FEATURES:x[c]=x[c].fillna('unknown').astype(str)
    assert not any(c in x for c in ['client_id','content_id','trend_pct','trend_direction','impressions_last_30d','impressions_prev_30d'])
    return x,nums

def pipeline(model,nums):
    return make_pipeline(ColumnTransformer([('num',SimpleImputer(strategy='median',add_indicator=True),nums),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),MODEL_CATEGORICAL_FEATURES)]),model)

def baseline(train,test):
    # Fit percentile reference distributions and CTR medians using training rows only.
    med=train.loc[train.avg_position.between(0,50,inclusive='right')].groupby('position_tier').ctr.median()
    def gap(d):return (d.position_tier.map(med).fillna(d.ctr)-d.ctr).clip(lower=0)
    def rank(ref,v):return np.searchsorted(np.sort(np.asarray(ref)),np.asarray(v),side='right')/len(ref)
    return .45*rank(train.impressions_90d,test.impressions_90d)+.35*rank(train.days_since_last_update,test.days_since_last_update)+.20*rank(gap(train),gap(test))*test.avg_position.gt(0).to_numpy()

def run():
    d=pd.read_csv(ROOT/'data/raw/content_refresh_anonymized.csv')
    d=d.loc[(d.impressions_90d>0)&(d.content_age_days>=90)].drop_duplicates('content_id').reset_index(drop=True)
    y=d.trend_direction.str.lower().eq('down').astype(int).to_numpy()
    clients=d.client_id.drop_duplicates().to_numpy(); hold=np.random.default_rng(42).permutation(clients)[:round(len(clients)*.2)]
    test=d.client_id.isin(hold).to_numpy(); dev=np.flatnonzero(~test);testidx=np.flatnonzero(test)
    assert set(d.loc[test,'client_id']).isdisjoint(set(d.loc[~test,'client_id']))
    candidates={
      'forest_reference':(False,RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_leaf=25,class_weight='balanced_subsample',n_jobs=4,random_state=42)),
      'forest_deeper':(True,RandomForestClassifier(n_estimators=300,max_depth=18,min_samples_leaf=8,max_features=.8,class_weight='balanced_subsample',n_jobs=4,random_state=42)),
      'extra_trees':(True,ExtraTreesClassifier(n_estimators=300,max_depth=None,min_samples_leaf=8,max_features=.8,class_weight='balanced',n_jobs=4,random_state=42)),
      'boosting_small':(False,HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=42)),
      'boosting_features':(True,HistGradientBoostingClassifier(max_iter=200,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=42)),
      'boosting_deeper':(True,HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=60,l2_regularization=20,early_stopping=False,random_state=42))}
    folds=list(GroupKFold(n_splits=4).split(dev,y[dev],d.loc[dev,'client_id']))
    results={};fitted={}
    for name,(expanded,model) in candidates.items():
        x,nums=features(d,expanded);p=pipeline(model,nums);scores=[]
        for fold,(a,b) in enumerate(folds):
            tr,va=dev[a],dev[b]
            assert set(d.loc[tr,'client_id']).isdisjoint(set(d.loc[va,'client_id']))
            fitted_fold=clone(p).fit(x.iloc[tr],y[tr]);m=measure(y[va],fitted_fold.predict_proba(x.iloc[va])[:,1]);m['fold']=fold+1;scores.append(m)
        results[name]={'folds':scores,'mean_precision_at_50':float(np.mean([s['precision_at_50'] for s in scores])),'mean_precision_at_20':float(np.mean([s['precision_at_20'] for s in scores]))}
        fitted[name]=(x,p)
        print(name,round(results[name]['mean_precision_at_50'],3),flush=True)
    # Selection uses only development cross-validation, never the historical holdout.
    winner=max(results,key=lambda k:results[k]['mean_precision_at_50'])
    test_results={'rule_baseline':measure(y[testidx],baseline(d.loc[~test],d.loc[test]))}
    for name in dict.fromkeys(['forest_reference',winner]):
        x,p=fitted[name];p.fit(x.iloc[dev],y[dev]);pred=p.predict_proba(x.iloc[testidx])[:,1];test_results[name]=measure(y[testidx],pred)
        if name==winner:
            (ROOT/'work/models').mkdir(exist_ok=True)
            joblib.dump(p,ROOT/'work/models/tuned_capstone.joblib')
            importance=permutation_importance(p,x.iloc[testidx],y[testidx],scoring='roc_auc',n_repeats=3,random_state=42,n_jobs=1)
            important=[{'feature':str(x.columns[i]),'mean_auc_drop':float(importance.importances_mean[i])} for i in np.argsort(-importance.importances_mean)[:5]]
            top=np.argsort(-pred,kind='stable')[:50];wrong=top[y[testidx][top]==0]
            errors={'false_positives_in_top_50':int(len(wrong)),'content_type_counts':d.iloc[testidx[wrong]].content_type.value_counts().to_dict(),'median_impressions_false_positive':float(d.iloc[testidx[wrong]].impressions_90d.median()) if len(wrong) else None}
    payload={'selection':'Highest unweighted mean P@50 across four GroupKFold splits on the 26 development clients. Six fixed candidates; ties use declared order.','selected_model':winner,'development_cv':results,'historical_holdout':test_results,'selected_error_summary':errors,'post_selection_permutation_importance':important,'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','scipy']},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Exploratory current-state decline detection. Seed-42 holdout was used in earlier work and is not sealed. 90-day inputs overlap label window. No future or causal claim.','dataset_sha256':hashlib.sha256((ROOT/'data/raw/content_refresh_anonymized.csv').read_bytes()).hexdigest(),'python':platform.python_version(),'model_parameters':{k:v[1].get_params() for k,v in candidates.items()}}
    out=ROOT/'work/outputs/tuning_results.json';out.write_text(json.dumps(payload,indent=2)+'\n')
    print('Selected:',winner);print(json.dumps(test_results,indent=2))
    lines=['# Model improvement experiment','',payload['scope'],'',payload['selection'],'','## Development validation','', '| Model | Mean P@50 | Mean P@20 |','|---|---:|---:|']
    lines += [f"| {k} | {v['mean_precision_at_50']:.3f} | {v['mean_precision_at_20']:.3f} |" for k,v in results.items()]
    lines += ['','Fold base rates: '+', '.join(f"{r['base_rate']:.3f}" for r in results['forest_reference']['folds'])+'. Individual fold metrics are in the results JSON.']
    lines += ['',f'Selected before holdout scoring: **{winner}**.','','## Historical holdout comparison','','| Method | P@50 | P@20 | ROC-AUC | Base rate |','|---|---:|---:|---:|---:|']
    lines += [f"| {k} | {v['precision_at_50']:.3f} | {v['precision_at_20']:.3f} | {v['roc_auc']:.3f} | {v['base_rate']:.3f} |" for k,v in test_results.items()]
    lines += ['','The reference forest here uses training-only median imputation, missingness flags and category encoding. It is a controlled comparator, not an exact replay of the previous zero-imputed 68% run. Stable ranking resolves ties by dataset order.','',f"Selected model has {errors['false_positives_in_top_50']} false positives among its top 50.",'','Feature additions: activity per active day, relative staleness, characters per word and missing-position flag. No trend labels, comparison-window ingredients or IDs enter the model. Preprocessors fit within each training fold. The rule baseline also fits its distributions and tier medians on training clients only.','','Do not replace the published capstone headline merely because this historical split improves. Inspect the development fold results and validate on genuinely new clients and non-overlapping future outcomes before deployment.','','Reproduce with the existing requirements-capstone.lock.txt environment and run work/notebooks/model_improvement.ipynb top to bottom. Aggregate evidence is in work/outputs/tuning_results.json.']
    lines += ['','## Model inspection','', 'Top features by post-selection permutation ROC-AUC drop: '+', '.join(f"{r['feature']} ({r['mean_auc_drop']:.4f})" for r in important)+'. This diagnostic is not used for selection and does not establish causality.','', 'Selected boosting_small uses the original feature list plus imputation missingness flags; the extra activity ratios did not improve validation performance.','', 'Trained pipeline saved locally to work/models/tuned_capstone.joblib (excluded from Git). Feed it the feature frame returned by features(data, expanded=False); it includes fitted imputation and encoding. Only load model binaries from trusted sources.']
    (ROOT/'work/model_improvement.md').write_text('\n'.join(lines)+'\n')
    return payload

if __name__=='__main__':run()
