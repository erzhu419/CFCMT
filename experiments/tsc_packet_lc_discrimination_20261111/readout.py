"""Fixed-law direction probabilities, linear scores and weighted ranking."""

from collections import Counter, defaultdict
from math import exp, fsum

import numpy as np
from scipy.special import logsumexp

from experiments.tsc_packet_fitted_cache_stability_20261111.readout import extended
from experiments.tsc_packet_lane_change_holdout_20261111.readout import _focal_branch_factors
from experiments.tsc_packet_mixture_confirmation_20261111.readout import MixtureParticleLaw


DIRECTIONS = {0: 'left', 2: 'right'}
TOL = 1e-10


def weighted_auc(positive_scores, positive_weights, stay_scores, stay_weights):
    """Weighted pair concordance, including half credit for tied log scores."""
    total, denominator = [], fsum(positive_weights)*fsum(stay_weights)
    if not positive_scores or not stay_scores or denominator <= 0:
        return None
    for a, wa in zip(positive_scores, positive_weights, strict=True):
        for b, wb in zip(stay_scores, stay_weights, strict=True):
            if a is None or b is None:
                return None
            tie = a == b or abs(a-b) <= TOL
            total.append(wa*wb*(.5 if tie else float(a > b)))
    return fsum(total)/denominator


def _latent_rows(law, theta):
    logw = law._evaluate(theta)[0]
    if isinstance(law, MixtureParticleLaw):
        survived = law.survived
        groups = [law.data['focal_groups'][i] for i in law.focal_groups[survived]]
        feature_ids = np.array([g['feature_id'] for g in groups], dtype=int)
        allowed = np.zeros((len(groups), 3), dtype=bool)
        for i, group in enumerate(groups):
            allowed[i, group['allowed']] = True
    else:
        survived = np.array([o['survived'] for o in law._leaf_outcomes()], dtype=bool)
        factors, _, all_allowed = _focal_branch_factors(law, theta)
        if not np.array_equal(factors >= 0, survived):
            raise ArithmeticError('Survival and cached focal categorical decisions disagree')
        feature_ids = np.array([law.data['factors'][i]['feature_id'] for i in factors[survived]])
        allowed = all_allowed[survived]
    return logw[survived], feature_ids, allowed, exp(float(logsumexp(logw[survived])-logsumexp(logw)))


def direction_readout(law, theta, direction):
    """Decompose permission and class probability without changing the law.

    Score terms are theta_d * E[phi | survival (and permission)], evaluated
    under this frozen theta. They are not marginal log-odds contributions.
    """
    logw, ids, masks, survival = _latent_rows(law, theta)
    denominator = float(logsumexp(logw))
    if not np.isfinite(denominator):
        raise ArithmeticError('The fixed direction readout has no survivor support')
    weights = np.exp(logw-denominator)
    phi, coefficients = law.features[ids], theta[3 if direction == 0 else 4]
    permitted = masks[:, direction]
    log_allowed = float(logsumexp(logw[permitted])-denominator)
    allowed_mass = exp(log_allowed)
    marginal = law.marginal(theta, 'lane_change', direction)
    p, logp = marginal['likelihood'], marginal['log_likelihood']
    conditional_log = float(logp-log_allowed) if np.any(permitted) else None
    conditional = exp(conditional_log) if conditional_log is not None else None
    all_terms = (weights @ phi)*coefficients
    permitted_terms = (np.exp(logw[permitted]-logsumexp(logw[permitted])) @ phi[permitted])*coefficients if np.any(permitted) else None
    if p > allowed_mass+TOL or (conditional is not None and abs(p-allowed_mass*conditional) > TOL):
        raise ArithmeticError('Direction probability did not factor into permission and conditional probability')
    branches = defaultdict(list)
    for i, (feature, mask) in enumerate(zip(ids, masks, strict=True)):
        branches[(int(feature), tuple(int(x) for x in np.flatnonzero(mask)))].append(i)
    branch_summaries = []
    for (feature, choices), indices in sorted(branches.items()):
        terms = law.features[feature]*coefficients
        branch_summaries.append(dict(feature_id=feature, allowed=list(choices),
            survivor_mass=float(weights[indices].sum()), category_score=float(terms.sum()),
            score_terms=terms.tolist()))
    return dict(survivor_probability=survival, direction_probability=p,
        direction_log_probability=extended(logp), direction_nll=extended(marginal['nll']),
        allowed_mass=allowed_mass, allowed_log_mass=extended(log_allowed),
        conditional_on_allowed_probability=conditional,
        conditional_log_probability=extended(conditional_log),
        mean_category_score=float(all_terms.sum()), survivor_score_terms=all_terms.tolist(),
        mean_allowed_category_score=float(permitted_terms.sum()) if permitted_terms is not None else None,
        latent_score_terms=permitted_terms.tolist() if permitted_terms is not None else None,
        branch_summaries=branch_summaries)


def score(row, metric):
    if metric in ('mean_category_score', 'mean_allowed_category_score'):
        return row[metric]
    value = row[metric]
    return value['value'] if value['state'] == 'finite' else (-float('inf') if value['state'] == 'negative_infinity' else None)


METRICS = ('direction_log_probability', 'allowed_log_mass', 'conditional_log_probability',
           'mean_category_score', 'mean_allowed_category_score')


def rank(a, b):
    if a is None or b is None:
        return 'undefined'
    if a == b or abs(a-b) <= TOL:
        return 'ties'
    return 'wins' if a > b else 'losses'


def rank_summary(ranks):
    count = Counter(ranks)
    return dict(**{k:count[k] for k in ('wins','ties','losses','undefined')},
        score=(count['wins']+.5*count['ties'])/len(ranks) if ranks and not count['undefined'] else None)


def describe(rows, weights):
    total = fsum(weights)
    result = dict(count=len(rows), weight=total,
        zero_permission=sum(row['allowed_mass'] == 0 for row in rows),
        zero_direction_probability=sum(row['direction_probability'] == 0 for row in rows))
    for metric in ('direction_probability','allowed_mass','conditional_on_allowed_probability','mean_category_score','mean_allowed_category_score'):
        values = [row[metric] for row in rows]
        defined=[(v,w) for v,w in zip(values,weights,strict=True) if v is not None]
        defined_weight=fsum(w for v,w in defined)
        result[metric] = dict(missing=sum(v is None for v in values),
            defined_weight=defined_weight,
            defined_weighted_mean=fsum(v*w for v,w in defined)/defined_weight if defined_weight else None,
            weighted_mean=fsum(v*w for v,w in zip(values,weights,strict=True))/total
                if values and total and all(v is not None for v in values) else None)
    for metric in ('survivor_score_terms','latent_score_terms'):
        values = [row[metric] for row in rows]
        result[metric] = dict(missing=sum(v is None for v in values),
            weighted_mean=(np.average(np.asarray(values),axis=0,weights=weights).tolist()
                if values and total and all(v is not None for v in values) else None))
    return result


def training_summary(records, point):
    out = {}
    for direction,name in DIRECTIONS.items():
        positives = [r for r in records if r['y'] == direction]
        stays = [r for r in records if r['y'] == 1]
        groups = {'all':(positives,stays)}
        for seed in sorted({r['reference']['seed'] for r in records}):
            groups[str(seed)] = ([r for r in positives if r['reference']['seed']==seed],
                                 [r for r in stays if r['reference']['seed']==seed])
        out[name] = {}
        for group,(pos,neg) in groups.items():
            p = [r['points'][point][str(direction)] for r in pos]
            n = [r['points'][point][str(direction)] for r in neg]
            pw,nw = [r['objective_weight'] for r in pos],[r['objective_weight'] for r in neg]
            subsets={}
            for key in METRICS:
                valid_p=[(score(r,key),w) for r,w in zip(p,pw,strict=True) if score(r,key) is not None]
                valid_n=[(score(r,key),w) for r,w in zip(n,nw,strict=True) if score(r,key) is not None]
                subsets[key]=dict(positive_count=len(valid_p),stay_count=len(valid_n),
                    excluded_positive_count=len(p)-len(valid_p),excluded_stay_count=len(n)-len(valid_n),
                    excluded_positive_weight=fsum(pw)-fsum(w for v,w in valid_p),
                    excluded_stay_weight=fsum(nw)-fsum(w for v,w in valid_n),
                    positive_weight=fsum(w for v,w in valid_p),stay_weight=fsum(w for v,w in valid_n),
                    auc=weighted_auc([v for v,w in valid_p],[w for v,w in valid_p],
                                     [v for v,w in valid_n],[w for v,w in valid_n]))
            out[name][group] = dict(positive=describe(p,pw),stay=describe(n,nw),
                weighted_auc={key:weighted_auc([score(r,key) for r in p],pw,[score(r,key) for r in n],nw) for key in METRICS},
                explicitly_defined_subset_auc=subsets)
    return out


def regression_summary(records, point, coverage):
    pairs = defaultdict(dict)
    for row in records:
        pairs[row['pair_index']][row['role']] = row
    out = {}
    for direction,name in DIRECTIONS.items():
        selected = [pair for pair in pairs.values() if pair['positive']['direction']==direction]
        metrics = {}
        for key in METRICS:
            groups = defaultdict(list)
            for pair in selected:
                a,b = pair['positive'],pair['stay']
                ranking = rank(score(a['points'][point][str(direction)],key),score(b['points'][point][str(direction)],key))
                groups['all'].append(ranking)
                groups[f"time_block_{int(a['reference']['time_sec'])//300}"].append(ranking)
                c = coverage[('regression',a['case_index'],direction)]
                groups['positive_covered' if c['matching_train_stay_count'] else 'positive_uncovered'].append(ranking)
            metrics[key] = {k:rank_summary(v) for k,v in groups.items()}
            defined=[r for r in groups['all'] if r!='undefined']
            metrics[key]['explicitly_defined_pairs']=dict(excluded_pairs=len(groups['all'])-len(defined),
                                                         **rank_summary(defined))
        pos = [pair['positive']['points'][point][str(direction)] for pair in selected]
        neg = [pair['stay']['points'][point][str(direction)] for pair in selected]
        terms = {}
        for key in ('survivor_score_terms','latent_score_terms'):
            for subset in ('all','wins','losses','ties'):
                values=[]
                selected_pairs=0
                for a,b in zip(pos,neg,strict=True):
                    outcome=rank(score(a,'direction_log_probability'),score(b,'direction_log_probability'))
                    if subset=='all' or outcome==subset:
                        selected_pairs+=1
                        if a[key] is not None and b[key] is not None:
                            values.append(np.asarray(a[key])-np.asarray(b[key]))
                terms[key+'_'+subset] = dict(pairs=len(values),selected_pairs=selected_pairs,
                    missing_pairs=selected_pairs-len(values),mean_terms=np.mean(values,axis=0).tolist() if values else None)
        out[name] = dict(pairs=len(selected),positive=describe(pos,[1.]*len(pos)),stay=describe(neg,[1.]*len(neg)),ranking=metrics,score_margin_terms=terms)
    return out
