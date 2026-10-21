"""Reliability and agreement statistics used in validity/reliability papers.

Everything here depends only on NumPy and SciPy.

References
----------
* Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
  assessing rater reliability. Psychological Bulletin, 86(2), 420-428.
* McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some
  intraclass correlation coefficients. Psychological Methods, 1(1), 30-46.
* Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting
  intraclass correlation coefficients for reliability research. Journal of
  Chiropractic Medicine, 15(2), 155-163.
* Bland, J. M., & Altman, D. G. (1999). Measuring agreement in method
  comparison studies. Statistical Methods in Medical Research, 8(2), 135-160.
* Hopkins, W. G. (2000). Measures of reliability in sports medicine and
  science. Sports Medicine, 30(1), 1-15.
* Lakens, D. (2013). Calculating and reporting effect sizes to facilitate
  cumulative science. Frontiers in Psychology, 4, 863.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as _st

__all__ = [
    "ICCResult",
    "icc",
    "interpret_icc",
    "BlandAltmanResult",
    "bland_altman",
    "typical_error",
    "cohens_d",
    "pearson",
    "paired_t",
    "mean_sd",
    "agreement",
    "reliability",
]


def _clean_pairs(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of the same length")
    keep = ~(np.isnan(a) | np.isnan(b))
    if keep.sum() < 3:
        raise ValueError("at least 3 complete pairs are required")
    return a[keep], b[keep]


# --------------------------------------------------------------------------- ICC
@dataclass
class ICCResult:
    model: str          # "1,1", "2,1", "3,1", "1,k", "2,k", "3,k"
    description: str
    value: float
    ci_low: float
    ci_high: float
    f: float
    df1: int
    df2: float
    p: float
    n: int
    k: int

    def to_dict(self) -> dict:
        return asdict(self)


_ICC_DESC = {
    "1,1": "One-way random effects, absolute agreement, single measurement",
    "2,1": "Two-way random effects, absolute agreement, single measurement",
    "3,1": "Two-way mixed effects, consistency, single measurement",
    "1,k": "One-way random effects, absolute agreement, mean of k measurements",
    "2,k": "Two-way random effects, absolute agreement, mean of k measurements",
    "3,k": "Two-way mixed effects, consistency, mean of k measurements",
}


def icc(data, model: str = "2,1", alpha: float = 0.05) -> ICCResult:
    """Intraclass correlation coefficient with (1 - alpha) confidence interval.

    Parameters
    ----------
    data : array (n_subjects, k_raters_or_trials)
        Rows with any NaN are dropped.
    model : {"1,1", "2,1", "3,1", "1,k", "2,k", "3,k"}
        Shrout & Fleiss notation. For test-retest or method comparison
        studies, "2,1" (absolute agreement) is the usual recommendation
        (Koo & Li, 2016).
    """
    if model not in _ICC_DESC:
        raise ValueError(f"model must be one of {list(_ICC_DESC)}")
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("data must be 2-D with at least 2 columns")
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 3:
        raise ValueError("at least 3 complete subjects are required")

    grand = x.mean()
    ss_total = ((x - grand) ** 2).sum()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    msw = (ss_total - ss_rows) / (n * (k - 1))

    q = 1 - alpha / 2
    fcrit = _st.f.ppf

    if model in ("1,1", "1,k"):
        f0 = msr / msw
        df1, df2 = n - 1, n * (k - 1)
        fl = f0 / fcrit(q, df1, df2)
        fu = f0 * fcrit(q, df2, df1)
        if model == "1,1":
            val = (msr - msw) / (msr + (k - 1) * msw)
            lo, hi = (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)
        else:
            val = (msr - msw) / msr
            lo, hi = 1 - 1 / fl, 1 - 1 / fu
    elif model in ("3,1", "3,k"):
        f0 = msr / mse
        df1, df2 = n - 1, (n - 1) * (k - 1)
        fl = f0 / fcrit(q, df1, df2)
        fu = f0 * fcrit(q, df2, df1)
        if model == "3,1":
            val = (msr - mse) / (msr + (k - 1) * mse)
            lo, hi = (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)
        else:
            val = (msr - mse) / msr
            lo, hi = 1 - 1 / fl, 1 - 1 / fu
    else:  # "2,1" / "2,k"
        f0 = msr / mse
        df1, df2 = n - 1, (n - 1) * (k - 1)
        icc21 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
        fj = msc / mse
        vn = (k - 1) * (n - 1) * (k * icc21 * fj + n * (1 + (k - 1) * icc21) - k * icc21) ** 2
        vd = (n - 1) * k ** 2 * icc21 ** 2 * fj ** 2 + (n * (1 + (k - 1) * icc21) - k * icc21) ** 2
        v = vn / vd
        f3u = fcrit(q, n - 1, v)
        f3l = fcrit(q, v, n - 1)
        lo = n * (msr - f3u * mse) / (f3u * (k * msc + (k * n - k - n) * mse) + n * msr)
        hi = n * (f3l * msr - mse) / (k * msc + (k * n - k - n) * mse + n * f3l * msr)
        if model == "2,1":
            val = icc21
        else:
            val = (msr - mse) / (msr + (msc - mse) / n)
            lo = lo * k / (1 + lo * (k - 1))
            hi = hi * k / (1 + hi * (k - 1))

    p = float(_st.f.sf(f0, df1, df2))
    return ICCResult(model, _ICC_DESC[model], float(val), float(lo), float(hi),
                     float(f0), int(df1), float(df2), p, int(n), int(k))


def interpret_icc(value: float, lang: str = "en") -> str:
    """Koo & Li (2016) qualitative label for an ICC value."""
    labels = {
        "en": ("poor", "moderate", "good", "excellent"),
        "tr": ("zayıf", "orta", "iyi", "mükemmel"),
    }[lang]
    if value < 0.5:
        return labels[0]
    if value < 0.75:
        return labels[1]
    if value < 0.9:
        return labels[2]
    return labels[3]


# ------------------------------------------------------------------ Bland-Altman
@dataclass
class BlandAltmanResult:
    n: int
    bias: float
    sd_diff: float
    loa_low: float
    loa_high: float
    bias_ci: tuple
    loa_low_ci: tuple
    loa_high_ci: tuple
    prop_bias_slope: float
    prop_bias_p: float
    means: np.ndarray
    diffs: np.ndarray

    def to_dict(self, include_points: bool = False) -> dict:
        d = asdict(self)
        if not include_points:
            d.pop("means")
            d.pop("diffs")
        else:
            d["means"] = self.means.tolist()
            d["diffs"] = self.diffs.tolist()
        return d


def bland_altman(a, b, alpha: float = 0.05) -> BlandAltmanResult:
    """Bland-Altman agreement between two methods (difference = a - b).

    Confidence intervals for the bias and the limits of agreement follow
    Bland & Altman (1999) (SE of each limit ~= sqrt(3 * s^2 / n)).
    Proportional bias is tested by regressing the differences on the means.
    """
    a, b = _clean_pairs(a, b)
    n = len(a)
    diff = a - b
    mean = (a + b) / 2
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    z = _st.norm.ppf(1 - alpha / 2)
    t = _st.t.ppf(1 - alpha / 2, n - 1)
    lo, hi = bias - z * sd, bias + z * sd
    se_bias = sd / np.sqrt(n)
    se_loa = np.sqrt(3 * sd ** 2 / n)
    reg = _st.linregress(mean, diff)
    return BlandAltmanResult(
        n=n, bias=bias, sd_diff=sd, loa_low=float(lo), loa_high=float(hi),
        bias_ci=(float(bias - t * se_bias), float(bias + t * se_bias)),
        loa_low_ci=(float(lo - t * se_loa), float(lo + t * se_loa)),
        loa_high_ci=(float(hi - t * se_loa), float(hi + t * se_loa)),
        prop_bias_slope=float(reg.slope), prop_bias_p=float(reg.pvalue),
        means=mean, diffs=diff,
    )


# ------------------------------------------------------------- test-retest
def typical_error(trial1, trial2, swc_factor: float = 0.2) -> dict:
    """Absolute reliability for test-retest data (Hopkins, 2000).

    Returns typical error (TE = SD of differences / sqrt 2, also used as the
    SEM), coefficient of variation (CV% = 100 x TE / grand mean), minimal
    detectable change (MDC95 = 1.96 x sqrt 2 x TE) and the smallest
    worthwhile change (SWC = ``swc_factor`` x between-subject SD of trial 1).
    """
    a, b = _clean_pairs(trial1, trial2)
    te = float(np.std(b - a, ddof=1) / np.sqrt(2))
    grand = float(np.mean(np.concatenate([a, b])))
    swc = float(swc_factor * np.std(a, ddof=1))
    return {
        "n": int(len(a)),
        "typical_error": te,
        "sem": te,
        "cv_percent": 100 * te / grand if grand else float("nan"),
        "mdc95": 1.96 * np.sqrt(2) * te,
        "swc": swc,
        "te_less_than_swc": te < swc,
    }


# ------------------------------------------------------------- effect sizes
def cohens_d(a, b, paired: bool = True) -> dict:
    """Cohen's d for the difference ``a - b``.

    For paired data both ``d_z`` (mean difference / SD of differences) and
    ``d_av`` (mean difference / average SD; Lakens, 2013) are returned.
    For independent groups the pooled-SD d and Hedges' g are returned.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if paired:
        a, b = _clean_pairs(a, b)
        diff = a - b
        dz = diff.mean() / diff.std(ddof=1)
        d_av = diff.mean() / ((a.std(ddof=1) + b.std(ddof=1)) / 2)
        return {"d_z": float(dz), "d_av": float(d_av)}
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    return {"d": float(d), "hedges_g": float(g)}


def pearson(a, b, alpha: float = 0.05) -> dict:
    """Pearson r with Fisher-z confidence interval."""
    a, b = _clean_pairs(a, b)
    r, p = _st.pearsonr(a, b)
    n = len(a)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zc = _st.norm.ppf(1 - alpha / 2)
    return {"r": float(r), "p": float(p), "ci_low": float(np.tanh(z - zc * se)),
            "ci_high": float(np.tanh(z + zc * se)), "n": n}


def paired_t(a, b) -> dict:
    """Paired-samples t-test (a - b)."""
    a, b = _clean_pairs(a, b)
    res = _st.ttest_rel(a, b)
    return {"t": float(res.statistic), "df": int(len(a) - 1), "p": float(res.pvalue)}


def mean_sd(x) -> tuple:
    """Mean and sample SD ignoring NaN."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean()), float(x.std(ddof=1))


# ------------------------------------------------------------- one-call summary
def agreement(a, b, icc_model: str = "2,1") -> dict:
    """All agreement statistics between two methods (or two trials) at once.

    Typical use: ``a`` = new method (e.g. video), ``b`` = criterion device.
    """
    a, b = _clean_pairs(a, b)
    ba = bland_altman(a, b)
    out = {
        "n": len(a),
        "mean_sd_a": mean_sd(a),
        "mean_sd_b": mean_sd(b),
        "paired_t": paired_t(a, b),
        "cohens_d": cohens_d(a, b, paired=True),
        "pearson": pearson(a, b),
        "icc": icc(np.column_stack([a, b]), icc_model).to_dict(),
        "bland_altman": ba.to_dict(),
    }
    return out


def reliability(trial1, trial2, icc_model: str = "2,1") -> dict:
    """Test-retest (or inter-rater) reliability summary for two trials."""
    a, b = _clean_pairs(trial1, trial2)
    return {
        "n": len(a),
        "mean_sd_1": mean_sd(a),
        "mean_sd_2": mean_sd(b),
        "paired_t": paired_t(b, a),
        "icc": icc(np.column_stack([a, b]), icc_model).to_dict(),
        "typical_error": typical_error(a, b),
        "bland_altman": bland_altman(b, a).to_dict(),
    }
