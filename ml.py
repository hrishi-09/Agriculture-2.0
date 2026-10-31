"""Models: RandomForest + neural-net (MLP) ensemble, GradientBoosting yield, statistics, SHAP."""
import os, joblib, numpy as np, pandas as pd, shap
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

F = ["N", "P", "K", "ph", "moisture", "temp", "rain", "humidity"]
PROFILE = {"rice": [100,45,40,6.3,75,27,1200,80], "wheat": [110,50,40,6.8,45,18,500,55],
           "maize": [90,50,45,6.5,55,24,750,65], "potato": [80,60,100,5.8,60,17,600,70],
           "tomato": [75,65,80,6.4,55,25,650,65]}
SD = [15, 8, 10, .4, 8, 2.5, 120, 7]
PATH = "models/m.joblib"

def synth(n=4000, seed=7):
    """SYNTHETIC data for demo. Replace with real datasets (see proposal 7.2)."""
    r = np.random.default_rng(seed); X, y = [], []
    for c, mu in PROFILE.items():
        X.append(r.normal(mu, SD, (n // 5, 8))); y += [c] * (n // 5)
    X = pd.DataFrame(np.vstack(X), columns=F).clip(lower=0)
    t = (4.5 + .012*(X.N-100) + .6*np.exp(-(X.ph-6.5)**2) + .01*(X.moisture-60)
         - .02*(X.temp-26)**2 + .0004*(X.rain-800) + r.normal(0, .3, len(X)))
    return X, np.array(y), t.clip(lower=.5)

def train():
    X, y, t = synth(); os.makedirs("models", exist_ok=True)
    Xa, Xb, ya, yb, ta, tb = train_test_split(X, y, t, test_size=.2, random_state=1, stratify=y)
    rf = RandomForestClassifier(300, random_state=1).fit(Xa, ya)
    nn = make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=600, random_state=1)).fit(Xa, ya)
    gb = GradientBoostingRegressor(random_state=1).fit(Xa, ta)
    pe = (rf.predict_proba(Xb) + nn.predict_proba(Xb)) / 2
    m = dict(rf=rf, nn=nn, gb=gb, mu=X.mean().values, ic=np.linalg.inv(np.cov(X.values.T)),
             rmse=float(np.sqrt(mean_squared_error(tb, gb.predict(Xb)))),
             metrics=dict(rf_acc=accuracy_score(yb, rf.predict(Xb)), nn_acc=accuracy_score(yb, nn.predict(Xb)),
                          ens_acc=accuracy_score(yb, rf.classes_[pe.argmax(1)]), yield_r2=r2_score(tb, gb.predict(Xb))))
    joblib.dump(m, PATH); return m

def load():
    return joblib.load(PATH) if os.path.exists(PATH) else train()

def analyze(m, x):
    v = pd.DataFrame([x])[F]; a = v.values[0]
    p = (m["rf"].predict_proba(v)[0] + m["nn"].predict_proba(v)[0]) / 2   # ensemble probability
    k = int(p.argmax()); cls = m["rf"].classes_
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))          # normalised entropy
    d2 = float((a - m["mu"]) @ m["ic"] @ (a - m["mu"]))
    ood = float(1 - stats.chi2.cdf(d2, len(F)))                           # Mahalanobis -> chi2 p-value
    sv = shap.TreeExplainer(m["rf"]).shap_values(v)
    s = sv[k][0] if isinstance(sv, list) else sv[0, :, k]
    return dict(crop=cls[k], conf=float(p[k]), probs=dict(zip(cls, p.astype(float))), entropy=ent, ood_p=ood,
                yield_t=float(m["gb"].predict(v)[0]), ci=1.96 * m["rmse"], shap=dict(zip(F, map(float, s))))
