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


def soil_report(crop, x):
    """Rule-based agronomy report for the recommended crop. Indicative only: confirm doses with a lab soil test."""
    t = dict(zip(F, PROFILE[crop])); sd = dict(zip(F, SD)); acts = []
    score = 100 * float(np.mean([max(0, 1 - abs(x[k] - t[k]) / (3 * sd[k])) for k in F]))
    label = "Excellent" if score >= 80 else "Good" if score >= 60 else "Fair" if score >= 40 else "Poor"
    nut = []
    for k, name, conv, prod in (("N", "Nitrogen", .46, "urea"), ("P", "Phosphorus", .46, "DAP"), ("K", "Potassium", .60, "MOP (potash)")):
        d = t[k] - x[k]; row = dict(nutrient=name, value=round(x[k], 1), target=t[k])
        if d > .15 * t[k]:
            kg = round(d / conv)
            row.update(status="Low", advice=f"Apply about {kg} kg/ha {prod}, split into 2 doses.")
            acts.append(("high", f"Apply ~{kg} kg/ha {prod} to correct low {name.lower()}"))
        elif -d > .30 * t[k]:
            row.update(status="High", advice="Skip this nutrient this season; excess can lock up others.")
            acts.append(("low", f"Skip {name.lower()} fertiliser this season (already high)"))
        else:
            row.update(status="OK", advice="Within the range for this crop.")
        nut.append(row)
    dp = x["ph"] - t["ph"]
    if dp < -.5:
        ph = dict(status="Acidic", advice="Add agricultural lime (roughly 1-2 t/ha per 0.5 pH unit on medium soils) and organic matter.")
        acts.append(("high", "Apply agricultural lime to raise soil pH"))
    elif dp > .7:
        ph = dict(status="Alkaline", advice="Use gypsum or elemental sulphur plus compost; prefer acid-forming fertilisers such as ammonium sulphate.")
        acts.append(("high", "Add gypsum/sulphur and compost to lower soil pH"))
    else:
        ph = dict(status="OK", advice="pH suits this crop.")
    dm = t["moisture"] - x["moisture"]
    if dm > 10:
        mm = round(dm * 3)
        water = dict(status="Dry", advice=f"Irrigate about {mm} mm now (about 3 mm per 1% moisture over a 30 cm root zone), preferably early morning.")
        acts.append(("high", f"Irrigate ~{mm} mm (soil is dry)"))
    elif dm < -20:
        water = dict(status="Too wet", advice="Improve drainage and hold irrigation; waterlogging starves roots of oxygen.")
        acts.append(("high", "Improve drainage / pause irrigation (waterlogged)"))
    else:
        water = dict(status="OK", advice="Moisture is in a good range.")
    if x["rain"] < .6 * t["rain"]:
        water["advice"] += " Rainfall is well below this crop's need, so plan supplementary irrigation."
    if x["temp"] > t["temp"] + 6:
        clim = "Heat stress likely: mulch, irrigate in the early morning, consider shade net for seedlings."
        acts.append(("medium", "Mulch and irrigate early morning (heat stress)"))
    elif x["temp"] < t["temp"] - 8:
        clim = "Cold stress likely: delay sowing or protect seedlings."
        acts.append(("medium", "Delay sowing / protect seedlings from cold"))
    else:
        clim = "Temperature is suitable."
    if x["humidity"] >= 80 and 18 <= x["temp"] <= 32:
        dis = dict(level="High", advice="Warm and humid: fungal diseases and blight spread fast. Scout every 3-4 days, avoid overhead watering, keep plants spaced.")
        acts.append(("medium", "Scout for fungal disease every 3-4 days (humid weather)"))
    elif x["humidity"] >= 70:
        dis = dict(level="Medium", advice="Moderate disease pressure. Scout weekly and keep leaves dry.")
    else:
        dis = dict(level="Low", advice="Low disease pressure from weather.")
    acts.append(("low", "Re-test soil in 4-6 weeks to confirm the changes worked"))
    order = {"high": 0, "medium": 1, "low": 2}
    acts = [dict(priority=p, text=s) for p, s in sorted(acts, key=lambda a: order[a[0]])]
    return dict(crop=crop, score=round(score), label=label, nutrients=nut, ph=ph, water=water, climate=clim, disease=dis, actions=acts)
