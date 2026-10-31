import os, json, re, secrets, bcrypt, numpy as np, pandas as pd
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, session, flash, abort, Response
from scipy import stats
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import ml

url = os.getenv("DATABASE_URL", "sqlite:///agri.db")
url = re.sub(r"^postgres(ql)?://", "postgresql+psycopg2://", url)
eng = create_engine(url, pool_pre_ping=True); Base = declarative_base(); Sess = sessionmaker(eng)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True); name = Column(String(80)); email = Column(String(120), unique=True)
    pw = Column(String(100)); role = Column(String(10), default="farmer"); active = Column(Boolean, default=True)
    created = Column(DateTime, default=datetime.utcnow)

class Pred(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True); user_id = Column(Integer); crop = Column(String(30))
    confidence = Column(Float); yield_t = Column(Float); ood_p = Column(Float); inputs = Column(Text)
    created = Column(DateTime, default=datetime.utcnow)

def hp(p): return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
Base.metadata.create_all(eng)
with Sess() as s:
    ae = os.getenv("ADMIN_EMAIL", "admin@agrixai.local").lower()
    if not s.query(User).filter_by(email=ae).first():
        s.add(User(name="Administrator", email=ae, pw=hp(os.getenv("ADMIN_PASSWORD", "ChangeMe123!")), role="admin")); s.commit()
M = ml.load()

app = Flask(__name__); app.secret_key = os.getenv("SECRET_KEY", "dev-change-me")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=bool(os.getenv("RENDER")))
RNG = dict(N=(0,200,100), P=(0,150,50), K=(0,200,60), ph=(3.5,9,6.4), moisture=(0,100,60), temp=(5,45,24), rain=(100,2500,800), humidity=(10,100,65))

@app.context_processor
def ctx():
    session.setdefault("csrf", secrets.token_hex(16)); return dict(csrf=session["csrf"], user=session.get("user"))

@app.before_request
def csrf():
    if request.method == "POST" and request.form.get("csrf") != session.get("csrf"): abort(400)

def login_required(f):
    @wraps(f)
    def w(*a, **k): return f(*a, **k) if session.get("user") else redirect("/login")
    return w

def admin_required(f):
    @wraps(f)
    def w(*a, **k): return f(*a, **k) if session.get("user", {}).get("role") == "admin" else abort(403)
    return w

def df(q, **kw):
    # Avoid pd.read_sql: newer pandas rejects older SQLAlchemy versions (caused HTTP 500 on Render).
    with eng.connect() as c:
        r = c.execute(text(q), kw)
        return pd.DataFrame(r.fetchall(), columns=list(r.keys()))

@app.route("/")
def home(): return redirect("/dashboard")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        with Sess() as s:
            u = s.query(User).filter_by(email=request.form["email"].strip().lower()).first()
            if u and u.active and bcrypt.checkpw(request.form["password"].encode(), u.pw.encode()):
                session["user"] = dict(id=u.id, name=u.name, role=u.role); return redirect("/dashboard")
        flash("Invalid credentials or disabled account.")
    return render_template("auth.html", mode="login")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        n, e, p, c = (request.form[k].strip() for k in ("name", "email", "password", "confirm")); e = e.lower()
        if not n or not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+$", e): flash("Enter a name and a valid email.")
        elif len(p) < 8 or p != c: flash("Passwords must match and be 8+ characters.")
        else:
            with Sess() as s:
                if s.query(User).filter_by(email=e).first(): flash("Email already registered.")
                else: s.add(User(name=n, email=e, pw=hp(p))); s.commit(); flash("Registered. Please sign in."); return redirect("/login")
    return render_template("auth.html", mode="register")

@app.route("/logout", methods=["POST"])
def logout(): session.clear(); return redirect("/login")

@app.route("/dashboard")
@login_required
def dashboard():
    u = session["user"]; p = df("select * from predictions" + ("" if u["role"] == "admin" else f" where user_id={int(u['id'])}"))
    d = dict(n=len(p), metrics=M["metrics"], crops={}, conf=[], mean=None, ci=None, flags=0, yld=None)
    if len(p) > 1:
        x = p.confidence; ci = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x))
        d.update(crops=p.crop.value_counts().to_dict(), conf=x.round(3).tolist(), mean=float(x.mean()), ci=[float(ci[0]), float(ci[1])],
                 flags=int((p.ood_p < .01).sum()), yld=float(p.yield_t.mean()))
    return render_template("dashboard.html", d=d)

@app.route("/advisor", methods=["GET", "POST"])
@login_required
def advisor():
    r = None; x = {k: v[2] for k, v in RNG.items()}
    if request.method == "POST":
        try: x = {k: min(max(float(request.form[k]), RNG[k][0]), RNG[k][1]) for k in RNG}
        except (ValueError, KeyError): abort(400)
        r = ml.analyze(M, x)
        with Sess() as s:
            s.add(Pred(user_id=session["user"]["id"], crop=r["crop"], confidence=r["conf"], yield_t=r["yield_t"], ood_p=r["ood_p"], inputs=json.dumps(x))); s.commit()
        top = sorted(r["shap"], key=lambda k: -abs(r["shap"][k]))[:3]
        r["unsure"] = r["entropy"] > .6 or r["ood_p"] < .01; r["top"] = top
    return render_template("advisor.html", r=r, x=x, RNG=RNG)

@app.route("/history")
@login_required
def history():
    p = df("select id,crop,confidence,yield_t,ood_p,created from predictions where user_id=:i order by id desc limit 200", i=session["user"]["id"])
    return render_template("table.html", title="My History", cols=list(p.columns), rows=p.round(3).values.tolist())

@app.route("/admin", methods=["GET", "POST"])
@login_required
@admin_required
def admin():
    if request.method == "POST":
        uid, act = int(request.form["uid"]), request.form["action"]
        with Sess() as s:
            t = s.get(User, uid)
            if t and t.id != session["user"]["id"]:
                if act == "toggle": t.active = not t.active
                elif act == "role" and request.form["role"] in ("farmer", "agronomist", "admin"): t.role = request.form["role"]
                elif act == "delete": s.delete(t)
                s.commit()
            else: flash("Cannot modify yourself or a missing user.")
    p = df("select id,name,email,role,active,created from users order by id")
    return render_template("admin.html", cols=list(p.columns), rows=p.values.tolist())

@app.route("/database", methods=["GET", "POST"])
@login_required
@admin_required
def database():
    t = request.values.get("table", "users"); t = t if t in ("users", "predictions") else "users"
    d = df(f"select * from {t}"); d = d.drop(columns=["pw"], errors="ignore")
    if request.args.get("export"): return Response(d.to_csv(index=False), mimetype="text/csv", headers={"Content-Disposition": f"attachment;filename={t}.csv"})
    q = request.form.get("sql", "select crop, count(*) n, avg(confidence) from predictions group by crop"); res = None
    if request.method == "POST":
        if not q.strip().lower().startswith("select") or ";" in q or re.search(r"\bpw\b|\*\s*from\s+users", q.lower()): flash("Only single SELECT statements that don't read password hashes are allowed.")
        else:
            try:
                with eng.connect() as c: r = c.execute(text(q)); res = (list(r.keys()), [list(x) for x in r.fetchmany(200)])
            except Exception as e: flash(str(e)[:200])
    return render_template("db.html", t=t, cols=list(d.columns), rows=d.head(200).values.tolist(), q=q, res=res)
