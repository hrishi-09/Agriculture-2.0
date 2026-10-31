import os, io, json, re, base64, secrets, bcrypt, numpy as np, pandas as pd
from datetime import datetime, date
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

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True); user_id = Column(Integer); title = Column(String(200)); due = Column(String(10))
    priority = Column(String(10), default="medium"); done = Column(Boolean, default=False); created = Column(DateTime, default=datetime.utcnow)

class Scan(Base):
    __tablename__ = "scans"
    id = Column(Integer, primary_key=True); user_id = Column(Integer); crop = Column(String(60)); health = Column(String(40))
    severity = Column(String(10)); confidence = Column(Float); result = Column(Text); created = Column(DateTime, default=datetime.utcnow)

class Chat(Base):
    __tablename__ = "chat"
    id = Column(Integer, primary_key=True); user_id = Column(Integer); role = Column(String(10)); content = Column(Text)
    created = Column(DateTime, default=datetime.utcnow)

def hp(p): return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
Base.metadata.create_all(eng)
with Sess() as s:
    ae = os.getenv("ADMIN_EMAIL", "admin@agrixai.local").lower()
    if not s.query(User).filter_by(email=ae).first():
        s.add(User(name="Administrator", email=ae, pw=hp(os.getenv("ADMIN_PASSWORD", "ChangeMe123!")), role="admin")); s.commit()
M = ml.load()

app = Flask(__name__); app.secret_key = os.getenv("SECRET_KEY", "dev-change-me")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=bool(os.getenv("RENDER")), MAX_CONTENT_LENGTH=8 * 1024 * 1024)
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
        r["report"] = ml.soil_report(r["crop"], x); r["top3"] = sorted(r["probs"].items(), key=lambda kv: -kv[1])[:3]
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


# ---------- AI helpers (Claude API) ----------
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

class NoKey(Exception): pass

def claude(system, messages, max_tokens=1000):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key: raise NoKey()
    import anthropic
    r = anthropic.Anthropic(api_key=key, timeout=60).messages.create(model=MODEL, max_tokens=max_tokens, system=system, messages=messages)
    return "".join(b.text for b in r.content if b.type == "text")

NOKEY_MSG = "AI features need an ANTHROPIC_API_KEY. Add it in Render > Environment, then redeploy."

def prep_image(stream):
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(stream)).convert("RGB"); im.thumbnail((1568, 1568))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85); return base64.b64encode(b.getvalue()).decode()

def parse_json(t):
    m = re.search(r"\{.*\}", t, re.S)
    return json.loads(m.group(0)) if m else None

def farm_context(uid):
    """Short text summary of the user's latest soil inputs, report, scan and open tasks for the AI."""
    out = []
    with Sess() as s:
        p = s.query(Pred).filter_by(user_id=uid).order_by(Pred.id.desc()).first()
        if p:
            x = json.loads(p.inputs); rep = ml.soil_report(p.crop, x)
            out.append(f"Latest soil/weather inputs: {x}. Model recommends {p.crop} (confidence {p.confidence:.0%}, expected yield {p.yield_t:.2f} t/ha).")
            out.append(f"Soil health score {rep['score']}/100 ({rep['label']}). Suggested actions: " + "; ".join(a["text"] for a in rep["actions"]))
        sc = s.query(Scan).filter_by(user_id=uid).order_by(Scan.id.desc()).first()
        if sc: out.append(f"Latest photo scan: crop {sc.crop}, condition {sc.health}, severity {sc.severity}.")
        ts = s.query(Task).filter_by(user_id=uid, done=False).order_by(Task.id).limit(30).all()
        out.append("Open tasks: " + ("; ".join(f"{t.title} (due {t.due or 'no date'}, {t.priority})" for t in ts) if ts else "none"))
    if not any(o.startswith("Latest soil") for o in out): out.insert(0, "The user has not run a soil analysis yet.")
    return "\n".join(out)

SCAN_SYS = """You are an expert plant pathologist and agronomist. The user uploads a photo of a crop (leaf, fruit, whole plant or field).
Reply with ONLY one JSON object, no markdown, with these keys:
is_plant (bool), crop (string, best guess or "unknown"), part (leaf/fruit/stem/whole plant/field/other),
health ("healthy","nutrient deficiency","fungal disease","bacterial disease","viral disease","pest damage","water stress","heat/cold stress","other","unclear"),
issue (short name of the most likely problem, or "none"), confidence (0-1), severity ("none","low","medium","high"),
evidence (list of up to 4 short visible signs you based this on), next_steps (ordered list of 3-6 concrete actions, most urgent first),
prevention (list of up to 3 tips), see_expert (bool, true if severity is high or you are unsure), summary (one plain sentence).
Be honest: if the photo is blurry or not a plant, set is_plant false or health "unclear" and ask for a better photo in next_steps.
Name chemical treatments only generically (e.g. "a copper-based fungicide") and tell the farmer to follow the label and local agriculture-office advice."""

CHAT_SYS = """You are the AgriXAI Assistant: a practical farm-work organiser and soil/crop advisor for small and mid-size farmers (India-friendly units: kg/ha, mm, INR).
Use the farm context below. Give short, specific, prioritised answers; use short bullet lists when planning. Never invent measurements that are not in the context; ask for missing data instead.
For pesticide/fertiliser doses, give indicative numbers and remind the user to confirm with a soil test or local agriculture office. If unsure, say so.
FARM CONTEXT:
"""

@app.errorhandler(413)
def too_big(e): flash("Image is too large (max 8 MB)."); return redirect("/scan")

@app.route("/scan", methods=["GET", "POST"])
@login_required
def scan():
    res = None; uid = session["user"]["id"]
    if request.method == "POST":
        f = request.files.get("image")
        if not f or not f.filename: flash("Choose a photo first.")
        else:
            try: b64 = prep_image(f.stream)
            except Exception: b64 = None; flash("That file is not a valid image (use JPG, PNG or WebP).")
            if b64:
                note = request.form.get("note", "")[:300]
                msg = [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                                                   {"type": "text", "text": "Analyse this crop photo. Farmer's note: " + (note or "none")}]}]
                try: res = parse_json(claude(SCAN_SYS, msg, 1200))
                except NoKey: flash(NOKEY_MSG)
                except Exception as e: flash("AI service error: " + str(e)[:150])
                else:
                    if not res: flash("Could not read the AI answer. Please try again.")
                    else:
                        for k in ("evidence", "next_steps", "prevention"):
                            if not isinstance(res.get(k), list): res[k] = []
                        try: conf = float(res.get("confidence") or 0)
                        except (TypeError, ValueError): conf = 0.0
                        with Sess() as s:
                            s.add(Scan(user_id=uid, crop=str(res.get("crop", "unknown"))[:60], health=str(res.get("health", ""))[:40],
                                       severity=str(res.get("severity", ""))[:10], confidence=conf, result=json.dumps(res))); s.commit()
    h = df("select crop,health,severity,confidence,created from scans where user_id=:i order by id desc limit 10", i=uid)
    return render_template("scan.html", r=res, hist=h.round(2).values.tolist())

@app.route("/tasks", methods=["GET", "POST"])
@login_required
def tasks():
    uid = session["user"]["id"]; plan = None
    if request.method == "POST":
        act = request.form.get("action")
        with Sess() as s:
            if act in ("add", "addmany"):
                pr = request.form.get("priority", "medium"); pr = pr if pr in ("high", "medium", "low") else "medium"
                due = request.form.get("due", "").strip(); due = due if re.fullmatch(r"\d{4}-\d{2}-\d{2}", due) else ""
                for ttl in request.form.getlist("title"):
                    if ttl.strip(): s.add(Task(user_id=uid, title=ttl.strip()[:200], due=due, priority=pr))
                s.commit()
            elif act in ("done", "reopen", "delete"):
                t = s.get(Task, int(request.form.get("id", 0)))
                if t and t.user_id == uid:
                    if act == "delete": s.delete(t)
                    else: t.done = act == "done"
                    s.commit()
        if act == "plan":
            try: plan = claude(CHAT_SYS + farm_context(uid), [{"role": "user", "content": f"Today is {date.today().isoformat()}. Build a prioritised 7-day work plan from my open tasks and soil situation. Group by day, give a one-line reason for each item, and list anything important I am missing."}], 1200)
            except NoKey: flash(NOKEY_MSG)
            except Exception as e: flash("AI service error: " + str(e)[:150])
        elif act in ("add", "addmany", "done", "reopen", "delete"): return redirect("/tasks")
    with Sess() as s:
        rows = s.query(Task).filter_by(user_id=uid).order_by(Task.done, Task.due == "", Task.due, Task.id).limit(200).all()
        today = date.today().isoformat()
        items = [dict(id=t.id, title=t.title, due=t.due, priority=t.priority, done=t.done, over=bool(t.due and not t.done and t.due < today)) for t in rows]
    return render_template("tasks.html", items=items, plan=plan)

@app.route("/assistant", methods=["GET", "POST"])
@login_required
def assistant():
    uid = session["user"]["id"]
    if request.method == "POST":
        if request.form.get("action") == "clear":
            with Sess() as s: s.query(Chat).filter_by(user_id=uid).delete(); s.commit()
            return redirect("/assistant")
        q = (request.form.get("quick") or request.form.get("message") or "").strip()[:2000]
        if q:
            with Sess() as s:
                prev = s.query(Chat).filter_by(user_id=uid).order_by(Chat.id.desc()).limit(12).all()[::-1]
            msgs = [{"role": c.role, "content": c.content} for c in prev] + [{"role": "user", "content": q}]
            try: a = claude(CHAT_SYS + farm_context(uid), msgs, 1000)
            except NoKey: flash(NOKEY_MSG); a = None
            except Exception as e: flash("AI service error: " + str(e)[:150]); a = None
            if a:
                with Sess() as s: s.add(Chat(user_id=uid, role="user", content=q)); s.add(Chat(user_id=uid, role="assistant", content=a)); s.commit()
        return redirect("/assistant")
    with Sess() as s:
        msgs = [(c.role, c.content) for c in s.query(Chat).filter_by(user_id=uid).order_by(Chat.id.desc()).limit(30).all()[::-1]]
    return render_template("assistant.html", msgs=msgs)
