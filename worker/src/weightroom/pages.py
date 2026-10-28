"""The few pages people see: landing, sign up, log in, dashboard and the agent guide.

Plain HTML with a little script; the dashboard calls /account/* with the login cookie.
Brand lines come from BRAND.md and are used as written.
"""

import html
import re

from . import config, credits, guide

CSS = """
:root{--bg:#f7f5f0;--ink:#16181d;--muted:#5d6370;--line:#dedad0;--card:#fffdf8;--accent:#1f6f4a;--accent-ink:#fff;
--code:#efece4;--warn:#9a3b1b}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--ink:#ecebe6;--muted:#a3a8b2;--line:#2b2f36;--card:#191c21;
--accent:#4fbf8a;--accent-ink:#0d1210;--code:#20242a;--warn:#f08a64}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent)}code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em}
code{background:var(--code);padding:.1em .35em;border-radius:4px}
pre{background:var(--code);padding:14px 16px;border-radius:8px;overflow-x:auto;white-space:pre-wrap;word-break:break-word}
pre code{background:none;padding:0}
.wrap{max-width:1040px;margin:0 auto;padding:0 20px}
header{border-bottom:1px solid var(--line)}
nav{display:flex;align-items:center;gap:18px;height:60px;flex-wrap:wrap}
nav .logo{font-weight:700;letter-spacing:-.02em;font-size:18px;color:var(--ink);text-decoration:none;margin-right:auto}
nav a{text-decoration:none;color:var(--muted);font-size:15px}nav a.btn{color:var(--accent-ink)}
.btn{display:inline-block;background:var(--accent);color:var(--accent-ink);padding:10px 16px;border-radius:8px;
text-decoration:none;font-weight:600;border:0;cursor:pointer;font-size:15px}
.btn.ghost{background:transparent;color:var(--ink);border:1px solid var(--line)}
.hero{padding:72px 0 40px}.hero h1{font-size:clamp(34px,6vw,56px);line-height:1.05;letter-spacing:-.03em;margin:0 0 18px;max-width:820px}
.hero p{font-size:19px;color:var(--muted);max-width:680px;margin:0 0 26px}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:var(--accent);letter-spacing:.04em;margin-bottom:14px}
.row{display:flex;gap:12px;flex-wrap:wrap}
section{padding:44px 0;border-top:1px solid var(--line)}h2{font-size:26px;letter-spacing:-.02em;margin:0 0 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
.card h3{margin:0 0 6px;font-size:17px}.card p{margin:0;color:var(--muted);font-size:15px}
.stat{font-size:34px;font-weight:700;letter-spacing:-.02em}.stat small{font-size:15px;color:var(--muted);font-weight:500}
.step{font-family:ui-monospace,Menlo,monospace;color:var(--accent);font-size:13px;margin-bottom:6px}
.price{font-size:30px;font-weight:700}.price small{font-size:14px;color:var(--muted);font-weight:500}
ul.ticks{padding-left:18px;margin:10px 0 0;color:var(--muted);font-size:15px}
footer{border-top:1px solid var(--line);padding:30px 0 50px;color:var(--muted);font-size:14px}
form.stack{display:flex;flex-direction:column;gap:12px;max-width:420px}
input,select{font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);width:100%}
label{font-size:14px;color:var(--muted)}
.msg{padding:12px 14px;border-radius:8px;background:var(--code);margin:12px 0}.msg.err{color:var(--warn)}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line);vertical-align:top}
.table-wrap{overflow-x:auto}
.muted{color:var(--muted)}.small{font-size:14px}
article.guide{max-width:780px;padding:40px 0}article.guide h1{letter-spacing:-.02em}
"""

LINES = ["From Models to Moats.", "Build Intelligence. Own It.", "Train Models. Build Moats.",
         "Your Models. Your Intelligence."]


def layout(title: str, body: str, *, script: str = "") -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><meta name="description" content="Train a model on your data. Own the weights.">
<link rel="alternate" type="text/plain" href="/llms.txt" title="Agent guide">
<style>{CSS}</style></head><body>
<header><div class="wrap"><nav><a class="logo" href="/">Weightroom</a>
<a href="/#how">How it works</a><a href="/#pricing">Pricing</a><a href="/docs/agents">Agent guide</a>
<a href="/login">Log in</a><a class="btn" href="/signup">Get an API key</a></nav></div></header>
{body}
<footer><div class="wrap">{" · ".join(LINES)}</div></footer>
{f"<script>{script}</script>" if script else ""}</body></html>"""


def landing() -> str:
    base = config.get().public_base_url
    p = credits.PLANS
    body = f"""
<main class="wrap">
<div class="hero"><div class="eyebrow">FROM MODELS TO MOATS.</div>
<h1>Train a model on your data. Own the weights.</h1>
<p>Weightroom is the tool your coding agent calls to fine-tune an open model on your data, prove it beats the
original on examples it never saw, and hand you weights you own. No ML team.</p>
<div class="row"><a class="btn" href="/signup">Get an API key, free credits included</a>
<a class="btn ghost" href="#connect">Connect your agent</a></div></div>

<section><h2>Proof, on a public benchmark</h2><div class="grid">
<div class="card"><div class="stat">67.5% → 82.5%</div><h3>Qwen3.5-4B, for about 41¢</h3>
<p>Banking77: sort customer messages into 77 intents. Accuracy on 200 held-back messages, before and after
training on 1,000 examples. About 11 minutes.</p></div>
<div class="card"><div class="stat">76.0% → 92.0%</div><h3>Kimi-K2.6 (1T parameters), for about $4.12</h3>
<p>Same benchmark, 1,540 examples. A trillion-parameter open model, trained and owned.</p></div>
</div></section>

<section id="how"><h2>How it works</h2><div class="grid">
<div class="card"><div class="step">1 · CHECK</div><h3>Free, instant</h3><p>Columns, labels, balance and errors,
each with the exact fix. 10% of rows are held back for the scorecard.</p></div>
<div class="card"><div class="step">2 · QUOTE</div><h3>The price before you spend</h3><p>Exact credits and minutes.
Your agent asks you before anything is charged.</p></div>
<div class="card"><div class="step">3 · TRAIN</div><h3>Runs on its own</h3><p>LoRA on Tinker. Retries never
double-charge. Failed runs are refunded.</p></div>
<div class="card"><div class="step">4 · OWN</div><h3>Proof, then weights</h3><p>Original vs trained on held-back
rows. Then the adapter, your private Hugging Face repo, or a live endpoint.</p></div>
</div></section>

<section id="connect"><h2>Connect your agent</h2>
<p class="muted">Remote MCP server. In Claude Code:</p>
<pre><code>claude mcp add --transport http weightroom {html.escape(base)}/mcp --header "Authorization: Bearer $WEIGHTROOM_API_KEY"</code></pre>
<p class="muted">Then ask: <em>"Train a model on tickets.csv to sort support tickets by urgency. Keep it private. Budget $10."</em>
Any MCP client works; there's a REST API too. Agents: read <a href="/llms.txt">/llms.txt</a>.</p></section>

<section id="pricing"><h2>Pricing</h2><div class="grid">
<div class="card"><h3>Free</h3><div class="price">$0</div><ul class="ticks"><li>10 credits a month</li>
<li>About one small training run</li><li>"Trained with Weightroom" on your model card</li></ul></div>
<div class="card"><h3>Pro</h3><div class="price">${p['pro']['usd_per_month']}<small>/month</small></div><ul class="ticks">
<li>{p['pro']['monthly_credits']} credits a month</li><li>Top-ups: {credits.TOPUP_CREDITS} credits for ${credits.TOPUP_USD}</li>
<li>Live endpoints from the same balance</li></ul></div>
<div class="card"><h3>Business</h3><div class="price">${p['business']['usd_per_month']}<small>/month</small></div><ul class="ticks">
<li>{p['business']['monthly_credits']} credits a month</li><li>Spend caps per agent key</li><li>Team keys</li></ul></div>
<div class="card"><h3>Enterprise</h3><div class="price">Custom</div><ul class="ticks"><li>Runs in your own cloud</li>
<li>Volume pricing and support</li><li><a href="{html.escape(config.get().enterprise_contact)}">Contact us</a></li></ul></div>
</div><p class="muted small">Credits buy training and endpoint use. A 4B model on ~1,000 examples is about 10 credits;
a 1T-parameter model about 100.</p></section>
</main>"""
    return layout("Weightroom: train a model on your data, own the weights", body)


def signup() -> str:
    body = """<main class="wrap"><div class="hero" style="padding-bottom:20px"><h1>Get an API key</h1>
<p>Free credits included. Your agent uses the key; you approve every spend.</p></div>
<form class="stack" id="f"><label>Email<input name="email" type="email" required autocomplete="email"></label>
<label>Password (10+ characters)<input name="password" type="password" minlength="10" required autocomplete="new-password"></label>
<button class="btn" type="submit">Create account</button><p class="small muted">Have an account? <a href="/login">Log in</a></p></form>
<div id="out"></div></main>"""
    script = """
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const d=Object.fromEntries(new FormData(e.target));
const r=await fetch('/account/signup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
const j=await r.json();const o=document.getElementById('out');
if(!r.ok){o.innerHTML='<div class="msg err"></div>';o.firstChild.textContent=j.error.message+' '+j.error.fix;return}
e.target.remove();o.innerHTML='<h2>Your API key</h2><p class="muted">Copy it now; it won\\'t be shown again.</p><pre><code id=k></code></pre>'+
'<h2>Connect Claude Code</h2><pre><code id=c></code></pre><p><a class="btn" href="/dashboard">Go to your dashboard</a></p>';
document.getElementById('k').textContent=j.api_key;
document.getElementById('c').textContent='claude mcp add --transport http weightroom '+location.origin+'/mcp --header "Authorization: Bearer '+j.api_key+'"';};
"""
    return layout("Sign up · Weightroom", body, script=script)


def login() -> str:
    body = """<main class="wrap"><div class="hero" style="padding-bottom:20px"><h1>Log in</h1></div>
<form class="stack" id="f"><label>Email<input name="email" type="email" required autocomplete="email"></label>
<label>Password<input name="password" type="password" required autocomplete="current-password"></label>
<button class="btn" type="submit">Log in</button><p class="small muted">New here? <a href="/signup">Get an API key</a></p></form>
<div id="out"></div></main>"""
    script = """
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const d=Object.fromEntries(new FormData(e.target));
const r=await fetch('/account/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
if(r.ok){location.href='/dashboard';return}const j=await r.json();const o=document.getElementById('out');
o.innerHTML='<div class="msg err"></div>';o.firstChild.textContent=j.error.message;};
"""
    return layout("Log in · Weightroom", body, script=script)


def dashboard() -> str:
    body = """<main class="wrap" style="padding:36px 20px 20px"><div id="app"><p class="muted">Loading…</p></div></main>"""
    script = r"""
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path,body){const r=await fetch(path,{method:body===undefined?'GET':'POST',
headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
const j=await r.json().catch(()=>({}));if(r.status===401){location.href='/login';throw 0}
if(!r.ok)throw new Error((j.error&&(j.error.message+' '+j.error.fix))||'Something went wrong');return j}
function flash(t,err){const m=document.getElementById('flash');m.className='msg'+(err?' err':'');m.textContent=t;m.hidden=false}
async function render(){const me=await api('/account/me');const a=document.getElementById('app');
const paid=new URLSearchParams(location.search).get('paid');
a.innerHTML=`<div class="row" style="align-items:center;justify-content:space-between"><h1 style="margin:0">Dashboard</h1>
<button class="btn ghost" id="logout">Log out</button></div><p class="muted">${esc(me.email)}</p>
<div id="flash" class="msg" hidden></div>${paid?'<div class="msg">Payment received. Credits appear within a minute.</div>':''}
<div class="grid" style="margin:20px 0"><div class="card"><h3>Credits</h3><div class="stat">${me.balance_credits}</div>
<p>${esc(me.plan_name)} plan</p></div><div class="card"><h3>Plan</h3><div class="row" style="margin-top:8px">
${me.plan==='free'?'<button class="btn" data-buy="pro">Upgrade to Pro</button><button class="btn ghost" data-buy="business">Business</button>':
'<button class="btn ghost" id="portal">Manage billing</button>'}
${me.can_top_up?'<button class="btn" data-buy="topup">Top up ${me.top_up.credits} credits ($${me.top_up.usd})</button>':''}</div></div></div>
<section><h2>API keys</h2><p class="muted small">Give each agent its own key. Caps are in credits; blank means no cap.</p>
<div class="table-wrap"><table><tr><th>Name</th><th>Key</th><th>Spent (30 days)</th><th>Monthly cap</th><th>Per-job cap</th><th></th></tr>
${me.keys.map(k=>`<tr data-id="${k.id}"><td>${esc(k.name)}</td><td><code>${esc(k.prefix)}</code></td><td>${k.spent_last_30_days}</td>
<td><input class="m" value="${k.monthly_cap_credits??''}" inputmode="numeric" style="width:90px"></td>
<td><input class="j" value="${k.per_job_cap_credits??''}" inputmode="numeric" style="width:90px"></td>
<td class="row"><button class="btn ghost save">Save</button><button class="btn ghost revoke">Revoke</button></td></tr>`).join('')}
</table></div><form class="row" id="newkey" style="margin-top:14px"><input name="name" placeholder="Key name, e.g. claude-code" style="max-width:240px">
<input name="monthly_cap" placeholder="Monthly cap" inputmode="numeric" style="max-width:140px">
<input name="per_job_cap" placeholder="Per-job cap" inputmode="numeric" style="max-width:140px"><button class="btn">New key</button></form>
<div id="newkeyout"></div></section>
<section><h2>Connect your agent</h2><pre><code>claude mcp add --transport http weightroom ${esc(location.origin)}/mcp --header "Authorization: Bearer YOUR_KEY"</code></pre></section>
<section><h2>Hugging Face</h2><p class="muted small">For pushing trained models to your private repos. Use a write token from
<a href="https://huggingface.co/settings/tokens">huggingface.co/settings/tokens</a>. Stored encrypted.</p>
<form class="row" id="hf"><input name="token" type="password" placeholder="${me.has_hf_token?'Saved. Paste a new one to replace':'hf_…'}" style="max-width:360px">
<button class="btn">Save token</button></form></section>
<section><h2>Training jobs</h2>${me.jobs.length?`<div class="table-wrap"><table><tr><th>Job</th><th>Model</th><th>State</th><th>Credits</th><th>Result</th></tr>
${me.jobs.map(j=>`<tr><td><code>${esc(j.job_id)}</code></td><td>${esc(j.base_model)}</td><td>${esc(j.state)}</td><td>${j.credits}</td><td class="small">${esc(j.summary)}</td></tr>`).join('')}</table></div>`:
'<p class="muted">None yet. Ask your agent to train one.</p>'}</section>`;
document.getElementById('logout').onclick=async()=>{await api('/account/logout',{});location.href='/'};
document.querySelectorAll('[data-buy]').forEach(b=>b.onclick=async()=>{try{location.href=(await api('/account/checkout',{item:b.dataset.buy})).url}catch(e){flash(e.message,1)}});
const p=document.getElementById('portal');if(p)p.onclick=async()=>{try{location.href=(await api('/account/portal',{})).url}catch(e){flash(e.message,1)}};
document.querySelectorAll('tr[data-id]').forEach(r=>{const id=r.dataset.id;
r.querySelector('.save').onclick=async()=>{try{await api(`/account/keys/${id}/caps`,{monthly_cap:r.querySelector('.m').value,per_job_cap:r.querySelector('.j').value});flash('Caps saved.')}catch(e){flash(e.message,1)}};
r.querySelector('.revoke').onclick=async()=>{if(!confirm('Revoke this key? Agents using it stop working.'))return;await api(`/account/keys/${id}/revoke`,{});render()};});
document.getElementById('newkey').onsubmit=async e=>{e.preventDefault();try{const j=await api('/account/keys',Object.fromEntries(new FormData(e.target)));
await render();const o=document.getElementById('newkeyout');o.innerHTML='<div class="msg">Copy this key now; it won\'t be shown again.</div><pre><code></code></pre>';
o.querySelector('code').textContent=j.api_key}catch(e){flash(e.message,1)}};
document.getElementById('hf').onsubmit=async e=>{e.preventDefault();try{await api('/account/hf-token',Object.fromEntries(new FormData(e.target)));flash('Hugging Face token saved.');render()}catch(e){flash(e.message,1)}};}
render();
"""
    return layout("Dashboard · Weightroom", body, script=script)


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"(https?://[^\s<)]+)", r'<a href="\1">\1</a>', text)


def markdown(md: str) -> str:
    """Just enough Markdown for the guide: headings, paragraphs, lists, quotes and code blocks."""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{_inline(line[level:].strip())}</h{level}>")
        elif re.match(r"^(\d+\.|-) ", line):
            tag = "ol" if line[0].isdigit() else "ul"
            items = []
            while i < len(lines) and (re.match(r"^(\d+\.|-) ", lines[i]) or lines[i].startswith("   ")):
                if lines[i].startswith("   ") and items:
                    items[-1] += " " + lines[i].strip()
                else:
                    items.append(re.sub(r"^(\d+\.|-) ", "", lines[i]))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(it)}</li>" for it in items) + f"</{tag}>")
            continue
        elif line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip("> "))
                i += 1
            out.append(f"<p class='muted'>{_inline(' '.join(quote))}</p>")
            continue
        elif line.strip():
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|```|>|\d+\. |- )", lines[i]):
                para.append(lines[i])
                i += 1
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            continue
        i += 1
    return "\n".join(out)


def agent_guide() -> str:
    return layout("Agent guide · Weightroom", f'<main class="wrap"><article class="guide">{markdown(guide.text())}</article></main>')
