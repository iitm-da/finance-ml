"""Assemble the second deck: AI across business functions (45 minutes)."""
import math, pathlib
import build  # reuses head, tail, css and js; also rebuilds deck one

d = pathlib.Path(__file__).parent
head = (d / "_head.html").read_text(encoding="utf-8")
tail = (d / "_tail.html").read_text(encoding="utf-8")

css2 = """
/* function slides */
.fl{position:absolute;left:120px;width:1140px}
.fl .lab{font-size:24px;font-weight:600;color:var(--muted);margin-bottom:12px!important}
.fl .row{display:flex;gap:30px}
.st{position:relative;flex:1;min-height:205px;border-radius:12px;padding:20px 20px;font-size:26px;line-height:1.28;background:var(--soft)}
.st:not(:last-child)::after{content:"";position:absolute;right:-22px;top:50%;margin-top:-9px;border:9px solid transparent;border-left:12px solid #B9BED6;border-right:0}
.st b{display:block;font-size:21px;font-weight:600;margin-bottom:10px}
.st.old{color:var(--muted)}
.st.llm{background:var(--bluesoft)}.st.llm b{color:#1B33C9}
.st.vlm{background:var(--redsoft)}.st.vlm b{color:#B8231A}
.st.slm{background:var(--yellowsoft)}.st.slm b{color:#7A5B00}
.st.rule{background:#fff;border:2px solid var(--line)}.st.rule b{color:var(--muted)}
.st.hum{background:var(--ink);color:#fff}.st.hum b{color:var(--yellow)}
.side{position:absolute;left:1340px;top:262px;width:460px}
.side .ev{border-top:2px solid var(--ink);padding-top:18px}
.side .ev .big{font-size:64px;color:var(--blue)}
.side .ev p.c{font-size:25px;line-height:1.32;color:var(--muted);margin-top:8px}
.side .kv{border-top:1px solid var(--line);padding-top:16px;margin-top:24px}
.side .kv p:first-child{font-size:22px;font-weight:600;color:var(--muted)}
.side .kv p:last-child{font-size:25px;line-height:1.32;margin-top:6px}
.fn{position:absolute;top:290px;width:216px;height:150px;border-radius:12px;background:var(--soft);padding:18px 18px;font-size:26px;font-weight:600;line-height:1.2;display:flex;flex-direction:column;justify-content:space-between}
.fn span{font-size:20px;font-weight:400;color:var(--muted);line-height:1.25}
.pt{position:absolute;display:flex;align-items:center;gap:12px;font-size:24px;font-weight:500;white-space:nowrap}
.pt i{width:20px;height:20px;border-radius:50%;background:var(--blue);border:2px solid #fff;box-shadow:0 0 0 1px var(--blue);flex:none}
.blank{border-bottom:2px solid var(--ink);height:112px;display:flex;align-items:flex-end;padding-bottom:10px;font-size:28px;font-weight:600}
.blank span{font-weight:400;color:var(--faint);margin-left:16px;font-size:24px}
"""

TAGS = {"llm": "Large model", "vlm": "Vision model", "slm": "Small model", "rule": "Rules", "hum": "Person", "old": ""}


def flow(label, steps, top, delay):
    cells = ""
    for i, (kind, text) in enumerate(steps):
        tag = f"<b>{TAGS[kind]}</b>" if TAGS[kind] else ""
        cells += f'<div class="st {kind} a" style="--d:{delay + i * .12:.2f}s">{tag}{text}</div>'
    return f'<div class="fl" style="top:{top}px"><p class="lab a" style="--d:{delay:.2f}s">{label}</p><div class="row">{cells}</div></div>'


def fslide(n, title, today, withai, big, cap, track, watch, src):
    return f"""
<!-- {n} -->
<section class="slide" data-sec="The functions">
  <h2 class="title">{title}</h2>
  {flow("Today", [("old", t) for t in today], 262, .2)}
  {flow("With language models", withai, 540, .6)}
  <div class="abs a" style="left:120px;top:850px;width:1140px;display:flex;gap:28px;font-size:22px;color:var(--muted);--d:1.2s">
    <span><span class="key" style="background:var(--yellowsoft);border:1px solid #E5CF7A"></span>Small model</span><span><span class="key" style="background:var(--redsoft);border:1px solid #F0B5B0"></span>Vision model</span><span><span class="key" style="background:var(--bluesoft);border:1px solid #B9C4FF"></span>Large model</span><span><span class="key" style="background:var(--ink)"></span>Person decides</span>
  </div>
  <div class="side">
    <div class="ev a" style="--d:.5s"><p class="big">{big}</p><p class="c">{cap}</p></div>
    <div class="kv a" style="--d:.7s"><p>Track</p><p>{track}</p></div>
    <div class="kv a" style="--d:.9s"><p>Watch</p><p>{watch}</p></div>
  </div>
  <div class="foot"><span>{src}</span><span class="pn"></span></div>
</section>"""


F = [
 ("Research: the model reads and drafts; the analyst verifies and decides",
  ["Analyst reads the annual report, concall and broker notes", "Updates the model and notes by hand", "Writes the research note"],
  [("vlm", "Pulls tables and figures out of filings"), ("llm", "Summarises, with a link to the source page for each claim"), ("hum", "Analyst checks the claims and adds the view"), ("llm", "Drafts the note in house format")],
  "2 days to 30 min", "for one macro analysis at Balyasny. BCG estimates 2&#8211;5x research coverage per analyst.",
  "Companies covered per analyst; share of claims carrying a source link",
  "Fluent but wrong figures. The widely cited &#8220;GPT-4 beats analysts&#8221; paper was withdrawn.",
  "Sources: OpenAI case study on Balyasny (vendor-published, March 2026); BCG (July 2026); arXiv 2407.17866"),
 ("Fund management: AI briefs, flags and records; it does not place orders",
  ["Manager scans news, notes and messages", "Decides; dealer executes", "Rationale is written up afterwards"],
  [("llm", "Morning brief on each holding from news and filings"), ("slm", "Flags items that touch a holding or a scheme limit"), ("hum", "Manager decides; dealer executes"), ("llm", "Drafts the rationale record from the manager's notes")],
  "30%", "lower trading costs at Norges Bank Investment Management over 2023&#8211;25, worth NOK 4.8 billion, with AI a significant part.",
  "Trade errors; time from decision to recorded rationale",
  "Holdings and pending orders must stay on systems the AMC controls.",
  "Source: NBIM chief executive, Storting hearing (May 2026). BCG estimates 50&#8211;70% fewer trade errors (July 2026)."),
 ("Operations: vision models read the forms; people handle only the exceptions",
  ["Form and proofs arrive as paper or scans", "Staff key in each field", "A checker verifies again", "Record goes to the registrar"],
  [("vlm", "Reads the form, ID, cheque and bank proof"), ("rule", "Fields checked against KYC and bank records"), ("hum", "Reviews only low-confidence cases"), ("rule", "Clean record goes to the registrar")],
  "55% to 87%", "field accuracy on KYC files at OCBC once the pipeline found the right page first, with an 8B model.",
  "First-time-right rate; share of cases a person touches",
  "Noisy scans, handwriting and long files still defeat small vision models.",
  "Source: OCBC, multistage KYC extraction pipeline (2026). BCG estimates 55&#8211;65% more operations capacity (July 2026)."),
 ("Risk and compliance: each circular mapped to the policies it changes",
  ["SEBI issues a circular", "Compliance reads and circulates it", "Each department interprets it", "Policies are updated"],
  [("llm", "Breaks the circular into separate obligations"), ("llm", "Matches each to policies and scheme documents"), ("hum", "Compliance officer confirms the gaps"), ("slm", "Screens marketing drafts and staff messages")],
  "Ranked first", "for cost savings from generative AI in EY's survey of wealth and asset managers: compliance, risk and IT operations.",
  "Days from circular to updated policy; alerts closed per reviewer",
  "A missed obligation. A person confirms every mapping.",
  "Sources: EY (Sept 2025); CAMS already does this for registrar clients with CAMSLens (Business Today, Nov 2025)"),
 ("Sales and distribution: one assistant behind 1,26,600 distributors and 271 locations",
  ["Distributor calls with a query", "Relationship manager searches factsheets and mail", "Reply goes out hours later"],
  [("slm", "Classifies the query and its language"), ("llm", "Answers from approved documents, citing the page"), ("hum", "Manager takes exceptions and large tickets"), ("rule", "Propensity models choose whom to call next")],
  "&#8377;5,000 cr", "incremental gross sales at Nippon India in FY26 from machine-learning propensity campaigns.",
  "Response time to distributors; clients covered per manager",
  "An assistant that recommends a scheme without a suitability check is giving advice.",
  "Sources: NAM India Annual Report 2025-26 and Q2 FY27 presentation. BCG estimates 3&#8211;5x client coverage and 5&#8211;10x RFP capacity."),
 ("Marketing: the same approved facts, written in each investor's language",
  ["Writer drafts the commentary", "Compliance reviews", "Agency translates", "Release"],
  [("llm", "Drafts from portfolio data and approved facts"), ("llm", "Writes Hindi and regional-language versions"), ("slm", "Checks claims and disclaimers against the code"), ("hum", "Compliance approves every release")],
  "5%", "of Indian households prefer investor content in English; 47% prefer Hindi and 47% a regional language.",
  "Days from data cut-off to published commentary; share of content in regional languages",
  "The common advertisement code takes effect around April 2027.",
  "Sources: SEBI Investor Survey 2025; SEBI common advertisement code (Oct 2026). Nippon India has piloted AI-made Gujarati videos."),
 ("Investor service: small models sort the requests; large models draft the replies",
  ["Email or call arrives", "Agent reads it and works out the request", "Looks up the folio", "Replies"],
  [("slm", "Tags the request, language and urgency"), ("llm", "Drafts the reply from folio data and policy"), ("hum", "Agent approves; complaints always go to a person"), ("slm", "Checks tone and logs the exchange")],
  "8 million", "calls a year are handled by CAMS for its mutual fund clients. Kotak AMC reports its AI email triage saved almost two person-months.",
  "Turnaround time; first-contact resolution; complaints on AI-handled journeys",
  "Klarna cut agents first, saw quality fall and hired people back.",
  "Sources: CAMS Annual Report FY2024-25; Microsoft case study on Kotak AMC (vendor-published); Maginative (2025)"),
]

# ---- slide 3: volumes on a log scale
vols = [("Folios", 40.6e6, "40.6 million"), ("Unique investors", 24.3e6, "24.3 million"), ("Digital purchases, half year", 9.0e6, "9.0 million"),
        ("Empanelled distributors", 126600, "1,26,600"), ("Employees", 1222, "1,222"), ("Investor complaints, FY26", 1196, "1,196"), ("Locations", 271, "271")]
W = 640
rows = ""
for i, (name, v, lab) in enumerate(vols):
    w = (math.log10(v) - 2) / 6 * W
    y = 318 + i * 74
    col = "var(--blue)" if v > 1e6 else ("#8FA0FF" if v > 1e4 else "var(--ink)")
    rows += f'<p class="abs a" style="left:120px;top:{y + 6}px;font-size:26px;font-weight:600;--d:{.2 + i * .08:.2f}s">{name}</p>'
    rows += f'<div class="track" style="left:520px;top:{y}px;width:{W}px"><div class="bar" style="--w:{w:.0f}px;background:{col};--d:{.35 + i * .1:.2f}s"></div></div>'
    rows += f'<p class="abs a num" style="left:{520 + w + 16:.0f}px;top:{y + 4}px;font-size:28px;font-weight:600;--d:{1.3 + i * .1:.2f}s">{lab}</p>'
ticks = "".join(f'<p class="abs k" style="left:{520 + k * W / 6 - 40:.0f}px;top:842px;width:80px;text-align:center;font-size:21px">{t}</p><div class="abs" style="left:{520 + k * W / 6:.0f}px;top:318px;width:1px;height:516px;background:var(--line)"></div>'
                for k, t in enumerate(["100", "1,000", "10,000", "1 lakh", "10 lakh", "1 crore", "10 crore"]))

# ---- 2x2 points (x: cost of a wrong output 0..1, y: volume 0..1)
pts = [("Email and request triage", .12, .88), ("Form and KYC extraction", .30, .80), ("Circular-to-policy mapping", .16, .30),
       ("Research drafts for analysts", .28, .58), ("Distributor assistant", .56, .72), ("Regional-language content", .70, .40),
       ("Investor-facing chatbot", .84, .84), ("Dealing support", .88, .16)]
PX, PY, PW, PH = 420, 270, 1000, 560
plot = ""
for i, (name, x, y) in enumerate(pts):
    left = PX + x * PW - 12; top = PY + (1 - y) * PH - 12
    pos = f"right:{1920 - left - 24:.0f}px;flex-direction:row-reverse" if x > .6 else f"left:{left:.0f}px"
    plot += f'<div class="pt a" style="{pos};top:{top:.0f}px;--d:{.4 + i * .1:.2f}s"><i></i>{name}</div>'

fns = [("Research", "filings, notes"), ("Fund management", "briefs, records"), ("Operations", "forms, KYC"), ("Risk and compliance", "circulars, alerts"),
       ("Sales", "distributor queries"), ("Marketing", "content, languages"), ("Investor service", "emails, calls")]
fnb = "".join(f'<div class="fn a" style="left:{120 + i * 244}px;--d:{.2 + i * .08:.2f}s">{a}<span>{b}</span></div>' for i, (a, b) in enumerate(fns))
fnl = "".join(f'<div class="abs a" style="left:{120 + i * 244 + 107}px;top:440px;width:2px;height:90px;background:#B9BED6;--d:{.8 + i * .05:.2f}s"></div>' for i in range(7))

S = f"""
<svg style="display:none"><g id="flow"></g></svg><svg id="g95" style="display:none"></svg>
<!-- 1 COVER -->
<section class="slide dark" data-sec="">
  <canvas id="cv" class="abs" style="left:0;top:0;width:1920px;height:1080px" width="1920" height="1080" aria-hidden="true"></canvas>
  <div class="abs" style="left:120px;top:0;bottom:0;width:4px;background:var(--yellow)"></div>
  <p class="abs a" style="left:170px;top:300px;font-size:28px;font-weight:500;color:var(--yellow);--d:.1s">AI Masterclass for CXOs</p>
  <h1 class="abs a" style="left:166px;top:350px;width:1000px;font-size:120px;font-weight:600;letter-spacing:-.045em;line-height:1;--d:.2s">AI across business functions</h1>
  <p class="abs a" style="left:170px;top:640px;width:900px;font-size:38px;line-height:1.35;font-weight:300;color:rgba(255,255,255,.82);--d:.35s">Where language models fit in each part of an asset management company</p>
  <p class="abs a" style="left:170px;top:910px;font-size:26px;color:rgba(255,255,255,.6);--d:.5s">Dr. Arun Ayyar</p>
</section>

<!-- 2 MAP -->
<section class="slide" data-sec="The map">
  <h2 class="title">Seven functions do different work on one shared platform</h2>
  {fnb}{fnl}
  <div class="abs a" style="left:120px;top:530px;width:1680px;height:150px;border-radius:14px;background:var(--ink);color:#fff;display:grid;grid-template-columns:300px repeat(4,1fr);align-items:center;padding:0 40px;gap:24px;--d:1.1s">
    <p style="font-size:32px;font-weight:600">Shared platform</p>
    <p style="font-size:25px;line-height:1.3;color:rgba(255,255,255,.8)"><b style="color:#fff;display:block">AI gateway</b>routing, cost, access</p>
    <p style="font-size:25px;line-height:1.3;color:rgba(255,255,255,.8)"><b style="color:#fff;display:block">Approved data</b>documents, folio records</p>
    <p style="font-size:25px;line-height:1.3;color:rgba(255,255,255,.8)"><b style="color:#fff;display:block">Evaluation sets</b>real cases with answers</p>
    <p style="font-size:25px;line-height:1.3;color:rgba(255,255,255,.8)"><b style="color:#fff;display:block">Logs and controls</b>records, AMFI reporting</p>
  </div>
  <div class="abs" style="left:120px;top:750px;width:1680px;display:grid;grid-template-columns:repeat(3,1fr);gap:48px">
    <div class="a" style="--d:1.3s;border-top:2px solid var(--ink);padding-top:20px"><p style="font-size:30px;font-weight:600">Each function owns</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">its process, its number and the person who checks the output</p></div>
    <div class="a" style="--d:1.4s;border-top:2px solid var(--ink);padding-top:20px"><p style="font-size:30px;font-weight:600">The platform team owns</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">models, hosting, logging and the cost per task</p></div>
    <div class="a" style="--d:1.5s;border-top:2px solid var(--ink);padding-top:20px"><p style="font-size:30px;font-weight:600">Risk and compliance own</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:8px">testing before launch and what is reported to the regulator</p></div>
  </div>
  <div class="foot"><span>The next seven slides take one function each: the work today, the work with language models, one result and one risk</span><span class="pn"></span></div>
</section>

<!-- 3 VOLUMES -->
<section class="slide" data-sec="The map">
  <h2 class="title">The work ranges from 271 locations to 40.6 million folios, so the model must fit the volume</h2>
  <p class="abs a k" style="left:520px;top:270px;--d:.15s">Nippon India, latest published figures, on a log scale</p>
  {ticks}{rows}
  <div class="abs" style="left:1380px;top:318px;width:420px">
    <div class="a" style="--d:.8s;border-top:4px solid var(--blue);padding-top:16px"><p style="font-size:28px;font-weight:600">Millions of items</p><p class="body" style="font-size:25px;color:var(--muted);margin-top:6px">Small models and rules, because cost per item decides</p></div>
    <div class="a" style="--d:1s;border-top:4px solid #8FA0FF;padding-top:16px;margin-top:40px"><p style="font-size:28px;font-weight:600">Lakhs of partners</p><p class="body" style="font-size:25px;color:var(--muted);margin-top:6px">Large models answering from approved documents</p></div>
    <div class="a" style="--d:1.2s;border-top:4px solid var(--ink);padding-top:16px;margin-top:40px"><p style="font-size:28px;font-weight:600">About a thousand people and cases</p><p class="body" style="font-size:25px;color:var(--muted);margin-top:6px">Large models as assistants; a person decides</p></div>
  </div>
  <div class="foot"><span>Sources: NAM India Q2 FY27 release and investor presentation (Oct 2026); NAM India Annual Report 2025-26</span><span class="pn"></span></div>
</section>
"""
for i, f in enumerate(F):
    S += fslide(4 + i, *f)

S += f"""
<!-- 11 ENABLING -->
<section class="slide" data-sec="The functions">
  <h2 class="title">Finance, HR, legal and technology use the same three model types</h2>
  <table class="t" style="left:120px;top:260px;width:1680px;font-size:27px">
    <tr><th style="width:16%">Function</th><th style="width:34%">Work a model can take</th><th style="width:14%">Model</th><th>Person who checks</th></tr>
    <tr><td>Finance</td><td>Reading invoices and expense claims; first draft of board and management reports</td><td><span class="tag t-vlm">VLM</span> <span class="tag t-llm">LLM</span></td><td>Finance controller signs off every figure against the ledger</td></tr>
    <tr><td>HR</td><td>Answering policy questions; drafting role descriptions and learning plans</td><td><span class="tag t-llm">LLM</span></td><td>HR decides on hiring and appraisal; models are tested for bias</td></tr>
    <tr><td>Legal</td><td>Comparing distributor and vendor agreements against the standard terms</td><td><span class="tag t-llm">LLM</span></td><td>Counsel reviews every flagged clause</td></tr>
    <tr><td>Technology</td><td>Writing and testing code; explaining old systems; answering IT tickets</td><td><span class="tag t-llm">LLM</span> <span class="tag t-slm">SLM</span></td><td>Engineer reviews each change before release</td></tr>
  </table>
  <p class="abs a body" style="left:120px;top:830px;width:1680px;font-size:28px;--d:.9s"><b>These are usually the easiest places to begin,</b> because the output stays inside the company and a named person already signs it off.</p>
  <div class="foot"><span></span><span class="pn"></span></div>
</section>

<!-- AGENTS A1 -->
<section class="slide" data-sec="Agents">
  <h2 class="title">An agent chains the steps a function does today, with the three models sharing the work</h2>
  <table class="t tight" style="left:120px;top:240px;width:1680px;font-size:24px;line-height:1.25">
    <tr><th style="width:15%">Function</th><th style="width:16%">Agent</th><th>What it does from start to finish</th><th style="width:6%;text-align:center;padding-right:0">Small</th><th style="width:6%;text-align:center;padding-right:0">Vision</th><th style="width:6%;text-align:center;padding-right:0">Large</th><th style="width:14%;padding-left:24px">Who approves</th></tr>
    <tr><td>Research</td><td style="font-weight:600">Earnings-season agent</td><td>Collects the filing and transcript when results come out, pulls the tables, updates the draft note</td><td style="text-align:center"><span class="dot off"></span></td><td style="text-align:center"><span class="dot" style=background:var(--red)></span></td><td style="text-align:center"><span class="dot" style=background:var(--blue)></span></td><td>Analyst</td></tr>
    <tr><td>Fund management</td><td style="font-weight:600">Morning-brief agent</td><td>Watches overnight news on holdings, checks scheme limits, prepares the brief</td><td style="text-align:center"><span class="dot" style=background:var(--yellow)></span></td><td style="text-align:center"><span class="dot off"></span></td><td style="text-align:center"><span class="dot" style=background:var(--blue)></span></td><td>Fund manager</td></tr>
    <tr><td>Operations</td><td style="font-weight:600">Onboarding agent</td><td>Reads the form and proofs, checks KYC, asks the investor for what is missing</td><td style="text-align:center"><span class="dot" style=background:var(--yellow)></span></td><td style="text-align:center"><span class="dot" style=background:var(--red)></span></td><td style="text-align:center"><span class="dot" style=background:var(--blue)></span></td><td>Operations staff</td></tr>
    <tr><td>Risk and compliance</td><td style="font-weight:600">Circular agent</td><td>Spots a new circular, splits it into obligations, maps them to policies, opens tasks</td><td style="text-align:center"><span class="dot" style=background:var(--yellow)></span></td><td style="text-align:center"><span class="dot off"></span></td><td style="text-align:center"><span class="dot" style=background:var(--blue)></span></td><td>Compliance officer</td></tr>
    <tr><td>Sales</td><td style="font-weight:600">Distributor-desk agent</td><td>Answers the query from approved documents, logs it, schedules the follow-up</td><td style="text-align:center"><span class="dot" style=background:var(--yellow)></span></td><td style="text-align:center"><span class="dot off"></span></td><td style="text-align:center"><span class="dot" style=background:var(--blue)></span></td><td>Relationship manager</td></tr>
    <tr><td>Marketing</td><td style="font-weight:600">Factsheet agent</td><td>Pulls portfolio data, drafts commentary, translates it, runs the code check</td><td style="text-align:center"><span class="dot" style=background:var(--yellow)></span></td><td style="text-align:center"><span class="dot off"></span></td><td style="text-align:center"><span class="dot" style=background:var(--blue)></span></td><td>Compliance</td></tr>
    <tr><td>Investor service</td><td style="font-weight:600">Service agent</td><td>Tags the request, reads any attachment, fetches folio status, drafts the reply</td><td style="text-align:center"><span class="dot" style=background:var(--yellow)></span></td><td style="text-align:center"><span class="dot" style=background:var(--red)></span></td><td style="text-align:center"><span class="dot" style=background:var(--blue)></span></td><td>Service agent</td></tr>
    </table>
  <div class="abs a" style="left:120px;top:880px;width:1680px;--d:1.2s"><p class="body" style="font-size:27px"><b>The pattern repeats:</b> a small model handles the volume, a vision model reads documents where there are any, a large model plans and drafts, and a named person approves.</p></div>
  <div class="foot"><span>Illustrative designs; the model mix follows the function slides before this one</span><span class="pn"></span></div>
</section>

<!-- AGENTS A2 -->
<section class="slide" data-sec="Agents">
  <h2 class="title">Agents earn autonomy in three levels, and most AMC work belongs at the first two</h2>
  <div class="abs a" style="left:120px;top:560px;width:540px;height:290px;background:var(--bluesoft);border-radius:14px 14px 0 0;padding:26px 30px;--d:.2s"><p style="font-size:24px;font-weight:600;color:#1B33C9">Level 1</p><p style="font-size:34px;font-weight:600;margin-top:6px!important">Drafts</p><p style="font-size:25px;line-height:1.3;color:var(--muted);margin-top:10px!important">The model prepares; a person does everything after</p></div>
  <div class="abs a" style="left:680px;top:470px;width:540px;height:380px;background:#C9D3FF;border-radius:14px 14px 0 0;padding:26px 30px;--d:.4s"><p style="font-size:24px;font-weight:600;color:#1B33C9">Level 2</p><p style="font-size:34px;font-weight:600;margin-top:6px!important">Acts after approval</p><p style="font-size:25px;line-height:1.3;color:var(--ink2);margin-top:10px!important">The agent prepares the action; a person approves each one</p></div>
  <div class="abs a" style="left:1240px;top:320px;width:560px;height:530px;background:var(--blue);color:#fff;border-radius:14px 14px 0 0;padding:26px 30px;--d:.6s"><p style="font-size:24px;font-weight:600;color:var(--yellow)">Level 3</p><p style="font-size:34px;font-weight:600;margin-top:6px!important">Acts within limits</p><p style="font-size:25px;line-height:1.3;color:rgba(255,255,255,.85);margin-top:10px!important">The agent completes routine cases alone, inside set limits, with every action logged and sampled</p></div>
  <div class="abs" style="left:120px;top:850px;width:1680px;height:2px;background:var(--ink)"></div>
  <p class="abs a" style="left:150px;top:776px;width:480px;font-size:23px;line-height:1.3;--d:.8s"><b>Fits:</b> research notes, commentary, rationale records</p>
  <p class="abs a" style="left:710px;top:740px;width:480px;font-size:23px;line-height:1.3;--d:.9s"><b>Fits:</b> onboarding exceptions, distributor and investor replies</p>
  <p class="abs a" style="left:1270px;top:740px;width:500px;font-size:23px;line-height:1.3;color:#fff;--d:1s"><b>Fits:</b> tagging and routing requests, opening compliance tasks, internal look-ups</p>
  <div class="abs a" style="left:120px;top:270px;width:1040px;--d:1.1s"><p class="body" style="font-size:28px"><b>A working rule:</b> anything that reaches an investor or the order book stays at level 2 or below.</p><p class="body" style="font-size:26px;color:var(--muted);margin-top:14px">Today 7% of asset managers have deployed agents and 5% give AI any autonomous authority in investing.</p></div>
  <div class="foot"><span>The rule is our recommendation. Sources: EY (Sept 2025); Mercer (May 2026)</span><span class="pn"></span></div>
</section>

<!-- 12 WHERE TO START -->
<section class="slide" data-sec="Choosing">
  <h2 class="title">Start where volume is high and a wrong output stays inside the company</h2>
  <div class="abs" style="left:{PX}px;top:{PY}px;width:{PW}px;height:{PH}px;border-left:2px solid var(--ink);border-bottom:2px solid var(--ink)"></div>
  <div class="abs a" style="left:{PX}px;top:{PY}px;width:{PW // 2}px;height:{PH // 2}px;background:var(--bluesoft);--d:.2s"></div>
  <div class="abs" style="left:{PX + PW // 2}px;top:{PY}px;width:1px;height:{PH}px;background:var(--line)"></div>
  <div class="abs" style="left:{PX}px;top:{PY + PH // 2}px;width:{PW}px;height:1px;background:var(--line)"></div>
  <p class="abs a" style="left:{PX + 18}px;top:{PY + 14}px;font-size:24px;font-weight:600;color:#1B33C9;--d:.3s">Start here</p>
  <p class="abs a" style="left:{PX + PW // 2 + 18}px;top:{PY + 14}px;font-size:24px;font-weight:600;color:var(--muted);--d:.3s">Next, with a person approving</p>
  <p class="abs a" style="left:{PX + 18}px;top:{PY + PH - 44}px;font-size:24px;font-weight:600;color:var(--muted);--d:.3s">Useful, smaller payoff</p>
  <p class="abs a" style="left:{PX + PW // 2 + 18}px;top:{PY + PH - 44}px;font-size:24px;font-weight:600;color:var(--muted);--d:.3s">Later</p>
  {plot}
  <p class="abs k" style="left:120px;top:{PY}px;width:270px;text-align:right">High volume of language work</p>
  <p class="abs k" style="left:120px;top:{PY + PH - 34}px;width:270px;text-align:right">Low volume</p>
  <p class="abs k" style="left:{PX}px;top:{PY + PH + 16}px">Wrong output stays internal</p>
  <p class="abs k" style="left:{PX + PW - 420}px;top:{PY + PH + 16}px;width:420px;text-align:right">Wrong output reaches an investor or an order</p>
  <div class="abs a" style="left:1480px;top:{PY}px;width:320px;--d:1.3s"><p class="body" style="font-size:26px;color:var(--muted)">Positions are our assessment, for discussion. Each function head should place their own processes on this chart.</p></div>
  <div class="foot"><span>SEBI's June 2025 draft takes the same view: lighter rules for tools that do not directly affect investors</span><span class="pn"></span></div>
</section>

<!-- 13 WHAT EACH HEAD BRINGS -->
<section class="slide" data-sec="Choosing">
  <h2 class="title">A function head brings four things to every use case</h2>
  <div class="abs" style="left:120px;top:270px;width:1680px;display:grid;grid-template-columns:repeat(4,1fr);gap:40px">
    <div class="a" style="--d:.2s;border-top:4px solid var(--ink);padding-top:24px"><p class="big" style="font-size:56px;color:var(--blue)">1</p><p style="font-size:34px;font-weight:600;margin-top:16px">An owner</p><p class="body" style="font-size:27px;color:var(--muted);margin-top:14px">One named person accountable for the outcome, from the business and not from IT</p></div>
    <div class="a" style="--d:.3s;border-top:4px solid var(--ink);padding-top:24px"><p class="big" style="font-size:56px;color:var(--blue)">2</p><p style="font-size:34px;font-weight:600;margin-top:16px">A baseline</p><p class="body" style="font-size:27px;color:var(--muted);margin-top:14px">Today's cost, turnaround time and error rate for the process, measured before any build</p></div>
    <div class="a" style="--d:.4s;border-top:4px solid var(--ink);padding-top:24px"><p class="big" style="font-size:56px;color:var(--blue)">3</p><p style="font-size:34px;font-weight:600;margin-top:16px">200&#8211;500 real cases</p><p class="body" style="font-size:27px;color:var(--muted);margin-top:14px">Past items with the correct answer, so every model and every release is scored the same way</p></div>
    <div class="a" style="--d:.5s;border-top:4px solid var(--ink);padding-top:24px"><p class="big" style="font-size:56px;color:var(--blue)">4</p><p style="font-size:34px;font-weight:600;margin-top:16px">A checkpoint</p><p class="body" style="font-size:27px;color:var(--muted);margin-top:14px">The step where a person approves, and the cases that always go to a person</p></div>
  </div>
  <div class="abs a" style="left:120px;top:800px;width:1680px;background:var(--soft);border-radius:14px;padding:28px 40px;--d:.8s"><p class="body" style="font-size:28px"><b>Without these four, a pilot cannot show a return.</b> In Mercer's 2026 survey, 55% of managers use AI in an investment process and 8% report a measurable improvement.</p></div>
  <div class="foot"><span>Source: Mercer, AI in asset management survey of 131 managers (May 2026)</span><span class="pn"></span></div>
</section>

<!-- 14 EXERCISE -->
<section class="slide" data-sec="Exercise">
  <h2 class="title">Exercise: describe one process from your function in five lines</h2>
  <div class="abs" style="left:120px;top:250px;width:1060px">
    <div class="blank a" style="--d:.2s">Process<span>for example, replies to distributor queries</span></div>
    <div class="blank a" style="--d:.3s">Volume a month<span>items, and people working on them</span></div>
    <div class="blank a" style="--d:.4s">What is read and written<span>text, scans, tables, languages</span></div>
    <div class="blank a" style="--d:.5s">Who checks it today<span>and how long that takes</span></div>
    <div class="blank a" style="--d:.6s">Cost of one wrong output<span>internal rework, or an investor affected</span></div>
  </div>
  <div class="abs a" style="left:1280px;top:270px;width:520px;background:var(--ink);color:#fff;border-radius:16px;padding:36px 40px;--d:.8s">
    <p style="font-size:26px;font-weight:500;color:var(--yellow)">Then read the answer off</p>
    <p style="font-size:26px;line-height:1.4;margin-top:20px">Scans or tables in line 3: <b>vision model</b></p>
    <p style="font-size:26px;line-height:1.4;margin-top:14px">High volume, narrow task: <b>small model</b></p>
    <p style="font-size:26px;line-height:1.4;margin-top:14px">Many sources, drafting: <b>large model</b></p>
    <p style="font-size:26px;line-height:1.4;margin-top:14px">Line 5 names an investor: <b>a person approves</b></p>
  </div>
  <div class="foot"><span>Ten minutes: five to write, five to compare across functions</span><span class="pn"></span></div>
</section>

<!-- 15 CLOSE -->
<section class="slide dark" data-sec="">
  <div class="abs" style="left:120px;top:0;bottom:0;width:4px;background:var(--yellow)"></div>
  <h2 class="abs a" style="left:166px;top:330px;width:1640px;font-size:84px;font-weight:600;letter-spacing:-.04em;line-height:1.08;--d:.15s">Each function owns its process and its number.<br><span style="color:rgba(255,255,255,.55)">The platform and the controls are shared.</span></h2>
  <p class="abs a" style="left:170px;top:800px;font-size:30px;color:rgba(255,255,255,.7);--d:.5s">Thank you. Questions and discussion.</p>
</section>

"""

head = head.replace("</style>", build.css + css2 + "</style>")
head = head.replace("<title>Language is the ledger: a CXO briefing</title>", "<title>AI across business functions in asset management</title>")
tail = tail.replace("</body>", build.js + "</body>")
out = d.parent / "ai_across_business_functions.html"
out.write_text(head + S + tail, encoding="utf-8")
print("wrote", out)
