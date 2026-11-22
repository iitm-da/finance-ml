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
.sp{display:grid;grid-template-columns:150px 1fr;align-items:center;column-gap:20px;min-height:98px;border-radius:12px;padding:12px 22px;margin-bottom:12px;background:var(--soft)}
.sp b{font-size:21px;font-weight:600;line-height:1.2}.sp span{font-size:26px;line-height:1.28}
.sp.llm{background:var(--bluesoft)}.sp.llm b{color:#1B33C9}
.sp.vlm{background:var(--redsoft)}.sp.vlm b{color:#B8231A}
.sp.slm{background:var(--yellowsoft)}.sp.slm b{color:#7A5B00}
.sp.rule{background:#fff;border:2px solid var(--line)}.sp.rule b{color:var(--muted)}
.sp.hum{background:var(--ink);color:#fff}.sp.hum b{color:var(--yellow)}
.card{position:relative;background:#fff;border-radius:12px;padding:22px 24px;box-shadow:0 4px 16px rgba(14,19,48,.08);overflow:hidden}
.card .ln{height:9px;border-radius:4px;background:#D5D9E8;margin:0 0 13px}.card .ln.d{background:#3A4166;height:13px;margin-bottom:20px}
.card .hl{position:relative;height:22px;border-radius:4px;background:#FFE58A;margin:0 0 13px;width:92%}
.card .hl i{position:absolute;right:-12px;top:-8px;width:28px;height:28px;border-radius:50%;background:var(--blue);color:#fff;font:600 16px/28px Geist,Arial;text-align:center}
.card .tb{height:92px;margin:4px 0 16px;border:2px solid #D5D9E8;background:linear-gradient(#D5D9E8,#D5D9E8) 0 30px/100% 2px no-repeat,linear-gradient(#D5D9E8,#D5D9E8) 0 60px/100% 2px no-repeat,linear-gradient(#D5D9E8,#D5D9E8) 45% 0/2px 100% no-repeat,linear-gradient(#D5D9E8,#D5D9E8) 72% 0/2px 100% no-repeat}
.cap{font-size:19px;color:var(--faint)}
.ct{font-size:25px;font-weight:600;margin-bottom:14px!important}
.cl{font-size:23px;line-height:1.3;padding:14px 0;border-top:1px solid var(--line)}
.cl.flag{color:#B8231A;font-weight:600}
.cite{display:inline-block;font-size:18px;font-weight:600;color:#1B33C9;background:var(--bluesoft);border-radius:999px;padding:3px 12px;margin-left:8px;white-space:nowrap}
.rw{display:grid;grid-template-columns:18px 1fr;column-gap:18px;align-items:start;padding:15px 0;border-bottom:1px solid var(--line)}
.rw i{width:18px;height:18px;border-radius:50%;margin-top:7px}
.rw p{font-size:24px;line-height:1.3}.rw p b{font-weight:600}
.rw p span{display:block;font-size:21px;color:var(--muted);margin-top:5px}
.rw em{font-style:normal;font-size:18px;font-weight:600;background:var(--soft);border-radius:999px;padding:3px 11px;margin-right:6px;color:var(--ink2)}
.fk{font-size:15px;color:var(--faint);margin:10px 0 4px!important}
.fb{border:2px solid var(--red);border-radius:4px;padding:5px 9px;font:700 19px 'Courier New',monospace;color:var(--ink2)}
.ob{background:var(--bluesoft);border-radius:10px;padding:14px 18px;margin-bottom:14px;font-size:22px;font-weight:600;color:#1B33C9}
.ob span{display:block;font-size:20px;font-weight:400;color:var(--ink2);margin-top:3px}
.pol{position:absolute;left:470px;width:310px;height:84px;background:#fff;border-radius:12px;padding:12px 18px;border:2px solid var(--blue);font-size:22px;display:flex;flex-direction:column;justify-content:center}
.pol.gap{border:2px dashed var(--red)}.pol span{font-size:20px;margin-top:3px}.pol.gap span{color:#B8231A;font-weight:600}
.bub{max-width:560px;background:#fff;border-radius:16px 16px 16px 4px;padding:12px 20px;font-size:22px;line-height:1.3;box-shadow:0 2px 8px rgba(14,19,48,.06)}
.bub.me{align-self:flex-end;background:var(--bluesoft);border-radius:16px 16px 4px 16px}
.bub span:last-child{display:block;font-size:17px;color:var(--faint);margin-top:6px}
.lang{font-size:24px;line-height:1.45;font-weight:500;font-family:'Geist','Nirmala UI','Noto Sans Devanagari','Noto Sans Tamil','Lohit Devanagari','Lohit Tamil',sans-serif}
.tk{display:inline-block;width:14px;height:14px;border-radius:50%;background:var(--blue);margin-right:10px}
.en{border-top:4px solid var(--ink);padding-top:22px}
.en svg{display:block;margin-bottom:18px}
.en h3{font-size:34px;font-weight:600;margin-bottom:12px!important}
.en p{font-size:25px;line-height:1.32}
.en p.who{font-size:22px;color:var(--muted);margin-top:16px!important;padding-top:14px;border-top:1px solid var(--line)}
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


def fslide(n, title, today, withai, big, cap, track, watch, src, art=""):
    steps = ""
    for i, (kind, text) in enumerate(withai):
        steps += f'<div class="sp {kind} a" style="--d:{.5 + i * .12:.2f}s"><b>{TAGS[kind]}</b><span>{text}</span></div>'
    return f"""
<!-- {n} -->
<section class="slide" data-sec="The functions">
  <h2 class="title">{title}</h2>
  <p class="abs a" style="left:120px;top:262px;width:790px;font-size:23px;line-height:1.35;color:var(--muted);--d:.2s"><b style="color:var(--ink)">Today:</b> {" &#8594; ".join(t[0].lower() + t[1:] if i else t for i, t in enumerate(today))}</p>
  <div class="abs" style="left:120px;top:372px;width:790px">{steps}</div>
  <div class="abs a" style="left:960px;top:262px;width:840px;height:548px;background:var(--soft);border-radius:16px;padding:22px 30px 26px;--d:.4s">
    <p style="font-size:21px;font-weight:600;color:var(--muted);margin-bottom:14px!important">What it looks like (illustrative)</p>
    {art}
  </div>
  <div class="abs" style="left:120px;top:842px;width:1680px;display:grid;grid-template-columns:720px 1fr 1fr;gap:40px">
    <div class="a" style="--d:.9s"><p class="big" style="font-size:50px;color:var(--blue)">{big}</p><p style="font-size:23px;line-height:1.3;color:var(--muted);margin-top:8px!important">{cap}</p></div>
    <div class="a" style="--d:1s;border-left:1px solid var(--line);padding-left:28px"><p style="font-size:21px;font-weight:600;color:var(--muted)">Track</p><p style="font-size:23px;line-height:1.3;margin-top:6px!important">{track}</p></div>
    <div class="a" style="--d:1.1s;border-left:1px solid var(--line);padding-left:28px"><p style="font-size:21px;font-weight:600;color:var(--muted)">Watch</p><p style="font-size:23px;line-height:1.3;margin-top:6px!important">{watch}</p></div>
  </div>
  <div class="foot"><span>{src}</span><span class="pn"></span></div>
</section>"""


ART = [
# research
"""<div style="display:grid;grid-template-columns:290px 1fr;gap:26px;height:440px">
 <div class="card"><div class="ln d" style="width:60%"></div><div class="ln" style="width:90%"></div><div class="ln" style="width:80%"></div>
  <div class="hl"><i>1</i></div><div class="ln" style="width:85%"></div>
  <div class="tb"></div>
  <div class="hl" style="width:70%"><i>2</i></div><div class="ln" style="width:75%"></div><div class="ln" style="width:88%"></div><div class="hl" style="width:55%"><i>3</i></div>
  <p class="cap">Annual report, 212 pages</p></div>
 <div class="card"><p class="ct">Result note, first draft</p>
  <p class="cl">Revenue up on higher volumes <span class="cite">1 &#183; p. 14</span></p>
  <p class="cl">Margin lower on input costs <span class="cite">2 &#183; p. 32</span></p>
  <p class="cl">Net debt reduced in the year <span class="cite">3 &#183; p. 41</span></p>
  <p class="cl flag">Guidance for next year: no source found. Analyst to check.</p></div>
</div>""",
# fund management
"""<div class="card" style="height:440px"><p class="ct">Morning brief &#183; 7:30 am</p>
 <div class="rw"><i style="background:var(--blue)"></i><p><b>Bank A</b> reported results after market hours<span>Summary ready, 3 sources</span></p></div>
 <div class="rw"><i style="background:var(--blue)"></i><p><b>Auto B</b> rating changed by one agency<span>Note attached</span></p></div>
 <div class="rw"><i style="background:var(--red);border-radius:3px"></i><p><b>Pharma C</b> is close to the single-issuer limit in the scheme<span style="color:#B8231A;font-weight:600">Flag for the fund manager</span></p></div>
 <div class="rw" style="border:0"><i style="background:#C9CEE3"></i><p><b>41 other holdings</b><span>No material news</span></p></div>
 <p class="cap" style="position:absolute;right:26px;bottom:20px">No order is created by the model</p></div>""",
# operations
"""<div style="display:grid;grid-template-columns:270px 1fr;gap:26px;height:440px">
 <div class="card" style="transform:rotate(-1.2deg)"><div class="ln d" style="width:70%"></div>
  <p class="fk">Name</p><div class="fb">A SHARMA</div><p class="fk">PAN</p><div class="fb">ABCDE1234F</div><p class="fk">Bank account</p><div class="fb">0012 3456 7890</div><p class="fk">Signature</p><div class="fb" style="border-style:dashed;height:44px"></div></div>
 <div class="card"><p class="ct">Extracted record</p>
  <div class="rw"><i style="background:var(--blue)"></i><p><b>Name</b><span>Matches KYC record</span></p></div>
  <div class="rw"><i style="background:var(--blue)"></i><p><b>PAN</b><span>Format valid</span></p></div>
  <div class="rw"><i style="background:var(--blue)"></i><p><b>Bank account</b><span>Matches bank proof</span></p></div>
  <div class="rw" style="border:0"><i style="background:var(--red);border-radius:3px"></i><p><b>Signature</b><span style="color:#B8231A;font-weight:600">Low confidence, sent to a person</span></p></div></div>
</div>""",
# compliance
"""<div style="position:relative;height:440px">
 <svg width="780" height="440" style="position:absolute;left:0;top:0" aria-hidden="true"><g fill="none" stroke-width="3"><path d="M330,112 C400,112 400,80 470,80" stroke="#2B4BFF"/><path d="M330,222 C400,222 400,222 470,222" stroke="#E8382D" stroke-dasharray="7 6"/><path d="M330,332 C400,332 400,364 470,364" stroke="#E8382D" stroke-dasharray="7 6"/></g></svg>
 <div class="card" style="position:absolute;left:0;top:0;width:330px;height:440px"><p class="ct">New SEBI circular</p>
  <div class="ob">Obligation 1<span>valuation of a security type</span></div><div class="ob">Obligation 2<span>wording in advertisements</span></div><div class="ob">Obligation 3<span>disclosure in scheme documents</span></div></div>
 <div class="pol" style="top:40px"><b>Valuation policy</b><span style="color:#1B33C9">Already covered</span></div>
 <div class="pol gap" style="top:182px"><b>Advertisement policy</b><span>Gap: clause to add</span></div>
 <div class="pol gap" style="top:324px"><b>Scheme documents</b><span>Gap: 12 documents to update</span></div>
</div>""",
# sales
"""<div style="height:440px;display:flex;flex-direction:column;gap:8px">
 <div class="bub">Exit load on Fund X if my client redeems after 8 months?<span>Distributor</span></div>
 <div class="bub me">Yes. The scheme document sets an exit load for redemptions within 12 months.<br><span class="cite" style="margin:8px 0 0">Scheme document, p. 23</span><span>Assistant, 4 seconds later</span></div>
 <div class="bub">Which fund suits a retired client?<span>Distributor</span></div>
 <div class="bub me" style="background:#fff;border:2px solid var(--red)">That needs a suitability check. I have passed it to your relationship manager.<span style="color:#B8231A">Handed to a person</span></div>
</div>""",
# marketing
"""<div style="height:440px">
 <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px">
  <div class="card" style="height:250px"><p class="cap" style="margin:0 0 10px">English</p><p class="lang">The fund added to banks this month.</p></div>
  <div class="card" style="height:250px"><p class="cap" style="margin:0 0 10px">Hindi</p><p class="lang">&#2347;&#2306;&#2337; &#2344;&#2375; &#2311;&#2360; &#2350;&#2361;&#2368;&#2344;&#2375; &#2348;&#2376;&#2306;&#2325;&#2379;&#2306; &#2350;&#2375;&#2306; &#2344;&#2367;&#2357;&#2375;&#2358; &#2348;&#2338;&#2364;&#2366;&#2351;&#2366;&#2404;</p></div>
  <div class="card" style="height:250px"><p class="cap" style="margin:0 0 10px">Tamil</p><p class="lang">&#2951;&#2984;&#3021;&#2980; &#2990;&#3006;&#2980;&#2990;&#3021; &#2984;&#3007;&#2980;&#3007; &#2997;&#2969;&#3021;&#2965;&#3007;&#2965;&#2995;&#3007;&#2994;&#3021; &#2990;&#3009;&#2980;&#2994;&#3008;&#2975;&#3021;&#2975;&#3016; &#2949;&#2980;&#3007;&#2965;&#2992;&#3007;&#2980;&#3021;&#2980;&#2980;&#3009;.</p></div>
 </div>
 <div class="card" style="margin-top:16px;height:174px"><p class="ct" style="margin-bottom:8px">Checks before release</p>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 30px;font-size:22px"><p><i class="tk"></i>Figures match portfolio data</p><p><i class="tk"></i>No promise of returns</p><p><i class="tk"></i>Risk disclaimer present</p><p><i class="tk" style="background:var(--ink)"></i>Compliance approval pending</p></div></div>
</div>""",
# investor service
"""<div class="card" style="height:440px"><p class="ct">Service inbox</p>
 <div class="rw"><i style="background:var(--yellow)"></i><p><b>Change my bank mandate</b><span><em>Service request</em><em>Hindi</em><em>Routine</em> Draft reply ready</span></p></div>
 <div class="rw"><i style="background:var(--yellow)"></i><p><b>SIP not debited this month</b><span><em>Query</em><em>English</em><em>Urgent</em> Draft reply ready</span></p></div>
 <div class="rw"><i style="background:var(--red);border-radius:3px"></i><p><b>Unhappy with redemption delay</b><span><em style="background:var(--redsoft);color:#B8231A">Complaint</em><em>Marathi</em> <strong style="color:#B8231A">Goes to a person</strong></span></p></div>
 <div class="rw" style="border:0"><i style="background:var(--yellow)"></i><p><b>Update nominee details</b><span><em>Service request</em><em>Tamil</em><em>Routine</em> Draft reply ready</span></p></div></div>""",
]


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
    S += fslide(4 + i, *f, ART[i])

S += f"""
<!-- 11 ENABLING -->
<section class="slide" data-sec="The functions">
  <h2 class="title">Finance, HR, legal and technology use the same three model types</h2>
  <div class="abs" style="left:120px;top:262px;width:1680px;display:grid;grid-template-columns:repeat(4,1fr);gap:44px">
    <div class="en a" style="--d:.2s"><svg width="96" height="96" viewBox="0 0 96 96" fill="none" stroke="#2B4BFF" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M22 10h52v76l-9-7-9 7-8-7-9 7-8-7-9 7z"/><path d="M34 32h28M34 48h28M34 64h16"/></svg>
      <h3>Finance</h3><p>Reads invoices and expense claims; drafts board and management reports</p><p style="margin-top:14px!important"><span class="tag t-vlm">VLM</span> <span class="tag t-llm">LLM</span></p><p class="who">The controller signs off every figure against the ledger</p></div>
    <div class="en a" style="--d:.3s"><svg width="96" height="96" viewBox="0 0 96 96" fill="none" stroke="#2B4BFF" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="48" cy="30" r="16"/><path d="M16 86c2-20 14-30 32-30s30 10 32 30"/></svg>
      <h3>HR</h3><p>Answers policy questions; drafts role descriptions and learning plans</p><p style="margin-top:14px!important"><span class="tag t-llm">LLM</span></p><p class="who">HR decides on hiring and appraisal; models are tested for bias</p></div>
    <div class="en a" style="--d:.4s"><svg width="96" height="96" viewBox="0 0 96 96" fill="none" stroke="#2B4BFF" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 8h40l18 18v62H20z"/><path d="M60 8v18h18"/><path d="M32 46h32M32 60h32"/><path d="M32 74c6-8 10 6 16-2s8 4 14 0"/></svg>
      <h3>Legal</h3><p>Compares distributor and vendor agreements against the standard terms</p><p style="margin-top:14px!important"><span class="tag t-llm">LLM</span></p><p class="who">Counsel reviews every flagged clause</p></div>
    <div class="en a" style="--d:.5s"><svg width="96" height="96" viewBox="0 0 96 96" fill="none" stroke="#2B4BFF" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M30 26L8 48l22 22M66 26l22 22-22 22M56 14L40 82"/></svg>
      <h3>Technology</h3><p>Writes and tests code; explains old systems; answers IT tickets</p><p style="margin-top:14px!important"><span class="tag t-llm">LLM</span> <span class="tag t-slm">SLM</span></p><p class="who">An engineer reviews each change before release</p></div>
  </div>
  <p class="abs a body" style="left:120px;top:870px;width:1680px;font-size:28px;--d:.9s"><b>These are usually the easiest places to begin,</b> because the output stays inside the company and a named person already signs it off.</p>
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

head = head.replace("family=Geist:wght@300..800", "family=Geist:wght@300..800&family=Noto+Sans+Devanagari:wght@500&family=Noto+Sans+Tamil:wght@500")
head = head.replace("</style>", build.css + css2 + "</style>")
head = head.replace("<title>Language is the ledger: a CXO briefing</title>", "<title>AI across business functions in asset management</title>")
tail = tail.replace("</body>", build.js + "</body>")
out = d.parent / "ai_across_business_functions.html"
out.write_text(head + S + tail, encoding="utf-8")
print("wrote", out)
