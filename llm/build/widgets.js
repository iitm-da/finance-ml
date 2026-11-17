/* ============================================================
   widgets for the LLM deck. Each one reads a JSON block the
   builder wrote and redraws from it; nothing is fetched.
   ============================================================ */
const MONO_='JetBrains Mono,monospace', BODY_='Public Sans,system-ui,sans-serif';
function seg_(id,cb){
  const box=document.getElementById(id); if(!box) return;
  box.querySelectorAll('button').forEach(b=>b.onclick=()=>{
    box.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b)); cb(b.dataset.v);});
}
function tx_(x,y,s,o){o=o||{};
  return '<text x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" font-family="'+(o.f||MONO_)+'" font-size="'+(o.z||10)+
    '" fill="'+(o.c||'#53666F')+'" text-anchor="'+(o.a||'start')+'" font-weight="'+(o.w||400)+'">'+s+'</text>';}
function esc_(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

/* ---------- 1 · retrieval explorer ---------- */
(function(){
  const el=document.getElementById('rx-data'); if(!el) return;
  const D=JSON.parse(el.textContent);
  const $=id=>document.getElementById(id);
  let k=0, rk='BM25';
  const GC={correct:'#0E7A55',abstained:'#53666F',wrong:'#B02F3B'};
  function draw(){
    const q=D.q[k];
    $('rx-q').innerHTML='<b>question '+(k+1)+' of '+D.q.length+'</b> &#183; '+q.kind+'<br/>'+esc_(q.q)+
      '<br/><span style="color:#53666F">'+(q.gold?'answer in '+q.gold+': '+esc_(q.ans.join(', ')):'no section answers this')+'</span>';
    $('rx-top').innerHTML=q.top[rk].map((t,i)=>{
      const hit=t[0]===q.gold;
      return '<div style="'+(hit?'color:#0E7A55;font-weight:700':'')+'">'+(i+1)+'. '+esc_(t[0]).padEnd(9,' ')+' '+esc_(t[2])+' / '+esc_(t[1])+(hit?'   &#8592; answer':'')+'</div>';
    }).join('');
    const box=(id,txt,g)=>{ const e=$(id); e.innerHTML=esc_(txt)+'<br/><b style="color:'+GC[g]+'">'+g+'</b>'; e.style.borderLeftColor=GC[g]; };
    box('rx-closed',q.closed,q.g[0]); box('rx-rag',q.rag,q.g[1]);
  }
  $('rx-prev').onclick=()=>{k=(k+D.q.length-1)%D.q.length;draw();};
  $('rx-next').onclick=()=>{k=(k+1)%D.q.length;draw();};
  seg_('rx-rank',v=>{rk=v;draw();});
  draw();
})();

/* ---------- 2 · LoRA rank ---------- */
(function(){
  const el=document.getElementById('lr-data'); if(!el) return;
  const D=JSON.parse(el.textContent), sv=document.getElementById('lr-svg'), out=document.getElementById('lr-out'), sl=document.getElementById('lr-r');
  const W=680,H=290,l=50,r=16,t=14,b=38;
  const lo=Math.min(D.head,...D.f1)-0.04, hi=Math.max(D.full,...D.f1)+0.04;
  const X=i=>l+(i+0.5)/D.ranks.length*(W-l-r), Y=v=>t+(hi-v)/(hi-lo)*(H-t-b);
  const fmt=n=>n.toLocaleString('en-GB');
  const size=n=>{const by=n*4; return by>=1e9?(by/1e9).toFixed(2)+' GB':by>=1e6?(by/1e6).toFixed(1)+' MB':(by/1e3).toFixed(0)+' KB';};
  function draw(){
    const k=+sl.value, rr=D.ranks[k];
    let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="#DCE4E8"/>';
    for(let v=Math.ceil(lo*20)/20; v<=hi; v+=0.05){
      s+='<line x1="'+l+'" x2="'+(W-r)+'" y1="'+Y(v).toFixed(1)+'" y2="'+Y(v).toFixed(1)+'" stroke="#DCE4E8"/>'+tx_(l-6,Y(v)+3.5,v.toFixed(2),{a:'end'});}
    [[D.full,'full fine-tuning','#0E7A55'],[D.head,'head only','#8FA0A9']].forEach(q=>{
      s+='<line x1="'+l+'" x2="'+(W-r)+'" y1="'+Y(q[0]).toFixed(1)+'" y2="'+Y(q[0]).toFixed(1)+'" stroke="'+q[2]+'" stroke-dasharray="5 4" stroke-width="1.6"/>'+
         tx_(W-r-6,Y(q[0])-5,q[1]+' '+q[0].toFixed(3),{a:'end',c:q[2],w:700,z:11});});
    s+='<polyline points="'+D.f1.map((v,i)=>X(i).toFixed(1)+','+Y(v).toFixed(1)).join(' ')+'" fill="none" stroke="#B02F3B" stroke-width="2.6"/>';
    D.f1.forEach((v,i)=>{
      s+='<circle cx="'+X(i).toFixed(1)+'" cy="'+Y(v).toFixed(1)+'" r="'+(i===k?8:4.5)+'" fill="#B02F3B"'+(i===k?' stroke="#14222B" stroke-width="2"':'')+'/>';
      s+=tx_(X(i),H-b+15,'r = '+D.ranks[i],{a:'middle',z:11,c:i===k?'#14222B':'#53666F',w:i===k?700:400});
      if(i===k) s+=tx_(X(i),Y(v)+24,v.toFixed(3),{a:'middle',z:12,c:'#14222B',w:700});
    });
    s+=tx_(l+6,t+13,'macro-F1',{});
    sv.innerHTML=s;
    let txt='rank '+rr+'   macro-F1 '+D.f1[k].toFixed(3)+' (measured)\n';
    D.models.forEach(m=>{
      const n=2*m.d*rr*m.mats;
      txt+='\n'+m.name+'\n  '+m.mats+' matrices x 2 x '+fmt(m.d)+' x '+rr+' = '+fmt(n)+' weights'+
           '\n  '+(100*n/m.total).toFixed(3)+'% of the model   adapter file '+size(n);
    });
    out.textContent=txt;
    document.getElementById('lr-rl').textContent=rr;
  }
  sl.oninput=draw; draw();
})();

/* ---------- chunk explorer ---------- */
(function(){
  const el=document.getElementById('cx-data'); if(!el) return;
  const D=JSON.parse(el.textContent), $=id=>document.getElementById(id);
  const BC={whole:'#0E7A55',part:'#B02F3B',none:'#C3CFD5'};
  function draw(n){
    const d=D[n];
    $('cx-v').innerHTML='<b>'+esc_(n)+'</b> &#183; answering sentence <b style="color:'+(d.ok?'#0E7A55':'#B02F3B')+'">'+d.v+
      '</b> &#183; served over all questions: '+d.served;
    $('cx-list').innerHTML=d.c.map((c,i)=>'<div class="out xs" style="white-space:normal;margin:0;border-left-color:'+BC[c[1]]+
      '"><span style="color:#53666F">chunk '+(i+1)+' &#183; '+c[2]+' words</span><br/>'+c[0]+'</div>').join('');
  }
  seg_('cx-seg',draw); draw(Object.keys(D)[0]);
})();

/* ---------- query expansion explorer ---------- */
(function(){
  const el=document.getElementById('qx-data'); if(!el) return;
  const D=JSON.parse(el.textContent), $=id=>document.getElementById(id);
  let k=0;
  const cell=p=>'<td style="text-align:right;font-weight:700;color:'+(p<=3?'#0E7A55':'#B02F3B')+'">'+p+'</td>';
  function draw(){
    const q=D.q[k];
    $('qx-q').innerHTML='<b>question '+(k+1)+' of '+D.q.length+'</b> &#183; '+q.kind+' &#183; answer in '+q.gold+'<br/>'+esc_(q.q);
    $('qx-pos').innerHTML='<table class="d sm free"><tr><th>expansion</th><th>BM25</th><th>dense</th></tr>'+
      D.m.map(m=>'<tr><td>'+m+'</td>'+cell(q.pos[m][0])+(m==='feedback terms'?'<td style="text-align:right;color:#8FA0A9">'+q.pos[m][1]+'</td>':cell(q.pos[m][1]))+'</tr>').join('')+'</table>';
    $('qx-prf').innerHTML='+ '+esc_(q.prf.join(', '));
    $('qx-rw').innerHTML=q.rw.length?q.rw.map(esc_).join('<br/>'):'(none parsed)';
    $('qx-hy').innerHTML=esc_(q.hyde);
    $('qx-syn').innerHTML=q.syn.length?'+ '+esc_(q.syn.join(', ')):'no term of the ontology is in this question';
  }
  $('qx-prev').onclick=()=>{k=(k+D.q.length-1)%D.q.length;draw();};
  $('qx-next').onclick=()=>{k=(k+1)%D.q.length;draw();};
  draw();
})();
