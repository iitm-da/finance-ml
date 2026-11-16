/* ============================================================
   widgets for the NLP deck. Each one reads a JSON block the
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

/* ---------- 1 · attention explorer ---------- */
(function(){
  const el=document.getElementById('ax-data'); if(!el) return;
  const D=JSON.parse(el.textContent), sv=document.getElementById('ax-svg'), out=document.getElementById('ax-out'),
        sl=document.getElementById('ax-layer'), sh=document.getElementById('ax-head');
  let sent=0, layer=D.layer, head=D.head, src=D.src;
  const W=1120, Y0=210;
  function draw(){
    const S=D.w[sent], n=S.tokens.length, cw=W/n;
    if(src>=n) src=1;
    const w=S.a[layer][head][src];
    let s='';
    const sx=cw*src+cw/2;
    for(let j=0;j<n;j++){
      if(j===src||w[j]<0.01) continue;
      const cx=cw*j+cw/2, h=Math.min(190,40+Math.abs(sx-cx)*0.36);
      s+='<path d="M'+sx.toFixed(1)+','+Y0+' C'+sx.toFixed(1)+','+(Y0-h).toFixed(1)+' '+cx.toFixed(1)+','+(Y0-h).toFixed(1)+' '+cx.toFixed(1)+','+Y0+
         '" fill="none" stroke="#B02F3B" stroke-width="'+Math.max(1,16*w[j]).toFixed(1)+'" stroke-opacity="'+(0.3+0.7*w[j]).toFixed(2)+'" stroke-linecap="round"/>';
    }
    if(w[src]>=0.01) s+='<circle cx="'+sx.toFixed(1)+'" cy="'+(Y0-14)+'" r="'+(6+10*w[src]).toFixed(1)+'" fill="none" stroke="#B02F3B" stroke-width="'+Math.max(1,10*w[src]).toFixed(1)+'" stroke-opacity=".6"/>';
    for(let j=0;j<n;j++){
      const cx=cw*j+cw/2, on=j===src;
      s+='<g class="ax-t" data-j="'+j+'" style="cursor:pointer"><rect x="'+(cx-cw/2+4).toFixed(1)+'" y="'+Y0+'" width="'+(cw-8).toFixed(1)+'" height="32" fill="'+(on?'#14222B':'#fff')+
         '" stroke="#C3CFD5"/>'+tx_(cx,Y0+21,esc_(S.tokens[j]),{a:'middle',z:n>13?11:13,c:on?'#fff':'#14222B',w:600})+'</g>';
      s+='<rect x="'+(cx-18).toFixed(1)+'" y="'+(Y0+44)+'" width="'+(36*w[j]).toFixed(1)+'" height="7" fill="#B02F3B"/>'+
         '<rect x="'+(cx-18).toFixed(1)+'" y="'+(Y0+44)+'" width="36" height="7" fill="none" stroke="#DCE4E8"/>';
      s+=tx_(cx,Y0+68,w[j].toFixed(2),{a:'middle',z:11,c:w[j]>=0.2?'#B02F3B':'#53666F',w:w[j]>=0.2?700:400});
    }
    sv.innerHTML=s;
    sv.querySelectorAll('.ax-t').forEach(g=>g.onclick=()=>{src=+g.dataset.j;draw();});
    const top=w.map((v,j)=>[v,j]).sort((a,b)=>b[0]-a[0]).slice(0,3);
    out.textContent='layer '+(layer+1)+', head '+(head+1)+'   from "'+S.tokens[src]+'"\n'+
      top.map(t=>'  '+t[0].toFixed(2)+'  '+S.tokens[t[1]]).join('\n');
    document.getElementById('ax-ll').textContent=layer+1; document.getElementById('ax-hl').textContent=head+1;
  }
  seg_('ax-sent',v=>{sent=+v;draw();});
  sl.oninput=()=>{layer=+sl.value-1;draw();};
  sh.oninput=()=>{head=+sh.value-1;draw();};
  layer=+sl.value-1; head=+sh.value-1; draw();
})();

/* ---------- 2 · sentiment row browser ---------- */
(function(){
  const el=document.getElementById('sb-data'); if(!el) return;
  const D=JSON.parse(el.textContent), body=document.getElementById('sb-body'), out=document.getElementById('sb-out');
  let filt='all', page=0; const PER=8;
  const COL=['#B02F3B','#0E7A55','#53666F'];
  function rows(){
    return D.rows.filter(r=>{
      if(filt==='all') return true;
      if(filt==='ft') return r.p[D.ft]!==r.y;
      if(filt==='gen') return r.p[D.gen]!==r.y && r.p[D.ft]===r.y;
      return D.models.every(m=>r.p[m]!==r.y);
    });
  }
  function lab(k,y){ return '<td style="font-family:'+MONO_+';font-size:11px;font-weight:'+(k===y?700:400)+';color:'+(k===y?COL[k]:'#fff')+
      ';background:'+(k===y?'transparent':COL[k])+';text-align:center;white-space:nowrap">'+D.lab[k]+'</td>'; }
  function draw(){
    const R=rows(), np=Math.max(1,Math.ceil(R.length/PER)); if(page>=np) page=0;
    let h='<tr><th>headline</th><th>label</th>'+D.short.map(s=>'<th>'+s+'</th>').join('')+'</tr>';
    R.slice(page*PER,page*PER+PER).forEach(r=>{
      h+='<tr><td style="font-size:12.5px">'+esc_(r.t)+'</td><td style="font-family:'+MONO_+';font-size:11px;font-weight:700;color:'+COL[r.y]+'">'+D.lab[r.y]+'</td>'+
         D.models.map(m=>lab(r.p[m],r.y)).join('')+'</tr>';
    });
    body.innerHTML=h;
    out.textContent=R.length+' of '+D.rows.length+' sampled test rows   page '+(page+1)+' of '+np;
  }
  seg_('sb-filter',v=>{filt=v;page=0;draw();});
  document.getElementById('sb-next').onclick=()=>{page++;draw();};
  draw();
})();

/* ---------- 3 · confidence threshold ---------- */
(function(){
  const el=document.getElementById('ct-data'); if(!el) return;
  const D=JSON.parse(el.textContent), sv=document.getElementById('ct-svg'), out=document.getElementById('ct-out'),
        sl=document.getElementById('ct-thr');
  const W=700,H=300,l=48,r=14,t=14,b=40, NB=20;
  const hist=[[],[]]; for(let i=0;i<NB;i++){hist[0].push(0);hist[1].push(0);}
  D.sc.forEach(p=>{ const k=Math.min(NB-1,Math.floor(p[0]*NB)); hist[p[1]][k]++; });
  const mx=Math.max(...hist[0].map((v,i)=>v+hist[1][i]));
  function draw(){
    const thr=+sl.value;
    const X=v=>l+v*(W-l-r), Y=v=>t+(1-v/mx)*(H-t-b);
    let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="#DCE4E8"/>';
    s+='<rect x="'+l+'" y="'+t+'" width="'+(X(thr)-l).toFixed(1)+'" height="'+(H-t-b)+'" fill="#FFF6E0"/>';
    for(let i=0;i<NB;i++){
      const x=X(i/NB)+1, w=(W-l-r)/NB-2, g=hist[1][i], bd=hist[0][i];
      s+='<rect x="'+x.toFixed(1)+'" y="'+Y(g).toFixed(1)+'" width="'+w.toFixed(1)+'" height="'+(Y(0)-Y(g)).toFixed(1)+'" fill="#0E7A55"/>';
      s+='<rect x="'+x.toFixed(1)+'" y="'+Y(g+bd).toFixed(1)+'" width="'+w.toFixed(1)+'" height="'+(Y(g)-Y(g+bd)).toFixed(1)+'" fill="#B02F3B"/>';
    }
    for(let v=0;v<=1.001;v+=0.2) s+=tx_(X(v),H-b+14,v.toFixed(1),{a:'middle'});
    s+=tx_(W-r,H-6,'score the model gives its answer',{a:'end'});
    s+=tx_(l+6,t+13,'answers',{});
    s+='<line x1="'+X(thr).toFixed(1)+'" x2="'+X(thr).toFixed(1)+'" y1="'+t+'" y2="'+(H-b)+'" stroke="#14222B" stroke-width="2"/>';
    s+=tx_(Math.min(X(thr)+6,W-150),t+28,'accept above '+thr.toFixed(2),{c:'#14222B',w:700,z:11});
    s+=tx_(Math.max(l+6,X(thr)-6),t+44,thr>0.12?'to a person':'',{c:'#8F5D0A',w:700,z:11,a:'end'});
    sv.innerHTML=s;
    const acc=D.sc.filter(p=>p[0]>=thr), bad=acc.filter(p=>p[1]===0).length, n=D.sc.length;
    const rev=n-acc.length, badAll=D.sc.filter(p=>p[1]===0).length;
    out.textContent='threshold            '+thr.toFixed(2)+
      '\naccepted unchecked   '+acc.length+' of '+n+'  ('+(100*acc.length/n).toFixed(1)+'%)'+
      '\nwrong among accepted '+bad+'  ('+(acc.length?(100*bad/acc.length).toFixed(2):'0.00')+'%)'+
      '\nsent to a person     '+rev+'  ('+(100*rev/n).toFixed(1)+'%)'+
      '\nwrong answers caught '+(badAll-bad)+' of '+badAll;
    document.getElementById('ct-tl').textContent=thr.toFixed(2);
  }
  sl.oninput=draw; draw();
})();
