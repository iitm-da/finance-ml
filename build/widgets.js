/* ============================================================
   widgets for the deck. Each one reads a JSON block the
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

/* ---------- 1 · forecast explorer ---------- */
(function(){
  const el=document.getElementById('fx-data'); if(!el) return;
  const D=JSON.parse(el.textContent), sv=document.getElementById('fx-svg'), out=document.getElementById('fx-out'),
        sl=document.getElementById('fx-origin');
  let model='arima', k=0;
  const W=1120,H=290,l=52,r=16,t=14,b=44, BACK=28;
  function draw(){
    const o=D.origins[k], a0=o-BACK, a1=o+D.H;
    const act=D.y.slice(a0,a1), fc=D.fc[model][k], q=D.q[k];
    let lo=Math.min(...act,...fc), hi=Math.max(...act,...fc);
    if(model==='tft'){ lo=Math.min(lo,...q.map(v=>v[0])); hi=Math.max(hi,...q.map(v=>v[2])); }
    lo=Math.floor((lo-8)/20)*20; hi=Math.ceil((hi+8)/20)*20;
    const X=i=>l+(i-a0)/(a1-a0-1)*(W-l-r), Y=v=>t+(hi-v)/(hi-lo)*(H-t-b);
    let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="#DCE4E8"/>';
    s+='<rect x="'+X(o-0.5).toFixed(1)+'" y="'+t+'" width="'+(X(a1-1)-X(o-0.5)).toFixed(1)+'" height="'+(H-t-b)+'" fill="#F3F7F9"/>';
    for(let v=lo;v<=hi;v+=(hi-lo>200?50:20)){
      s+='<line x1="'+l+'" x2="'+(W-r)+'" y1="'+Y(v).toFixed(1)+'" y2="'+Y(v).toFixed(1)+'" stroke="#DCE4E8"/>'+tx_(l-6,Y(v)+3.5,v,{a:'end'});}
    for(let i=a0;i<a1;i++){
      const d=D.dates[i], dow=D.dow[i];
      if(dow===0) s+=tx_(X(i),H-b+14,d.slice(5),{a:'middle',z:9.5});
      const tag=D.hol[i]?'H':D.pay[i]?'P':D.dd[i]?'D':'';
      if(tag) s+='<rect x="'+(X(i)-7).toFixed(1)+'" y="'+(H-b+20)+'" width="14" height="14" fill="'+(tag==='H'?'#B02F3B':tag==='P'?'#0E7A55':'#8F5D0A')+'"/>'+
                 tx_(X(i),H-b+30.5,tag,{a:'middle',c:'#fff',w:700,z:9.5});
    }
    if(model==='tft'){
      let p='';
      q.forEach((v,h)=>{p+=(h?'L':'M')+X(o+h).toFixed(1)+','+Y(v[2]).toFixed(1);});
      for(let h=D.H-1;h>=0;h--) p+='L'+X(o+h).toFixed(1)+','+Y(q[h][0]).toFixed(1);
      s+='<path d="'+p+'z" fill="'+D.col.tft+'" fill-opacity=".16"/>';
    }
    const line=(xs,ys,c,w,dash)=>'<polyline points="'+ys.map((v,i)=>X(xs+i).toFixed(1)+','+Y(v).toFixed(1)).join(' ')+
      '" fill="none" stroke="'+c+'" stroke-width="'+w+'" stroke-linejoin="round"'+(dash?' stroke-dasharray="'+dash+'"':'')+'/>';
    s+=line(a0,act.slice(0,BACK),'#8FA0A9',1.8);
    s+=line(o-1,act.slice(BACK-1),'#14222B',2.4);
    s+=line(o,fc,D.col[model],2.6);
    fc.forEach((v,h)=>{s+='<circle cx="'+X(o+h).toFixed(1)+'" cy="'+Y(v).toFixed(1)+'" r="2.6" fill="'+D.col[model]+'"/>';});
    s+='<line x1="'+X(o-0.5).toFixed(1)+'" x2="'+X(o-0.5).toFixed(1)+'" y1="'+t+'" y2="'+(H-b)+'" stroke="#14222B" stroke-dasharray="3 3"/>';
    s+=tx_(X(o-0.5)+6,t+13,'forecast made here',{c:'#14222B',w:700});
    s+=tx_(l+6,t+13,'GBP million',{});
    sv.innerHTML=s;
    const mae=fc.reduce((a,v,h)=>a+Math.abs(v-act[BACK+h]),0)/D.H;
    let extra='';
    if(model==='tft'){ const inb=q.filter((v,h)=>act[BACK+h]>=v[0]&&act[BACK+h]<=v[2]).length;
      extra='\ninside P10-P90   '+inb+' of '+D.H+' days'; }
    out.textContent=D.name[model]+'\norigin '+(k+1)+' of '+D.origins.length+'   '+D.dates[o]+
      '\nMAE this window  '+mae.toFixed(1)+'\nMASE, all origins and segments  '+D.mase[model].toFixed(3)+extra;
    out.style.borderLeftColor=D.col[model];
  }
  seg_('fx-model',v=>{model=v;draw();});
  sl.oninput=()=>{k=+sl.value;draw();};
  k=+sl.value; draw();
})();

/* ---------- 2 · pinball loss ---------- */
(function(){
  const el=document.getElementById('pb-data'); if(!el) return;
  const D=JSON.parse(el.textContent), sv=document.getElementById('pb-svg'), out=document.getElementById('pb-out'),
        sl=document.getElementById('pb-q'), ql=document.getElementById('pb-ql');
  const ys=D.y.slice().sort((a,b)=>a-b), n=ys.length, lo=D.lo, hi=D.hi;
  const W=700,H=400,l=50,r=14,t=14,b=64;
  const loss=(c,q)=>ys.reduce((a,y)=>a+(y>=c?q*(y-c):(1-q)*(c-y)),0)/n;
  function draw(){
    const q=+sl.value; ql.textContent=q.toFixed(2);
    const cs=[]; for(let c=lo;c<=hi;c+=(hi-lo)/160) cs.push(c);
    const ls=cs.map(c=>loss(c,q)), lmax=D.lmax;
    const X=c=>l+(c-lo)/(hi-lo)*(W-l-r), Y=v=>t+(lmax-Math.min(v,lmax))/lmax*(H-t-b);
    let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="#DCE4E8"/>';
    for(let v=0;v<=lmax;v+=10) s+='<line x1="'+l+'" x2="'+(W-r)+'" y1="'+Y(v).toFixed(1)+'" y2="'+Y(v).toFixed(1)+'" stroke="#DCE4E8"/>'+tx_(l-6,Y(v)+3.5,v,{a:'end'});
    for(let v=Math.ceil(lo/20)*20;v<=hi;v+=20) s+=tx_(X(v),H-b+14,v,{a:'middle'});
    s+='<polyline points="'+cs.map((c,i)=>X(c).toFixed(1)+','+Y(ls[i]).toFixed(1)).join(' ')+'" fill="none" stroke="#125F9E" stroke-width="2.6"/>';
    let bi=0; ls.forEach((v,i)=>{if(v<ls[bi])bi=i;});
    const cm=cs[bi], below=ys.filter(y=>y<=cm).length;
    s+='<line x1="'+X(cm).toFixed(1)+'" x2="'+X(cm).toFixed(1)+'" y1="'+t+'" y2="'+(H-b+34)+'" stroke="#B02F3B" stroke-width="1.6"/>';
    s+='<circle cx="'+X(cm).toFixed(1)+'" cy="'+Y(ls[bi]).toFixed(1)+'" r="5" fill="#B02F3B"/>';
    ys.forEach(y=>{s+='<circle cx="'+X(y).toFixed(1)+'" cy="'+(H-b+30)+'" r="4" fill="'+(y<=cm?'#8FA0A9':'#14222B')+'" fill-opacity=".8"/>';});
    s+=tx_(l+6,t+13,'mean pinball loss of forecasting the value on the x axis');
    s+=tx_(W-r,H-8,D.xlab,{a:'end'});
    sv.innerHTML=s;
    out.textContent='q = '+q.toFixed(2)+'\nloss is lowest at  '+cm.toFixed(1)+'\nvalues at or below it  '+below+' of '+n+'  ('+(below/n*100).toFixed(0)+'%)'+
      '\nunder-forecast costs  '+q.toFixed(2)+' per unit\nover-forecast costs   '+(1-q).toFixed(2)+' per unit';
  }
  sl.oninput=draw; draw();
})();

/* ---------- 3 · GAN training scrubber ---------- */
(function(){
  const el=document.getElementById('gs-data'); if(!el) return;
  const D=JSON.parse(el.textContent), sv=document.getElementById('gs-svg'), out=document.getElementById('gs-out'),
        sl=document.getElementById('gs-ep'), play=document.getElementById('gs-play'), ws=document.getElementById('gs-w');
  const W=640,H=420,l=46,r=12,t=12,b=34;
  const X=a=>l+(Math.log10(Math.max(a,0.5))-Math.log10(0.5))/(Math.log10(3000)-Math.log10(0.5))*(W-l-r);
  const Y=h=>t+(23.5-h)/24*(H-t-b);
  const jit=D.jit;
  let timer=null;
  function pts(P,col,op){let s='';
    P.amount.forEach((a,i)=>{s+='<circle cx="'+X(a).toFixed(1)+'" cy="'+(Y(P.hour[i])+jit[i%jit.length]).toFixed(1)+'" r="2.6" fill="'+col+'" fill-opacity="'+op+'"/>';});
    return s;}
  function draw(){
    const k=+sl.value, S=D.snaps[k];
    let s='<rect x="'+l+'" y="'+t+'" width="'+(W-l-r)+'" height="'+(H-t-b)+'" fill="#fff" stroke="#DCE4E8"/>';
    [0,6,12,18,23].forEach(h=>{s+='<line x1="'+l+'" x2="'+(W-r)+'" y1="'+Y(h).toFixed(1)+'" y2="'+Y(h).toFixed(1)+'" stroke="#DCE4E8"/>'+tx_(l-6,Y(h)+3.5,h+':00',{a:'end',z:9.5});});
    [1,10,100,1000].forEach(a=>{s+=tx_(X(a),H-b+14,'£'+a,{a:'middle'});});
    s+=pts(D.real,'#8FA0A9',.55)+pts(S,'#B02F3B',.6);
    s+=tx_(W-r-6,t+14,'grey: 600 real rows',{a:'end',c:'#53666F',w:700});
    s+=tx_(W-r-6,t+28,'red: 600 synthetic rows',{a:'end',c:'#B02F3B',w:700});
    s+=tx_(W-r,H-6,'amount, log scale',{a:'end'}); s+=tx_(l+6,t+13,'hour of day');
    sv.innerHTML=s;
    const ks=S.ks, mean=Object.values(ks).reduce((a,v)=>a+v,0)/Object.keys(ks).length;
    out.textContent='epoch '+S.epoch+' of '+D.epochs+'\nKS  amount  '+ks.amount.toFixed(3)+'\nKS  hour    '+ks.hour.toFixed(3)+
      '\nKS  mean of 6 columns  '+mean.toFixed(3)+'\nheld-out real rows score  '+D.floor.toFixed(3);
    // critic's distance estimate
    const w=D.wdist, WW=340,HH=120,ll=40,rr=8,tt=10,bb=22, wmax=Math.max(...w), wmin=Math.min(0,...w);
    const XX=e=>ll+(e-1)/(w.length-1)*(WW-ll-rr), YY=v=>tt+(wmax-v)/(wmax-wmin)*(HH-tt-bb);
    let g='<rect x="'+ll+'" y="'+tt+'" width="'+(WW-ll-rr)+'" height="'+(HH-tt-bb)+'" fill="#fff" stroke="#DCE4E8"/>';
    g+='<polyline points="'+w.map((v,i)=>XX(i+1).toFixed(1)+','+YY(v).toFixed(1)).join(' ')+'" fill="none" stroke="#125F9E" stroke-width="1.8"/>';
    if(S.epoch>0) g+='<line x1="'+XX(S.epoch).toFixed(1)+'" x2="'+XX(S.epoch).toFixed(1)+'" y1="'+tt+'" y2="'+(HH-bb)+'" stroke="#B02F3B" stroke-width="1.6"/>';
    g+=tx_(ll-5,YY(wmax)+8,wmax.toFixed(1),{a:'end',z:9.5})+tx_(ll-5,YY(0)+3,'0',{a:'end',z:9.5});
    g+=tx_(ll,HH-6,'epoch 1',{z:9.5})+tx_(WW-rr,HH-6,'epoch '+w.length,{a:'end',z:9.5});
    ws.innerHTML=g;
  }
  function stop(){ if(timer){clearInterval(timer);timer=null;} play.textContent='play'; }
  play.onclick=()=>{
    if(timer){stop();return;}
    if(+sl.value>=D.snaps.length-1) sl.value=0;
    play.textContent='pause';
    timer=setInterval(()=>{ if(+sl.value>=D.snaps.length-1){stop();return;} sl.value=+sl.value+1; draw(); },750);
    draw();
  };
  sl.oninput=()=>{stop();draw();};
  const host=sv.closest('.slide'); if(host) host.addEventListener('hidden',stop);
  addEventListener('beforeunload',stop);
  draw();
})();

/* ---------- 4 · alert budget ---------- */
(function(){
  const el=document.getElementById('ab-data'); if(!el) return;
  const D=JSON.parse(el.textContent), sl=document.getElementById('ab-k'), kl=document.getElementById('ab-kl'),
        sv=document.getElementById('ab-svg'), out=document.getElementById('ab-out');
  let model='gb';
  const W=700,H=300,l=190,r=120;
  function bar(y,lab,frac,txt,col){
    const w=(W-l-r)*Math.max(0,Math.min(1,frac));
    return tx_(l-10,y+22,lab,{a:'end',z:13,c:'#14222B',w:500})+
      '<rect x="'+l+'" y="'+y+'" width="'+(W-l-r)+'" height="34" fill="#EEF2F4"/>'+
      '<rect x="'+l+'" y="'+y+'" width="'+w.toFixed(1)+'" height="34" fill="'+col+'"/>'+
      tx_(l+w+8,y+22,txt,{z:13,c:'#14222B',w:700});
  }
  function draw(){
    const per=+sl.value, k=per*D.days, C=D.curve[model];
    kl.textContent=per;
    const tp=C.tp[k-1], gbp=C.gbp[k-1], prec=tp/k, rec=tp/D.fraud;
    let s='';
    s+=bar(10,'fraud caught',rec,tp+' of '+D.fraud,'#0E7A55');
    s+=bar(64,'fraud missed',1-rec,(D.fraud-tp)+'','#B02F3B');
    s+=bar(118,'alerts that are fraud',prec,(prec*100).toFixed(1)+'%','#125F9E');
    s+=bar(172,'fraud value caught',gbp/D.gbp,'£'+gbp.toLocaleString('en-GB'),'#0E7A55');
    s+=tx_(l,236,'false alerts reviewed a day: '+((k-tp)/D.days).toFixed(1),{z:12.5,c:'#14222B'});
    s+=tx_(l,258,'one fraud found in every '+(tp?(k/tp).toFixed(1):'-')+' alerts',{z:12.5,c:'#14222B'});
    sv.innerHTML=s;
    out.textContent=D.name[model]+'\n'+per+' alerts a day x '+D.days+' days = '+k.toLocaleString('en-GB')+' alerts\nrecall     '+(rec*100).toFixed(1)+'%\nprecision  '+(prec*100).toFixed(1)+'%';
  }
  seg_('ab-model',v=>{model=v;draw();});
  sl.oninput=draw; draw();
})();

/* ---------- margin, shortfall and idle cash ---------- */
(function(){
  const el=document.getElementById('cs-data'); if(!el) return;
  const D=JSON.parse(el.textContent), sv=document.getElementById('cs-svg'), out=document.getElementById('cs-out'),
        sl=document.getElementById('cs-margin'), ml=document.getElementById('cs-ml');
  let model='naive', r=9;
  const W=1120,H=280,l=52,t=14,b=30, xa=760, xb=830, xr=1104;
  const all=[].concat(...Object.values(D.err)), lo=Math.floor(Math.min(...all)/20)*20, hi=Math.ceil(Math.max(...all,90)/20)*20;
  const Y=v=>t+(hi-v)/(hi-lo)*(H-t-b);
  function stats(e,c){
    let sh=0,idle=0,n=0;
    e.forEach(v=>{ if(v>c){sh+=v-c;n++;} else idle+=c-v; });
    return {n:n, sh:sh/e.length, idle:idle/e.length};
  }
  function draw(){
    const e=D.err[model], c=+sl.value, n=e.length, st=stats(e,c), col=D.col[model];
    ml.textContent=c.toFixed(1);
    const X=i=>l+(i+0.5)/n*(xa-l), bw=(xa-l)/n-0.8;
    let s='<rect x="'+l+'" y="'+t+'" width="'+(xa-l)+'" height="'+(H-t-b)+'" fill="#fff" stroke="#DCE4E8"/>';
    for(let v=lo;v<=hi;v+=20){ if(v%40) continue;
      s+='<line x1="'+l+'" x2="'+xa+'" y1="'+Y(v).toFixed(1)+'" y2="'+Y(v).toFixed(1)+'" stroke="#DCE4E8"/>'+tx_(l-6,Y(v)+3.5,v,{a:'end'});}
    for(let k=1;k<13;k++){ const x=l+k*14/n*(xa-l); s+='<line x1="'+x.toFixed(1)+'" x2="'+x.toFixed(1)+'" y1="'+t+'" y2="'+(H-b)+'" stroke="#EEF2F4"/>'; }
    for(let k=0;k<13;k++) s+=tx_(l+(k*14+7)/n*(xa-l),H-b+13,k+1,{a:'middle',z:9.5});
    s+=tx_(xa,H-4,'test window',{a:'end',z:9.5})+tx_(l+6,t+13,'actual minus forecast, GBP million',{});
    e.forEach((v,i)=>{
      const y0=Y(Math.max(v,0)), y1=Y(Math.min(v,0));
      s+='<rect x="'+(X(i)-bw/2).toFixed(1)+'" y="'+y0.toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+Math.max(0.6,y1-y0).toFixed(1)+'" fill="'+col+'" fill-opacity=".55"/>';
      if(v>c) s+='<rect x="'+(X(i)-bw/2).toFixed(1)+'" y="'+Y(v).toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+Math.max(1.5,Y(c)-Y(v)).toFixed(1)+'" fill="#B02F3B"/>';
    });
    s+='<line x1="'+l+'" x2="'+xa+'" y1="'+Y(0).toFixed(1)+'" y2="'+Y(0).toFixed(1)+'" stroke="#53666F"/>';
    s+='<line x1="'+l+'" x2="'+xa+'" y1="'+Y(c).toFixed(1)+'" y2="'+Y(c).toFixed(1)+'" stroke="#14222B" stroke-width="2.2"/>';
    s+='<rect x="'+(xa-128)+'" y="'+(t+5)+'" width="122" height="16" fill="#fff" fill-opacity=".9"/>'+tx_(xa-10,t+17,'line: forecast + '+c.toFixed(1),{a:'end',c:'#14222B',w:700,z:10.5});
    /* cost curve */
    const cs=[]; let best=0, bc=Infinity, cmax=0;
    for(let m=0;m<=90;m+=0.5){ const q=stats(e,m), v=q.idle+r*q.sh; cs.push([m,v]); if(v<bc){bc=v;best=m;} if(v>cmax) cmax=v; }
    const top=Math.ceil(cmax/20)*20, CX=m=>xb+m/90*(xr-xb), CY=v=>t+(top-v)/top*(H-t-b);
    s+='<rect x="'+xb+'" y="'+t+'" width="'+(xr-xb)+'" height="'+(H-t-b)+'" fill="#fff" stroke="#DCE4E8"/>';
    [0,top/2,top].forEach(v=>{s+=tx_(xb-6,CY(v)+3.5,v.toFixed(0),{a:'end'});});
    [0,30,60,90].forEach(m=>{s+=tx_(CX(m),H-b+13,m,{a:'middle',z:9.5});});
    s+=tx_(xr,H-4,'margin',{a:'end',z:9.5})+tx_(xb+6,t+13,'cost per day',{});
    s+='<polyline points="'+cs.map(p=>CX(p[0]).toFixed(1)+','+CY(p[1]).toFixed(1)).join(' ')+'" fill="none" stroke="'+col+'" stroke-width="2.4"/>';
    s+='<line x1="'+CX(best).toFixed(1)+'" x2="'+CX(best).toFixed(1)+'" y1="'+CY(bc).toFixed(1)+'" y2="'+(H-b)+'" stroke="#0E7A55" stroke-dasharray="3 3"/>';
    s+='<circle cx="'+CX(best).toFixed(1)+'" cy="'+CY(bc).toFixed(1)+'" r="4" fill="#0E7A55"/>';
    s+=tx_(xr-8,t+13,'green: cheapest margin '+best.toFixed(1),{a:'end',c:'#0E7A55',w:700,z:10.5})+tx_(xr-8,t+27,'black: the margin chosen',{a:'end',c:'#14222B',w:700,z:10.5});
    const cur=st.idle+r*st.sh;
    s+='<circle cx="'+CX(c).toFixed(1)+'" cy="'+CY(cur).toFixed(1)+'" r="5.5" fill="#14222B"/>';
    sv.innerHTML=s;
    const bs=stats(e,best);
    out.textContent=D.name[model]+'   margin '+c.toFixed(1)+
      '\ndays short            '+st.n+' of '+n+'  ('+(st.n/n*100).toFixed(1)+'%)'+
      '\nidle cash per day     '+st.idle.toFixed(1)+
      '\nshortfall per day     '+st.sh.toFixed(2)+
      '\ncost per day at '+r+'x    '+cur.toFixed(1)+
      '\ncheapest margin at '+r+'x '+best.toFixed(1)+'  ('+bs.n+' days short, cost '+bc.toFixed(1)+')';
    out.style.borderLeftColor=col;
  }
  seg_('cs-model',v=>{model=v;draw();});
  seg_('cs-ratio',v=>{r=+v;draw();});
  sl.oninput=draw;
  draw();
})();
