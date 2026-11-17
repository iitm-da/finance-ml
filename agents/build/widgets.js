/* ============================================================
   widgets for the agents deck. Each one reads a JSON block the
   builder wrote and redraws from it; nothing is fetched.
   ============================================================ */
function seg_(id,cb){
  const box=document.getElementById(id); if(!box) return;
  box.querySelectorAll('button').forEach(b=>b.onclick=()=>{
    box.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b)); cb(b.dataset.v);});
}
function esc_(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

/* ---------- 1 · trace explorer: one task, one model call at a time ---------- */
(function(){
  const el=document.getElementById('tx-data'); if(!el) return;
  const D=JSON.parse(el.textContent), $=id=>document.getElementById(id);
  let k=0, shown=0;
  const GC={correct:'#0E7A55',wrong:'#B02F3B','no answer':'#53666F'};
  function rows(t){
    const r=[];
    t.steps.forEach((calls,i)=>{
      calls.forEach(c=>{
        const a=typeof c.args==='string'?c.args:Object.values(c.args).join(', ');
        r.push('<tr><td class="r">'+(i+1)+'</td><td><b style="color:#125F9E">'+esc_(c.name||'?')+'</b></td><td style="font-family:var(--mono);font-size:11.5px">'+esc_(a)+
          '</td><td style="font-size:11.5px;color:'+(c.err?'#B02F3B':'#14222B')+'">'+esc_(c.obs)+'</td></tr>');
      });
    });
    return r;
  }
  function draw(){
    const t=D[k], r=rows(t), n=t.steps.length+1;
    $('tx-q').innerHTML='<b>task '+(k+1)+' of '+D.length+'</b> &#183; level '+t.level+'<br/>'+esc_(t.q);
    let h='<table class="d sm free"><tr><th class="r">call</th><th>the model asks for</th><th>with</th><th>the code returns</th></tr>';
    let seen=0;
    t.steps.forEach((calls,i)=>{ if(i<shown){ h+=r.slice(seen,seen+calls.length).join(''); } seen+=calls.length; });
    if(shown>=n){
      h+='<tr class="best"><td class="r">'+n+'</td><td colspan="2"><b>answers</b></td><td style="font-family:var(--mono)">'+
        esc_(t.final===null?'(stopped at the step limit)':t.final)+'</td></tr>';
    }
    $('tx-tab').innerHTML=h+'</table>';
    $('tx-v').innerHTML=shown>=n?('expected <b>'+t.answer+'</b> = '+esc_(t.expr)+' &#183; <b style="color:'+GC[t.grade]+'">'+t.grade+'</b>'+
      (t.grade==='correct'?'':' &#183; '+esc_(t.cause))+' &#183; '+n+' model calls'):('model call '+shown+' of '+n+' shown');
    $('tx-step').textContent=shown>=n?'restart':'next call ›';
  }
  $('tx-prev').onclick=()=>{k=(k+D.length-1)%D.length;shown=0;draw();};
  $('tx-next').onclick=()=>{k=(k+1)%D.length;shown=0;draw();};
  $('tx-step').onclick=()=>{const n=D[k].steps.length+1;shown=shown>=n?0:shown+1;draw();};
  draw();
})();

/* ---------- 2 · cost of a loop: arithmetic on two sliders ---------- */
(function(){
  const sv=document.getElementById('cl-fig'); if(!sv) return;
  const $=id=>document.getElementById(id), W=560,H=250,l=46,r=14,t=14,b=30,pw=W-l-r,ph=H-t-b,N=20;
  const X=n=>l+(n-1)/(N-1)*pw, Y=v=>t+ph-v*ph;
  const tx=(x,y,s,o)=>{o=o||{};return '<text x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" font-family="JetBrains Mono,monospace" font-size="'+(o.z||10)+
    '" fill="'+(o.c||'#53666F')+'" text-anchor="'+(o.a||'start')+'" font-weight="'+(o.w||400)+'">'+s+'</text>';};
  function draw(){
    const p=+$('cl-p').value/100, n=+$('cl-n').value, base=+$('cl-b').value, add=+$('cl-a').value;
    let s='<rect x="'+l+'" y="'+t+'" width="'+pw+'" height="'+ph+'" fill="#fff" stroke="#DCE4E8"/>';
    [0,.25,.5,.75,1].forEach(v=>{s+='<line x1="'+l+'" y1="'+Y(v)+'" x2="'+(l+pw)+'" y2="'+Y(v)+'" stroke="#DCE4E8"/>'+tx(l-6,Y(v)+3.5,(100*v)+'%',{a:'end'});});
    [1,5,10,15,20].forEach(k=>{s+=tx(X(k),t+ph+14,k,{a:'middle'});});
    [0.99,0.95,0.9,0.8].forEach(q=>{let pts='';for(let k=1;k<=N;k++)pts+=X(k).toFixed(1)+','+Y(Math.pow(q,k)).toFixed(1)+' ';
      s+='<polyline points="'+pts+'" fill="none" stroke="#C3CFD5" stroke-width="1.2"/>'+tx(X(N)-4,Y(Math.pow(q,N))-5,q.toFixed(2),{a:'end',z:9.5});});
    let pts='';for(let k=1;k<=N;k++)pts+=X(k).toFixed(1)+','+Y(Math.pow(p,k)).toFixed(1)+' ';
    s+='<polyline points="'+pts+'" fill="none" stroke="#125F9E" stroke-width="2.6"/>';
    const v=Math.pow(p,n);
    s+='<line x1="'+X(n)+'" y1="'+Y(v)+'" x2="'+X(n)+'" y2="'+(t+ph)+'" stroke="#B02F3B" stroke-dasharray="3 3"/><circle cx="'+X(n)+'" cy="'+Y(v)+'" r="5" fill="#B02F3B"/>';
    s+=tx(Math.min(X(n)+9,W-70),Math.max(Y(v)-9,t+12),(100*v).toFixed(1)+'%',{z:13,c:'#14222B',w:700});
    s+=tx(l+6,t+13,'tasks with every step right')+tx(l+pw,H-4,'steps in the task',{a:'end'});
    sv.innerHTML=s;
    const read=n*base+add*n*(n-1)/2;
    $('cl-pl').textContent=p.toFixed(2); $('cl-nl').textContent=n; $('cl-bl').textContent=base; $('cl-al').textContent=add;
    $('cl-out').textContent='every step right       '+p.toFixed(2)+' ^ '+n+' = '+(100*v).toFixed(1)+'%\nfails somewhere         '+(100*(1-v)).toFixed(1)+'%\n\n'+
      'tokens read, call 1     '+base.toLocaleString()+'\ntokens read, call '+n+(n<10?' ':'')+'    '+(base+add*(n-1)).toLocaleString()+
      '\ntokens read, all calls  '+read.toLocaleString()+'\nagainst one call        '+(read/base).toFixed(1)+' times';
  }
  ['cl-p','cl-n','cl-b','cl-a'].forEach(i=>$(i).oninput=draw); draw();
})();
