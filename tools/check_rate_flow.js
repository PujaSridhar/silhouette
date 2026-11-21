#!/usr/bin/env node
/* Drives the built app's Rate tab in Node, through its own click handlers.

     node tools/check_rate_flow.js app/index.html

   Checks the two things the parity fixture cannot see, because they are
   about the flow rather than the arithmetic:

   1. The shared set. Two stylists rating the same wardrobe each meet the
      same outfits, mixed in with their usual ones, and the regular queue
      never serves a shared outfit.
   2. Can't judge. With a name, the button saves a mark - in the marks, never
      in the labels - with the note as its reason; it is in the export; and
      that rater is not served the outfit again. Without a name it saves
      nothing.

   Same stub-DOM approach as check_parity.js: the UI renders into nothing and
   only state is read. Exits 1 on the first failed check. */
"use strict";
const fs = require("fs");
const vm = require("vm");

function stub() {
  const f = function () { return p; };
  const p = new Proxy(f, {
    get(t, k) {
      if (k === Symbol.toPrimitive) return () => "";
      if (k === "then") return undefined;
      if (k === "length") return 0;
      return p;
    },
    set() { return true; },
    apply() { return p; },
    construct() { return p; },
  });
  return p;
}

const path = process.argv[2] || "app/index.html";
const html = fs.readFileSync(path, "utf8");
const data = {};
let code = "";
for (const [, attrs, body] of html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)) {
  const id = (attrs.match(/id="([^"]+)"/) || [])[1];
  if (/application\/json/.test(attrs) && id) data[id] = body;
  else if (!/\bsrc=/.test(attrs)) code += body + "\n";
}

const S_ = stub();
const handlers = {};
const store = new Map();
const document = new Proxy({}, {
  get(_, k) {
    if (k === "getElementById") return (id) => (id in data ? { textContent: data[id] } : S_);
    if (k === "addEventListener") return (type, fn) => { handlers[type] = fn; };
    return S_;
  },
});
const localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};
const ctx = {
  document, window: S_, navigator: S_, localStorage, location: S_,
  console: { log() {}, warn() {}, error() {} },
  setTimeout: () => 0, clearTimeout: () => {}, requestAnimationFrame: () => 0,
  Promise, Math, JSON, Object, Array, Map, Set, String, Number, Date, RegExp, Error,
  Symbol, Proxy, Infinity, NaN, isNaN, parseFloat, parseInt,
};
vm.createContext(ctx);

// Everything below runs synchronously, before the page's async boot resumes,
// so the boot cannot overwrite the state set up here.
const test = `(function(){
  const out=[];
  const ok=(c,m)=>{out.push([!!c,m]);};
  const click=(id,ds)=>__handlers.click({target:{closest:()=>({id,dataset:ds||{}})}});
  const P=BUNDLE.parity;
  const c=newClient("Wearer");
  c.frame={landmarks:P.frame.landmarks,widths:P.frame.widths,heightCm:P.frame.heightCm};
  c.goals=P.goals.slice();
  S.clients=[c];S.activeId=c.id;S.tab="rate";
  // The sample wardrobe, without the drapes.
  S.garments[c.id]=Object.values(P.shared.garments)
    .filter(g=>g.category!=="drape").map(g=>Object.assign({},g));
  const shared=sharedForActive().map(p=>p.k);
  ok(shared.length===SHARED.n,"shared set has "+SHARED.n+" outfits ("+shared.length+")");
  ok(new Set(shared).size===shared.length,"shared outfits are distinct");
  ok(fillRateQueue(40).every(p=>!shared.includes(p.k)),"regular queue never serves a shared outfit");

  function rateAs(who,n,score){
    S.rater=who;S.rateTask=null;S.rateQueue=[];S.rateRepeatServed=false;
    const seen=[];
    for(let i=0;i<n;i++){
      if(!S.rateTask)S.rateTask=nextRateTask();
      const t=S.rateTask;if(!t)break;
      seen.push({k:outfitKeyOf(t.o,t.ctx),shared:!!t.shared,repeat:!!t.repeat});
      S.rating={proportion:null,balance:null,formality:null,colour:null,cohesion:null,
        overall:score(i),would:i%2===0,note:""};
      click("rateSave");
    }
    return seen;
  }
  // Two raters, different scores, same wardrobe.
  const a=rateAs("asha",40,i=>1+(i%5));
  const b=rateAs("bea",40,i=>1+((i+1)%5));
  const sa=a.filter(x=>x.shared).map(x=>x.k),sb=b.filter(x=>x.shared).map(x=>x.k);
  ok(sa.length===SHARED.n,"asha met all "+SHARED.n+" shared outfits in 40 ("+sa.length+")");
  ok(sb.join()===sa.join(),"bea met the same shared outfits, in the same order");
  ok(sa.every(k=>shared.includes(k)),"what was served as shared is the shared set");
  ok(a[0]&&!a[0].shared&&a[1]&&a[1].shared,"shared outfits are mixed in, not served as a block");
  const regA=new Set(a.filter(x=>!x.shared&&!x.repeat).map(x=>x.k));
  ok(b.filter(x=>!x.shared&&!x.repeat).every(x=>!regA.has(x.k)),
    "outside the shared set, raters still get fresh outfits");
  const st=labelStats();
  ok(st.ir.raters.length===2&&st.ir.nUnits===SHARED.n,
    "agreement is measured on the shared outfits ("+st.ir.nUnits+")");
  ok(!isNaN(st.ir.alpha),"alpha is a number once two raters overlap");

  // Can't judge, with a name.
  S.rater="chen";S.rateTask=null;S.rateQueue=[];
  const nLabels=LABELS().length;
  S.rateTask=nextRateTask();
  const t=S.rateTask,k=outfitKeyOf(t.o,t.ctx);
  S.rating.note="  can't tell the fabric  ";
  click("rateSkip");
  ok(LABELS().length===nLabels,"can't judge saves no label");
  ok(MARKS().length===1,"can't judge saves one mark");
  const m=MARKS()[0];
  ok(m&&m.labeller==="chen"&&labelKeyOf(m)===k,"the mark records the rater and the outfit");
  ok(m&&m.reason==="can't tell the fabric","the note is kept as the reason");
  ok(m&&m.overall===undefined&&m.would_wear===undefined,"the mark carries no score");
  const ex=JSON.parse(buildExport());
  ok(ex.cantJudge.length===1&&ex.labels.length===nLabels,"the export carries marks apart from labels");
  ok(labelStats().cantJudge===1,"the stats count the mark");
  let again=false;
  for(let i=0;i<40;i++){
    S.rateTask=null;const u=nextRateTask();if(!u)break;
    if(outfitKeyOf(u.o,u.ctx)===k)again=true;
    S.rating={proportion:null,balance:null,formality:null,colour:null,cohesion:null,
      overall:3,would:true,note:""};S.rateTask=u;click("rateSave");
  }
  ok(!again,"chen is not served the outfit they could not judge");
  ok(interRater(LABELS()).nUnits===SHARED.n,"a mark does not enter agreement");

  // Can't judge, without a name: nothing saved.
  S.rater="";S.rateTask=null;S.rateQueue=[];
  S.rateTask=nextRateTask();
  const before=MARKS().length;
  click("rateSkip");
  ok(MARKS().length===before,"without a name, can't judge saves nothing");
  return out;
})()`;
ctx.__handlers = handlers;
vm.runInContext(code + "\n;globalThis.__flow=" + test + ";", ctx, { filename: path });
const r = ctx.__flow;
let bad = 0;
for (const [pass, msg] of r) {
  if (!pass) bad++;
  console.log(`${pass ? "ok  " : "FAIL"} ${msg}`);
}
console.log(`${path}: rate flow ${r.length - bad}/${r.length}`);
process.exit(bad ? 1 : 0);
