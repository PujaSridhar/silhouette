#!/usr/bin/env node
/* Replays a built page's parity fixture through the page's own engine, in Node.

     node tools/check_parity.js demo/index.html app/index.html

   The pages check themselves in the browser on load; this runs the same
   runParity() without one, so `make parity` and the test suite can catch a
   drifted JS port before it is published. The page's UI code runs against a
   stub DOM that swallows everything, and only the parity result is read. */
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

function check(path) {
  const html = fs.readFileSync(path, "utf8");
  const scripts = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
  const data = {};
  let code = "";
  for (const [, attrs, body] of scripts) {
    const id = (attrs.match(/id="([^"]+)"/) || [])[1];
    if (/application\/json/.test(attrs) && id) data[id] = body;
    else if (!/\bsrc=/.test(attrs)) code += body + "\n";
  }
  const S = stub();
  const document = new Proxy({}, {
    get(_, k) {
      if (k === "getElementById") return (id) => (id in data ? { textContent: data[id] } : S);
      return S;
    },
  });
  const ctx = {
    document, window: S, navigator: S, localStorage: S, claude: S, location: S,
    console: { log() {}, warn() {}, error() {} },
    setTimeout: () => 0, clearTimeout: () => {}, requestAnimationFrame: () => 0,
    Promise, Math, JSON, Object, Array, Map, Set, String, Number, Date, RegExp, Error,
    Symbol, Proxy, Infinity, NaN, isNaN, parseFloat, parseInt,
  };
  vm.createContext(ctx);
  vm.runInContext(code + "\n;globalThis.__parity = runParity();", ctx, { filename: path });
  const r = ctx.__parity;
  const pass = r.pass !== undefined ? r.pass : r.n - r.fails.length;
  return { path, n: r.n, pass, fails: r.fails.map((f) => f.why) };
}

let bad = 0;
for (const p of process.argv.slice(2)) {
  const r = check(p);
  console.log(`${r.path}: parity ${r.pass}/${r.n}`);
  if (r.fails.length) { bad++; console.log("  first failures:", JSON.stringify(r.fails.slice(0, 5))); }
}
process.exit(bad ? 1 : 0);
