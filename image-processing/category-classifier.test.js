/**
 * category-classifier.test.js
 *
 * Dependency-free test runner (no jest):  node category-classifier.test.js
 *
 * Guards the pure-JS category_probe head. The Python side already PROVED the
 * formula matches sklearn exactly (extract-category-weights.py, max diff 0.0);
 * this guards the JS implementation (normalize / softmax / argmax) and pins it
 * to sklearn's actual numbers via golden fixtures generated from the .pkl.
 *
 * Golden fixtures live in the model repo (category_probe_model/), not shipped in
 * the Lambda zip. If absent, the golden cross-check is skipped and only the
 * property/invariance checks run.
 */
const fs = require("fs");
const path = require("path");
const { classifyCategory } = require("./category-classifier");

let failures = 0;
function ok(cond, msg) {
  if (cond) {
    console.log("  ✓", msg);
  } else {
    failures++;
    console.error("  ✗ FAIL:", msg);
  }
}

// ---- 1. Golden cross-check against sklearn (if fixtures present) ----
const FIXTURES = path.join(
  __dirname,
  "..",
  "..",
  "category_probe_model",
  "category-probe-fixtures.json"
);
if (fs.existsSync(FIXTURES)) {
  const { cases } = JSON.parse(fs.readFileSync(FIXTURES, "utf8"));
  console.log(`Golden cross-check (${cases.length} cases vs sklearn):`);
  for (const c of cases) {
    const r = classifyCategory(c.vector);
    ok(r !== null, `${c.note}: returns a result`);
    ok(r.rootType && r.category === c.expectedLabel,
      `${c.note}: label ${r && r.category} == ${c.expectedLabel}`);
    let maxDiff = 0;
    for (const cls of Object.keys(c.expectedProba)) {
      maxDiff = Math.max(maxDiff, Math.abs(c.expectedProba[cls] - r.proba[cls]));
    }
    ok(maxDiff < 1e-9, `${c.note}: proba matches sklearn (maxDiff=${maxDiff.toExponential(2)})`);
  }
} else {
  console.log("Golden fixtures not found — skipping sklearn cross-check.");
}

// ---- 2. Property / invariance checks ----
console.log("Property checks:");
const dim = 512;
const v = Array.from({ length: dim }, (_, i) => Math.sin(i)); // arbitrary
const r = classifyCategory(v);
ok(r !== null, "valid 512-d vector classifies");
ok(["bags", "clothing", "footwear"].includes(r.category), "category is a known class");
ok(["Bags", "Clothing", "Footwear"].includes(r.rootType), "rootType is a known DB value");
const probaSum = Object.values(r.proba).reduce((a, b) => a + b, 0);
ok(Math.abs(probaSum - 1) < 1e-12, "proba sums to 1");
ok(r.confidence === Math.max(...Object.values(r.proba)), "confidence is the top proba");

// scale-invariance: L2-normalize means scaling the input must not change output
const r10 = classifyCategory(v.map((x) => x * 10));
let invDiff = 0;
for (const cls of Object.keys(r.proba)) invDiff = Math.max(invDiff, Math.abs(r.proba[cls] - r10.proba[cls]));
ok(invDiff < 1e-12, "scale-invariant (x10 input gives same proba)");

// ---- 3. Bad-input guards ----
console.log("Guard checks:");
ok(classifyCategory(null) === null, "null -> null");
ok(classifyCategory([]) === null, "empty -> null");
ok(classifyCategory(new Array(511).fill(0)) === null, "wrong dim -> null");
const bad = new Array(dim).fill(0); bad[3] = NaN;
ok(classifyCategory(bad) === null, "NaN element -> null");

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
