/**
 * category-classifier.js
 *
 * Image-only product category classifier (category_probe).
 *
 * The model is a scikit-learn multinomial LogisticRegression head that runs on
 * the 512-d Marqo-FashionCLIP image embedding we already compute for every
 * upload (see index.js -> vectorizeImage). Classifying is therefore "a single
 * matrix-multiply" on a vector we already have — no Python, no extra deps.
 *
 * We do NOT load the .pkl here. Its weights were transcoded once, offline, into
 * category-probe-weights.json (see category_probe_model/extract-category-weights.py),
 * which also PROVED this JS reproduces sklearn's predict_proba exactly
 * (max |sklearn - js| = 0.0 over 2000 vectors). predict_proba for a multinomial
 * LogisticRegression is softmax(X @ coef.T + intercept).
 */

const WEIGHTS = require("./category-probe-weights.json");

const DIM = WEIGHTS.dim; // 512
const CLASSES = WEIGHTS.classes; // ["bags","clothing","footwear"]
const ROOT_TYPES = WEIGHTS.rootTypes; // ["Bags","Clothing","Footwear"]
const COEF = WEIGHTS.coef; // (n_classes, dim)
const INTERCEPT = WEIGHTS.intercept; // (n_classes,)
const MULTINOMIAL = WEIGHTS.multinomial !== false;
const VERSION = WEIGHTS.version;

/**
 * Classify a product image from its 512-d Marqo-FashionCLIP embedding.
 *
 * @param {number[]} vector - 512-d image embedding (raw or L2-normalized).
 * @returns {{category: string, rootType: string, confidence: number,
 *            proba: Object<string, number>, version: string}|null}
 *          null if the vector is missing / wrong-dimension / non-finite.
 */
function classifyCategory(vector) {
  if (!Array.isArray(vector) || vector.length !== DIM) {
    return null;
  }

  // L2-normalize defensively (the head was trained on L2-normalized vectors;
  // production vectors already are, so this is usually a no-op).
  let normSq = 0;
  for (let i = 0; i < DIM; i++) {
    const x = vector[i];
    if (typeof x !== "number" || !Number.isFinite(x)) return null;
    normSq += x * x;
  }
  const norm = Math.sqrt(normSq);
  const inv = norm > 0 ? 1 / norm : 1;

  // logits[k] = (v . coef[k]) + intercept[k]
  const nClasses = CLASSES.length;
  const logits = new Array(nClasses);
  let maxLogit = -Infinity;
  for (let k = 0; k < nClasses; k++) {
    const row = COEF[k];
    let dot = 0;
    for (let i = 0; i < DIM; i++) {
      dot += vector[i] * inv * row[i];
    }
    const logit = dot + INTERCEPT[k];
    logits[k] = logit;
    if (logit > maxLogit) maxLogit = logit;
  }

  // proba: softmax (multinomial) or normalized per-class sigmoid (OvR fallback)
  const proba = new Array(nClasses);
  let sum = 0;
  for (let k = 0; k < nClasses; k++) {
    const p = MULTINOMIAL
      ? Math.exp(logits[k] - maxLogit) // softmax, max-shifted for stability
      : 1 / (1 + Math.exp(-logits[k])); // sigmoid
    proba[k] = p;
    sum += p;
  }

  let topIdx = 0;
  const probaByClass = {};
  for (let k = 0; k < nClasses; k++) {
    proba[k] = proba[k] / sum;
    probaByClass[CLASSES[k]] = proba[k];
    if (proba[k] > proba[topIdx]) topIdx = k;
  }

  return {
    category: CLASSES[topIdx], // model class: bags|clothing|footwear
    rootType: ROOT_TYPES[topIdx], // product DB value: Bags|Clothing|Footwear
    confidence: proba[topIdx],
    proba: probaByClass,
    version: VERSION,
  };
}

module.exports = { classifyCategory, CATEGORY_MODEL_VERSION: VERSION };
