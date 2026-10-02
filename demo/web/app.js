// The web demo: a simulated RGB-D camera looking at the raw stock on a table.
// The box can be moved and rotated; "Estimate pose" renders the camera's RGB
// and depth images, sends them to the demo server, which runs the IOC pose
// estimator's pipeline with LiteRT, and shows the estimate against the truth.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as P from './pose.js';

// The simulated Orbbec Gemini 335Le of OMTS's Lab BB-01 cell (as in
// testdata/service_golden).
const W = 1280, H = 800;
const K = [[597, 0, 640], [0, 593, 400], [0, 0, 1]];
const NEAR = 0.05, FAR = 5;

// The table, in the camera's (OpenCV) frame: tilted by TABLE_TILT about the
// camera's x axis, TABLE_DISTANCE ahead of the camera on its optical axis.
const TABLE_TILT = 35 * Math.PI / 180;
const TABLE_DISTANCE = 0.36;
const BOX_HALF = [0.0381, 0.0635, 0.0254];  // 2 x 5 x 3 in, half sizes in m.

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- scene ---

const canvas = $('three');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xc9cdd2);

// The camera at the origin of the three.js world, looking along -z; the
// group `cv` is the camera's OpenCV frame (x right, y down, z forward), in
// which all poses are given.
const sensor = new THREE.PerspectiveCamera();
sensor.projectionMatrix.fromArray(P.projectionFromIntrinsics(K, W, H, NEAR, FAR));
sensor.projectionMatrixInverse.copy(sensor.projectionMatrix).invert();
scene.add(sensor);
const cv = new THREE.Group();
cv.rotation.x = Math.PI;
scene.add(cv);

const tableNormal = new THREE.Vector3(0, -Math.sin(TABLE_TILT), -Math.cos(TABLE_TILT));
const tableCenter = new THREE.Vector3(0, 0, TABLE_DISTANCE);
const tableFrame = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3().crossVectors(tableNormal, new THREE.Vector3(1, 0, 0)),
    tableNormal).setPosition(tableCenter);

const table = new THREE.Mesh(
    new THREE.PlaneGeometry(1.0, 1.0),
    new THREE.MeshStandardMaterial({ color: 0x5a5a5a, roughness: 0.9 }));
table.applyMatrix4(tableFrame);
cv.add(table);
// Aluminium profiles along the table's edges, as in the cell.
for (const [x, y, sx, sy] of [[0, 0.32, 0.7, 0.03], [0, -0.32, 0.7, 0.03],
                              [0.34, 0, 0.03, 0.67], [-0.34, 0, 0.03, 0.67]]) {
  const bar = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, 0.03),
                             new THREE.MeshStandardMaterial({ color: 0xe8e8e8, metalness: 0.2, roughness: 0.4 }));
  bar.applyMatrix4(new THREE.Matrix4().makeTranslation(x, y, 0.015).premultiply(tableFrame));
  cv.add(bar);
}

scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 1.8));
const sun = new THREE.DirectionalLight(0xffffff, 2.6);
sun.position.copy(tableCenter).addScaledVector(tableNormal, 1.0).add(new THREE.Vector3(0.3, 0, -0.2));
sun.target.position.copy(tableCenter);
cv.add(sun, sun.target);

// The raw stock (its pose is the ground truth) and the estimate's ghost.
const box = new THREE.Group();
cv.add(box);
const estimate = new THREE.Group();
estimate.matrixAutoUpdate = false;
estimate.visible = false;
cv.add(estimate);
let boxMesh = null;

function restingPose(u, v, yawDeg, face = 'z') {
  // The box lying on its largest face (or the given face) at (u, v) on the
  // table, rotated by yaw about the table's normal.
  const half = { x: BOX_HALF[0], y: BOX_HALF[1], z: BOX_HALF[2] }[face];
  const lay = new THREE.Matrix4();
  if (face === 'x') lay.makeRotationY(Math.PI / 2);
  if (face === 'y') lay.makeRotationX(Math.PI / 2);
  return new THREE.Matrix4().makeTranslation(u, v, half)
      .multiply(new THREE.Matrix4().makeRotationZ(yawDeg * Math.PI / 180))
      .multiply(lay).premultiply(tableFrame);
}
const INITIAL_POSE = restingPose(0.03, 0.04, 30);
function setBoxPose(m) {
  m.decompose(box.position, box.quaternion, box.scale);
  updateGroundTruth();
}

// --------------------------------------------------------------- views ---

const orbitCamera = new THREE.PerspectiveCamera(45, W / H, 0.01, 10);
// "Up" for the 3D view is the table's normal (in three.js world coordinates).
orbitCamera.up.copy(tableNormal).applyQuaternion(cv.quaternion);
cv.updateMatrixWorld();
const tableCenterWorld = cv.localToWorld(tableCenter.clone());
orbitCamera.position.copy(tableCenterWorld).addScaledVector(orbitCamera.up, 0.55)
    .add(new THREE.Vector3(0.5, 0, 0.3));
const orbit = new OrbitControls(orbitCamera, canvas);
orbit.target.copy(tableCenterWorld);
orbit.update();

const sensorHelper = new THREE.CameraHelper(sensor);
sensorHelper.setColors(new THREE.Color(0xffaa00), new THREE.Color(0xff2200),
                       new THREE.Color(0x00aaff), new THREE.Color(0xffffff), new THREE.Color(0x888888));
// Show the frustum only up to the table.
sensor.projectionMatrix.fromArray(P.projectionFromIntrinsics(K, W, H, NEAR, 0.5));
sensor.projectionMatrixInverse.copy(sensor.projectionMatrix).invert();
sensorHelper.update();
sensor.projectionMatrix.fromArray(P.projectionFromIntrinsics(K, W, H, NEAR, FAR));
sensor.projectionMatrixInverse.copy(sensor.projectionMatrix).invert();
scene.add(sensorHelper);

const gizmo = new TransformControls(orbitCamera, canvas);
gizmo.setSpace('local');
gizmo.setSize(0.8);
gizmo.addEventListener('dragging-changed', (e) => { orbit.enabled = !e.value; });
gizmo.addEventListener('objectChange', updateGroundTruth);
const gizmoHelper = gizmo.getHelper();
scene.add(gizmoHelper);

let view = 'orbit';
function setView(v) {
  view = v;
  gizmo.camera = v === 'orbit' ? orbitCamera : sensor;
  orbit.enabled = v === 'orbit';
  sensorHelper.visible = v === 'orbit';
  $('inset-label').hidden = v !== 'orbit';
  for (const b of $('view').children) b.classList.toggle('on', b.dataset.view === v);
}

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas;
  if (canvas.width !== Math.round(w * renderer.getPixelRatio())) {
    renderer.setSize(w, h, false);
  }
}

function draw() {
  resize();
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, w, h);
  renderer.render(scene, view === 'orbit' ? orbitCamera : sensor);
  if (view === 'orbit') {
    // Inset: what the camera sees.
    const iw = Math.round(w * 0.26), ih = Math.round(iw * H / W);
    renderer.setScissorTest(true);
    renderer.setScissor(w - iw - 8, 8, iw, ih);
    renderer.setViewport(w - iw - 8, 8, iw, ih);
    const helpers = hideHelpers();
    renderer.render(scene, sensor);
    helpers();
  }
  requestAnimationFrame(draw);
}

function hideHelpers({ keepEstimate = true } = {}) {
  const hidden = [sensorHelper, gizmoHelper, ...(keepEstimate ? [] : [estimate])]
      .filter((o) => o.visible);
  for (const o of hidden) o.visible = false;
  return () => { for (const o of hidden) o.visible = true; };
}

// ------------------------------------------------------- ground truth ---

function boxPose() {
  box.updateMatrix();
  const e = box.matrix.elements;  // Column-major, box in the camera's frame.
  return [0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => e[c * 4 + r]));
}

function fmt(v, digits = 1) { return v.toFixed(digits); }

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

function updateGroundTruth() {
  const pose = boxPose();
  const t = P.translationOf(pose).map((v) => v * 1000);
  const rpy = P.rpyDeg(P.rotationOf(pose));
  const labels = ['x mm', 'y mm', 'z mm', 'roll°', 'pitch°', 'yaw°'];
  $('gt').innerHTML = labels.map((l) => `<dt>${l}</dt>`).join('') +
      [...t, ...rpy].map((v) => `<dd>${fmt(v)}</dd>`).join('');
}

// ------------------------------------------------------------- capture ---

// Renders the camera's images at full resolution: RGB like the color
// camera, depth (meters along the optical axis, 0 where nothing is hit) like
// the depth camera, registered to it.
const capture = (() => {
  const offscreen = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  offscreen.setPixelRatio(1);
  offscreen.setSize(W, H, false);
  const depthTarget = new THREE.WebGLRenderTarget(W, H, {
    type: THREE.FloatType, format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  });
  const depthMaterial = new THREE.ShaderMaterial({
    vertexShader: `varying float vDepth;
      void main() {
        vec4 p = modelViewMatrix * vec4(position, 1.0);
        vDepth = -p.z;
        gl_Position = projectionMatrix * p;
      }`,
    fragmentShader: `varying float vDepth;
      void main() { gl_FragColor = vec4(vDepth, 0.0, 0.0, 1.0); }`,
    side: THREE.DoubleSide,
  });
  const canvas2d = document.createElement('canvas');
  canvas2d.width = W; canvas2d.height = H;
  const ctx = canvas2d.getContext('2d', { willReadFrequently: true });

  function rgb() {
    offscreen.setRenderTarget(null);
    offscreen.render(scene, sensor);
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(offscreen.domElement, 0, 0);
    const rgba = ctx.getImageData(0, 0, W, H).data;
    const out = new Uint8Array(W * H * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      out[j] = rgba[i]; out[j + 1] = rgba[i + 1]; out[j + 2] = rgba[i + 2];
    }
    return { out, imageData: ctx.getImageData(0, 0, W, H) };
  }

  function depth() {
    const background = scene.background;
    scene.background = null;
    scene.overrideMaterial = depthMaterial;
    offscreen.setRenderTarget(depthTarget);
    offscreen.setClearColor(0x000000, 0);
    offscreen.clear();
    offscreen.render(scene, sensor);
    offscreen.setRenderTarget(null);
    scene.overrideMaterial = null;
    scene.background = background;
    const rgba = new Float32Array(W * H * 4);
    offscreen.readRenderTargetPixels(depthTarget, 0, 0, W, H, rgba);
    const out = new Float32Array(W * H);
    for (let y = 0; y < H; ++y) {  // WebGL's rows go up.
      const src = (H - 1 - y) * W;
      for (let x = 0; x < W; ++x) out[y * W + x] = rgba[(src + x) * 4];
    }
    return out;
  }

  // Renders `object` alone (transparent elsewhere) as the camera sees it.
  function overlay(object) {
    const s = new THREE.Scene();
    s.add(new THREE.HemisphereLight(0xffffff, 0x404040, 2.5));
    const group = new THREE.Group();
    group.rotation.x = Math.PI;
    group.add(object);
    s.add(group);
    offscreen.setClearColor(0x000000, 0);
    offscreen.render(s, sensor);
    return offscreen.domElement;
  }

  return {
    frame() {
      const restore = hideHelpers({ keepEstimate: false });
      const t0 = performance.now();
      const color = rgb();
      const d = depth();
      restore();
      return { rgb: color.out, imageData: color.imageData, depth: d, ms: performance.now() - t0 };
    },
    // The scene as the camera would see it with the box at `pose` (4x4 rows).
    renderAt(pose) {
      box.updateMatrix();
      const saved = box.matrix.clone();
      setBoxPoseRaw(pose);
      const restore = hideHelpers({ keepEstimate: false });
      offscreen.setRenderTarget(null);
      offscreen.render(scene, sensor);
      restore();
      saved.decompose(box.position, box.quaternion, box.scale);
      return offscreen.domElement;
    },
    overlay,
  };
})();

function matrixFromRows(rows) {
  return new THREE.Matrix4().set(...rows.flat());
}
function setBoxPoseRaw(rows) {
  matrixFromRows(rows).decompose(box.position, box.quaternion, box.scale);
}

// ------------------------------------------------------------ estimate ---

let lastFrame = null, lastResult = null, imageTab = 'overlay';

async function runEstimate() {
  const button = $('estimate');
  button.disabled = true;
  setStatus('Rendering the camera images…');
  await new Promise(requestAnimationFrame);
  const truth = boxPose();
  const frame = capture.frame();
  const t0 = performance.now();
  const body = JSON.stringify({
    width: W, height: H, camera_matrix: K,
    rgb: P.toBase64(frame.rgb), depth: P.toBase64(frame.depth),
    accelerator: $('accelerator').value,
    cpu_fallback: cpuFallback,
    iterations: Number($('iterations').value),
  });
  const encodeMs = performance.now() - t0;
  setStatus(`Running segmentation and FoundationPose with LiteRT (${$('accelerator').value})…`);
  const ticker = setInterval(() => {
    setStatus(`Running segmentation and FoundationPose with LiteRT (${$('accelerator').value})… ${((performance.now() - t0) / 1000).toFixed(0)} s`);
  }, 1000);
  let result;
  try {
    const response = await fetch('/api/estimate', { method: 'POST', body, headers: { 'Content-Type': 'application/json' } });
    result = await response.json();
    if (!response.ok) throw new Error(result.error || response.statusText);
  } catch (e) {
    clearInterval(ticker);
    setStatus(`Estimation failed: ${e.message}`, true);
    button.disabled = false;
    report({ error: e.message });
    return;
  }
  clearInterval(ticker);
  const roundTripMs = performance.now() - t0 - encodeMs;
  result.client_ms = { capture: frame.ms, encode: encodeMs, round_trip: roundTripMs };
  lastFrame = frame; lastResult = result;
  show(result, truth);
  button.disabled = false;
}

function setStatus(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}

function best(result) {
  // The detection with the highest pose score.
  return result.detections.reduce((a, b) => (b.pose_score > a.pose_score ? b : a), result.detections[0]);
}

function show(result, truth) {
  $('results').hidden = false;
  const det = result.detections.length ? best(result) : null;
  if (!det) {
    setStatus('No raw stock found in the camera image (is it in view?).', true);
    estimate.visible = false;
    drawImage();
    showLatency(result);
    showAccelerators(result);
    report({ detections: 0, timings_ms: result.timings_ms });
    return;
  }
  setStatus(`Found ${result.detections.length} object${result.detections.length > 1 ? 's' : ''}; ` +
            `segmentation score ${det.score.toFixed(3)}, pose score ${det.pose_score.toFixed(2)}.`);

  // The ghost in the 3D view.
  estimate.matrix.copy(matrixFromRows(det.pose));
  estimate.visible = true;

  const et = P.translationOf(det.pose), gt = P.translationOf(truth);
  const er = P.rpyDeg(P.rotationOf(det.pose)), gr = P.rpyDeg(P.rotationOf(truth));
  const ids = ['x', 'y', 'z', 'r', 'p', 'w'];
  [...et.map((v) => v * 1000), ...er].forEach((v, i) => { $(`e-${ids[i]}`).textContent = fmt(v); });
  [...gt.map((v) => v * 1000), ...gr].forEach((v, i) => { $(`g-${ids[i]}`).textContent = fmt(v); });
  const sr = P.rpyDeg(P.closestSymmetric(P.rotationOf(det.pose), P.rotationOf(truth)));
  sr.forEach((v, i) => { $(`s-${ids[i + 3]}`).textContent = fmt(v); });

  const tErr = P.distanceMm(et, gt);
  const rErr = P.symmetricAngleDeg(P.rotationOf(det.pose), P.rotationOf(truth));
  const addS = P.addSMm(det.pose, truth, P.boxSurfacePoints(BOX_HALF));
  const metric = (value, label, good) =>
    `<div class="metric ${good ? 'good' : 'warn'}"><b>${value}</b><span>${label}</span></div>`;
  $('metrics').innerHTML =
      metric(`${fmt(tErr)} mm`, 'translation error', tErr < 5) +
      metric(`${fmt(rErr)}°`, 'rotation error (mod. symmetry)', rErr < 5) +
      metric(`${fmt(addS)} mm`, 'ADD-S', addS < 5) +
      metric(det.score.toFixed(3), 'segmentation score', true) +
      metric(`${det.mask_pixels}`, 'mask pixels', true);
  $('matrix').textContent = 'T_camera_object =\n' + det.pose.map(
      (row) => '  [' + row.map((v) => v.toFixed(5).padStart(9)).join(' ') + ' ]').join('\n');

  drawImage();
  showLatency(result);
  showAccelerators(result);
  report({
    detections: result.detections.length,
    truth, estimate: det.pose, translation_error_mm: tErr, rotation_error_deg: rErr,
    add_s_mm: addS, segmentation_score: det.score, pose_score: det.pose_score,
    timings_ms: result.timings_ms, client_ms: result.client_ms, accelerators: result.accelerators,
  });
}

function drawImage() {
  const ctx = $('result-image').getContext('2d');
  if (!lastFrame) return;
  const det = lastResult && lastResult.detections.length ? best(lastResult) : null;
  if (imageTab === 'render' && det) {
    ctx.drawImage(capture.renderAt(det.pose), 0, 0);
    return;
  }
  ctx.putImageData(lastFrame.imageData, 0, 0);
  if (imageTab !== 'overlay' || !det) return;
  // Mask, box and the CAD model at the estimated pose.
  const mask = P.decodeMask(det.mask_bits, W, H);
  const tint = ctx.getImageData(0, 0, W, H);
  for (let i = 0; i < mask.length; ++i) {
    if (!mask[i]) continue;
    tint.data[i * 4] = tint.data[i * 4] * 0.6;
    tint.data[i * 4 + 1] = tint.data[i * 4 + 1] * 0.6 + 40;
    tint.data[i * 4 + 2] = tint.data[i * 4 + 2] * 0.6 + 100;
  }
  ctx.putImageData(tint, 0, 0);
  ctx.globalAlpha = 0.85;
  ctx.drawImage(capture.overlay(ghostAt(det.pose)), 0, 0);
  ctx.globalAlpha = 1;
  const [x0, y0, x1, y1] = det.box_xyxy;
  ctx.strokeStyle = '#ffd400'; ctx.lineWidth = 3;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.font = '600 22px system-ui, sans-serif';
  ctx.fillStyle = '#ffd400';
  ctx.fillText(`raw stock ${det.score.toFixed(2)}`, x0, Math.max(24, y0 - 8));
}

// A translucent, outlined copy of the CAD model at `pose`.
function ghostAt(pose) {
  const g = new THREE.Group();
  g.matrixAutoUpdate = false;
  g.matrix.copy(matrixFromRows(pose));
  if (boxMesh) {
    boxMesh.traverse((o) => {
      if (!o.isMesh) return;
      const m = new THREE.Mesh(o.geometry, new THREE.MeshBasicMaterial({ color: 0x34d058, transparent: true, opacity: 0.35, depthWrite: false }));
      m.applyMatrix4(o.matrixWorld.clone().premultiply(boxMesh.matrixWorld.clone().invert()));
      g.add(m);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry, 30),
                                           new THREE.LineBasicMaterial({ color: 0x34d058 }));
      edges.applyMatrix4(m.matrix);
      g.add(edges);
    });
  }
  return g;
}

const LATENCY = [
  ['capture', 'Render RGB-D (browser)', '--c-capture', (r) => r.client_ms.capture],
  ['encode', 'Encode request (browser)', '--c-encode', (r) => r.client_ms.encode],
  ['transfer', 'Transfer and HTTP', '--c-transfer',
   (r) => r.client_ms.round_trip - r.timings_ms.decode -
          r.timings_ms.model_compilation - r.timings_ms.total],
  ['compile', 'Model compilation (first request per setting)', '--c-compile',
   (r) => r.timings_ms.model_compilation],
  ['decode', 'Decode request (server)', '--c-decode', (r) => r.timings_ms.decode],
  ['seg', 'Segmentation, RF-DETR (LiteRT)', '--c-seg', (r) => r.timings_ms.segmentation],
  ['pose', 'Pose estimation, FoundationPose (LiteRT)', '--c-pose', (r) => r.timings_ms.pose_estimation],
];

function showLatency(r) {
  // Model compilation only in the first request with these settings.
  const rows = LATENCY.filter(([id]) => id !== 'compile' || r.timings_ms.model_compilation > 0);
  const values = rows.map(([, , , f]) => Math.max(0, f(r)));
  const total = values.reduce((a, b) => a + b, 0);
  $('latency-bar').innerHTML = rows.map(([, label, color], i) =>
    `<div title="${label}" style="width:${(100 * values[i] / total).toFixed(2)}%;background:var(${color})"></div>`).join('');
  const ms = (v) => (v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(1)} ms`);
  $('latency').innerHTML = rows.map(([, label, color], i) =>
    `<tr><td><span class="swatch" style="background:var(${color})"></span>${label}</td><td>${ms(values[i])}</td></tr>`).join('') +
    `<tr><th>End to end</th><th>${ms(total)}</th></tr>`;
}

function showAccelerators(r) {
  const rows = [['RF-DETR', r.accelerators.segmentation], ...Object.entries(r.accelerators.foundationpose)
      .map(([k, v]) => [`FoundationPose ${k.replace('_', ' ')}`, v])];
  $('accel').innerHTML = rows.map(([name, a]) =>
    `<div><b>${name}</b>${a.accelerator === 'gpu' ? 'GPU (WebGPU accelerator' + (a.fully_accelerated ? ', fully delegated)' : ', partially delegated)') : 'CPU (XNNPACK)'}` +
    (a.fallback_reason ? `<br><span>${escapeHtml(a.fallback_reason)}</span>` : '') + '</div>').join('') +
    `<span>${r.config.iterations} refinement iteration(s), scorer batch ${r.config.batch_size}, ` +
    `${r.config.cpu_fallback ? 'CPU fallback on' : 'no CPU fallback'}.</span>`;
}

// Reports the outcome to the server for the browser test (?autorun=1).
function report(summary) {
  window.__result = summary;
  document.body.dataset.state = summary.error ? 'error' : 'done';
  if (params.has('autorun')) {
    fetch('/api/report', { method: 'POST', body: JSON.stringify(summary) });
  }
}

// --------------------------------------------------------------- wiring ---

function randomPose() {
  const face = ['z', 'z', 'x', 'y'][Math.floor(Math.random() * 4)];
  return restingPose((Math.random() - 0.5) * 0.14, (Math.random() - 0.5) * 0.12,
                     Math.random() * 360, face);
}

function setMode(mode) {
  gizmo.setMode(mode);
  for (const b of $('mode').children) b.classList.toggle('on', b.dataset.mode === mode);
}

$('mode').addEventListener('click', (e) => e.target.dataset.mode && setMode(e.target.dataset.mode));
$('view').addEventListener('click', (e) => e.target.dataset.view && setView(e.target.dataset.view));
$('random').addEventListener('click', () => setBoxPose(randomPose()));
$('reset').addEventListener('click', () => setBoxPose(INITIAL_POSE));
$('estimate').addEventListener('click', runEstimate);
$('iterations').addEventListener('input', (e) => { $('iterations-value').textContent = e.target.value; });
$('image-tabs').addEventListener('click', (e) => {
  if (!e.target.dataset.image) return;
  imageTab = e.target.dataset.image;
  for (const b of $('image-tabs').children) b.classList.toggle('on', b === e.target);
  drawImage();
});
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'w') setMode('translate');
  if (e.key === 'e') setMode('rotate');
  if (e.key === 'r') setBoxPose(INITIAL_POSE);
});

if (params.has('accelerator')) $('accelerator').value = params.get('accelerator');
// Run networks that the GPU can't run on the CPU instead of failing (?cpu_fallback=1).
const cpuFallback = params.get('cpu_fallback') === '1';
if (params.has('iterations')) {
  $('iterations').value = params.get('iterations');
  $('iterations-value').textContent = params.get('iterations');
}

new GLTFLoader().load('assets/raw_stock_2x3x5.glb', (gltf) => {
  boxMesh = gltf.scene;
  box.add(boxMesh);
  box.updateMatrixWorld(true);
  setBoxPose(INITIAL_POSE);
  gizmo.attach(box);
  // The estimate's ghost in the 3D view.
  const ghost = ghostAt([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
  ghost.matrixAutoUpdate = true;
  estimate.add(ghost);
  document.body.dataset.state = 'ready';
  // ?pose=<4x4 rows as JSON> or ?rest=<u>,<v>,<yaw deg>,<face x|y|z>.
  if (params.has('pose')) setBoxPose(matrixFromRows(JSON.parse(params.get('pose'))));
  if (params.has('rest')) {
    const [u, v, yaw, face] = params.get('rest').split(',');
    setBoxPose(restingPose(Number(u), Number(v), Number(yaw), face || 'z'));
  }
  if (params.has('autorun')) runEstimate();
}, undefined, (e) => setStatus(`Could not load the CAD model: ${e.message || e}`, true));

setView('orbit');
draw();
