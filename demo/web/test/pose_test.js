// Unit tests of pose.js, run in the browser by demo_test.py (or by opening
// test/pose_test.html). The results go to the page and to /api/report.

import * as P from '../pose.js';

const results = [];
function test(name, fn) {
  try {
    fn();
    results.push({ name, ok: true });
  } catch (e) {
    results.push({ name, ok: false, error: e.message });
  }
}
function near(actual, expected, tolerance, what = '') {
  if (!(Math.abs(actual - expected) <= tolerance)) {
    throw new Error(`${what} ${actual} is not within ${tolerance} of ${expected}`);
  }
}

const rz = (deg) => {
  const a = deg * Math.PI / 180;
  return [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]];
};
const ry = (deg) => {
  const a = deg * Math.PI / 180;
  return [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
};
const rx = (deg) => {
  const a = deg * Math.PI / 180;
  return [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
};
const mul = (a, b) => a.map((row) => [0, 1, 2].map((c) => row.reduce((s, v, k) => s + v * b[k][c], 0)));
const pose = (r, t) => [[...r[0], t[0]], [...r[1], t[1]], [...r[2], t[2]], [0, 0, 0, 1]];

test('angle between rotations', () => {
  near(P.angleDeg(rz(10), rz(10)), 0, 1e-6);
  near(P.angleDeg(rz(0), rz(90)), 90, 1e-6);
  near(P.angleDeg(rx(30), rx(-30)), 60, 1e-6);
});

test('box half-turns are no rotation error', () => {
  const r = mul(rz(37), ry(-12));
  for (const flip of [rx(180), ry(180), rz(180)]) {
    near(P.symmetricAngleDeg(r, mul(r, flip)), 0, 1e-5);
  }
  near(P.symmetricAngleDeg(r, mul(r, rz(90))), 90, 1e-5);  // Not a symmetry.
});

test('closest symmetric equivalent', () => {
  const truth = mul(rz(20), rx(5));
  const estimate = mul(mul(truth, ry(180)), rz(2));
  near(P.angleDeg(P.closestSymmetric(estimate, truth), truth), 2, 1e-5);
});

test('roll, pitch, yaw', () => {
  const [r, p, y] = P.rpyDeg(mul(mul(rz(-25), ry(16)), rx(140)));
  near(r, 140, 1e-6, 'roll'); near(p, 16, 1e-6, 'pitch'); near(y, -25, 1e-6, 'yaw');
});

test('projection matches the pinhole camera', () => {
  const k = [[597, 0, 640], [0, 593, 400], [0, 0, 1]];
  const m = P.projectionFromIntrinsics(k, 1280, 800, 0.05, 5);  // Column-major.
  for (const [x, y, z] of [[0.05, -0.03, 0.4], [-0.1, 0.07, 0.9], [0, 0, 0.3]]) {
    const gl = [x, -y, -z, 1];  // OpenCV to OpenGL camera coordinates.
    const clip = [0, 1, 2, 3].map((r) => [0, 1, 2, 3].reduce((s, c) => s + m[c * 4 + r] * gl[c], 0));
    const u = (clip[0] / clip[3] + 1) / 2 * 1280;
    const v = (1 - clip[1] / clip[3]) / 2 * 800;
    near(u, 597 * x / z + 640, 1e-6, 'u');
    near(v, 593 * y / z + 400, 1e-6, 'v');
    const ndcZ = clip[2] / clip[3];
    if (!(ndcZ > -1 && ndcZ < 1)) throw new Error(`depth ${ndcZ} is clipped`);
  }
});

test('masks packed by numpy.packbits', () => {
  // numpy.packbits([1, 0, 1, 1, 0, 0, 0, 0, 1]) == [0b10110000, 0b10000000].
  const mask = P.decodeMask(btoa(String.fromCharCode(0b10110000, 0b10000000)), 3, 3);
  if (mask.join('') !== '101100001') throw new Error(`got ${mask.join('')}`);
});

test('base64 of typed arrays', () => {
  if (P.toBase64(new Float32Array([1])) !== 'AACAPw==') throw new Error('float32');
  if (P.toBase64(new Uint8Array([1, 2, 3, 250])) !== 'AQID+g==') throw new Error('uint8');
  const big = new Uint8Array(100000).map((_, i) => i % 251);
  const back = Uint8Array.from(atob(P.toBase64(big)), (c) => c.charCodeAt(0));
  if (back.length !== big.length || back.some((v, i) => v !== big[i])) throw new Error('large');
});

test('ADD-S', () => {
  const points = P.boxSurfacePoints([0.0381, 0.0635, 0.0254]);
  const truth = pose(rz(30), [0.01, 0.02, 0.35]);
  near(P.addSMm(truth, truth, points), 0, 1e-9);
  near(P.addSMm(pose(mul(rz(30), rz(180)), [0.01, 0.02, 0.35]), truth, points), 0, 1e-6, 'symmetric');
  const shifted = P.addSMm(pose(rz(30), [0.012, 0.02, 0.35]), truth, points);
  if (!(shifted > 0.5 && shifted <= 2.0001)) throw new Error(`shifted by 2 mm: ${shifted}`);
});

const failed = results.filter((r) => !r.ok);
document.getElementById('out').textContent = results.map(
    (r) => `${r.ok ? 'PASS' : 'FAIL'} ${r.name}${r.ok ? '' : ': ' + r.error}`).join('\n');
document.body.dataset.state = failed.length ? 'failed' : 'passed';
fetch('/api/report', { method: 'POST', body: JSON.stringify({ tests: results }) });
