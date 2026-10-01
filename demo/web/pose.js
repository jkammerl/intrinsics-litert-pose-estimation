// Pose math for the demo, on plain arrays (3x3 rotations as nested rows,
// 4x4 poses as nested rows; OpenCV camera frame: x right, y down, z forward).

// The raw stock is a box: the half-turns about its axes map it onto itself.
export const BOX_SYMMETRIES = [
  [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
];

export function rotationOf(pose) {
  return [0, 1, 2].map((r) => pose[r].slice(0, 3));
}

export function translationOf(pose) {
  return [pose[0][3], pose[1][3], pose[2][3]];
}

// Angle in degrees of the rotation between a and b.
export function angleDeg(a, b) {
  let trace = 0;
  for (let i = 0; i < 3; ++i) {
    for (let k = 0; k < 3; ++k) trace += a[k][i] * b[k][i];
  }
  return Math.acos(Math.min(1, Math.max(-1, (trace - 1) / 2))) * 180 / Math.PI;
}

// Rotation error modulo the box's symmetries (as the golden tests measure it).
export function symmetricAngleDeg(a, b) {
  let best = 180;
  for (const s of BOX_SYMMETRIES) {
    const bs = b.map((row) => row.map((v, c) => v * s[c]));
    best = Math.min(best, angleDeg(a, bs));
  }
  return best;
}

// The rotation equivalent to a (modulo the box's symmetries) closest to b.
export function closestSymmetric(a, b) {
  let best = a, bestAngle = Infinity;
  for (const s of BOX_SYMMETRIES) {
    const as = a.map((row) => row.map((v, c) => v * s[c]));
    const angle = angleDeg(as, b);
    if (angle < bestAngle) { best = as; bestAngle = angle; }
  }
  return best;
}

// Points on the surface of a box with the given half sizes, n per edge.
export function boxSurfacePoints(half, n = 6) {
  const out = [];
  const steps = [...Array(n).keys()].map((i) => -1 + 2 * i / (n - 1));
  for (let axis = 0; axis < 3; ++axis) {
    for (const side of [-1, 1]) {
      for (const u of steps) for (const v of steps) {
        const p = [0, 0, 0];
        p[axis] = side * half[axis];
        p[(axis + 1) % 3] = u * half[(axis + 1) % 3];
        p[(axis + 2) % 3] = v * half[(axis + 2) % 3];
        out.push(p);
      }
    }
  }
  return out;
}

export function distanceMm(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) * 1000;
}

// Roll (x), pitch (y), yaw (z) in degrees, for R = Rz(yaw) Ry(pitch) Rx(roll).
export function rpyDeg(r) {
  const deg = 180 / Math.PI;
  const pitch = Math.asin(Math.max(-1, Math.min(1, -r[2][0])));
  if (Math.abs(r[2][0]) > 0.99999) {  // Gimbal lock: put it all in yaw.
    return [0, pitch * deg, Math.atan2(-r[0][1], r[1][1]) * deg];
  }
  return [Math.atan2(r[2][1], r[2][2]) * deg, pitch * deg,
          Math.atan2(r[1][0], r[0][0]) * deg];
}

// Average distance of the CAD's points to the closest point of the other
// pose's (ADD-S), in millimeters.
export function addSMm(poseA, poseB, points) {
  const apply = (pose, p) => [0, 1, 2].map(
      (r) => pose[r][0] * p[0] + pose[r][1] * p[1] + pose[r][2] * p[2] + pose[r][3]);
  const a = points.map((p) => apply(poseA, p));
  const b = points.map((p) => apply(poseB, p));
  let sum = 0;
  for (const p of a) {
    let best = Infinity;
    for (const q of b) best = Math.min(best, (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2);
    sum += Math.sqrt(best);
  }
  return sum / a.length * 1000;
}

// The perspective projection (OpenGL clip space) of a pinhole camera with
// camera matrix k, for an image of width x height, as a column-major array.
export function projectionFromIntrinsics(k, width, height, near, far) {
  const [fx, cx, fy, cy] = [k[0][0], k[0][2], k[1][1], k[1][2]];
  // Rows of the matrix; OpenGL looks along -z with y up, the image's rows go
  // down, so cy flips.
  const rows = [
    [2 * fx / width, 0, 1 - 2 * cx / width, 0],
    [0, 2 * fy / height, 2 * cy / height - 1, 0],
    [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
    [0, 0, -1, 0],
  ];
  const out = [];
  for (let c = 0; c < 4; ++c) for (let r = 0; r < 4; ++r) out.push(rows[r][c]);
  return out;
}

// Decodes a mask packed with numpy.packbits (row-major, MSB first) and
// base64-encoded into a Uint8Array of 0/1 per pixel.
export function decodeMask(base64, width, height) {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < mask.length; ++i) {
    mask[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1;
  }
  return mask;
}

// Base64 of the bytes of a typed array.
export function toBase64(typed) {
  const u8 = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  }
  return btoa(s);
}
