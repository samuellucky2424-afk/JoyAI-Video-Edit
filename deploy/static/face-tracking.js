// Shared, model-independent geometry and exact-frame result matching.
export function normalizeLipPoints(points, eyeA, eyeB, aspect = 1) {
  if (!eyeA || !eyeB || !Number.isFinite(aspect) || aspect <= 0) return null;
  const dx = (eyeB.x - eyeA.x) * aspect;
  const dy = eyeB.y - eyeA.y;
  const distance = Math.hypot(dx, dy);
  if (!Number.isFinite(distance) || distance < 1e-6) return null;
  const cx = (eyeA.x + eyeB.x) * aspect / 2;
  const cy = (eyeA.y + eyeB.y) / 2;
  const c = dx / distance, s = dy / distance;
  const normalized = points.map(point => {
    const x = point.x * aspect - cx, y = point.y - cy;
    return [(c * x + s * y) / distance, (-s * x + c * y) / distance];
  });
  return normalized.every(p => p.every(Number.isFinite)) ? normalized : null;
}

export function expressionMotion(previous, current) {
  if (!previous || !current || !current.length || previous.length !== current.length) return 0;
  const width = Math.max(...current.map(p => p[0])) - Math.min(...current.map(p => p[0]));
  if (!Number.isFinite(width) || width < 1e-6) return 0;
  const distance = current.reduce((sum, p, i) => (
    sum + Math.hypot(p[0] - previous[i][0], p[1] - previous[i][1])
  ), 0);
  return distance / current.length / width;
}

export class FrameResultBroker {
  constructor({ budgetMs = 25, setTimer = setTimeout, clearTimer = clearTimeout } = {}) {
    this.budgetMs = budgetMs;
    this.setTimer = setTimer;
    this.clearTimer = clearTimer;
    this.nextId = 0;
    this.active = null;
  }

  request(frame, send) {
    // A timed-out worker stays busy until its reply arrives: never queue work
    // behind slow inference, and never attach that late reply to a newer frame.
    if (this.active) return Promise.resolve(null);
    return new Promise(resolve => {
      const job = { ...frame, requestId: ++this.nextId, resolve };
      this.active = job;
      job.timer = this.setTimer(() => resolve(null), this.budgetMs);
      Promise.resolve().then(() => {
        if (this.active === job) return send(job);
      }).catch(() => {
        if (this.active === job) this.reset();
      });
    });
  }

  matches(result) {
    return Boolean(this.active && result.requestId === this.active.requestId
      && result.epoch === this.active.epoch && result.captureSeq === this.active.captureSeq);
  }

  receive(result) {
    if (!this.matches(result)) return false;
    const job = this.active;
    this.active = null;
    this.clearTimer(job.timer);
    job.resolve(result.type === "result" ? result : null);
    return true;
  }

  reset() {
    if (!this.active) return;
    this.clearTimer(this.active.timer);
    this.active.resolve(null);
    this.active = null;
  }
}
