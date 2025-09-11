export function sigmoid (x: number): number {
  return 1 / (1 + Math.exp(-x))
}

export function distance (p1: { x: number, y: number }, p2: { x: number, y: number }) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y)
}

export function angle (ct: { x: number, y: number }, pt: { x: number, y: number }) {
  const dy    = pt.y - ct.y
  const dx    = pt.x - ct.x
  const theta = (Math.atan2(dy, dx) * 180 / Math.PI)

  return (360 - theta) % 360
}

export function clamp (value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}
