import { useRef } from "react"

import { Trace, useTrace } from "@/model/trace"

const ink = "rgba(244, 236, 214, 0.92)"
const faint = "rgba(244, 236, 214, 0.28)"
const miss = "#e879f9"

function coeffs(alpha: number, beta: number) {
  const a2 = alpha / 2
  const b1 = 1 - a2
  const b2 = -2 * a2
  const b3 = beta - 2 * a2
  const r2 = b2 - 3 * b1
  const r3 = b3 - 6 * b1
  const a5 = (r3 - 6 * r2) / 2
  const a4 = r2 - 2 * a5
  const a3 = b1 - a4 - a5
  return [0, 0, a2, a3, a4, a5]
}

function sample(a: number[], n: number, deriv: 0 | 2) {
  const pts: [number, number][] = []
  for (let i = 0; i <= n; i += 1) {
    const t = i / n
    let y = 0
    if (deriv === 0) {
      let p = 1
      for (const c of a) {
        y += c * p
        p *= t
      }
    } else {
      y = 2 * a[2] + 6 * a[3] * t + 12 * a[4] * t * t + 20 * a[5] * t * t * t
    }
    pts.push([t, y])
  }
  return pts
}

function path(pts: [number, number][], X: (t: number) => number, Y: (y: number) => number) {
  return pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("")
}

const fans = [coeffs(0, 20), coeffs(0, -16), coeffs(8, 0)]
const unique = coeffs(0, 0)
const cubic = [0, 0, 3, -2]

function Xpos(t: number) {
  return 56 + t * 250
}
function Ypos(q: number) {
  return 168 - ((q + 0.2) / 1.6) * 140
}
function Xacc(t: number) {
  return 390 + t * 230
}
function Yacc(a: number) {
  return 168 - ((a + 12) / 28) * 140
}

export function Quintic() {
  const root = useRef<HTMLDivElement>(null)
  useTrace(root, [])
  const quinticQ = path(sample(unique, 48, 0), Xpos, Ypos)
  const quinticA = path(sample(unique, 48, 2), Xacc, Yacc)
  const cubicQ = path(
    sample(cubic, 48, 0).map(([t]) => [t, 3 * t * t - 2 * t * t * t] as [number, number]),
    Xpos,
    Ypos,
  )
  const cubicA = path(
    Array.from({ length: 49 }, (_, i) => [i / 48, 6 - 12 * (i / 48)] as [number, number]),
    Xacc,
    Yacc,
  )

  return (
    <div ref={root} className="size-full">
      <svg viewBox="0 0 680 250" className="size-full" role="img" aria-label="One quintic meets six boundary conditions. A cubic cannot.">
        <title>One quintic meets six boundary conditions. A cubic cannot.</title>
        <Axis x={56} y={Ypos(0)} w={250} />
        <Axis x={390} y={Yacc(0)} w={230} />
        {fans.map((a, index) => (
          <path key={index} d={path(sample(a, 40, 0), Xpos, Ypos)} fill="none" stroke={faint} strokeWidth="1.2" />
        ))}
        <path d={cubicQ} fill="none" stroke={miss} strokeWidth="1.3" strokeDasharray="4 4" />
        <Trace id="quintic-q" d={quinticQ} stroke={ink} width={2} dashes={null} />
        <path d={cubicA} fill="none" stroke={miss} strokeWidth="1.3" strokeDasharray="4 4" />
        <Trace id="quintic-a" d={quinticA} stroke={ink} width={2} dashes={null} />
        <Pins x={56} />
        <Pins x={306} />
        <circle cx={Xacc(0)} cy={Yacc(0)} r="3.5" fill={ink} />
        <circle cx={Xacc(1)} cy={Yacc(0)} r="3.5" fill={ink} />
        <circle cx={Xacc(0)} cy={Yacc(6)} r="3" fill={miss} />
        <circle cx={Xacc(1)} cy={Yacc(-6)} r="3" fill={miss} />
        <text x={56} y={28} className="fill-muted-foreground font-mono" fontSize="10">
          position
        </text>
        <text x={390} y={28} className="fill-muted-foreground font-mono" fontSize="10">
          acceleration
        </text>
      </svg>
    </div>
  )
}

function Axis({ x, y, w }: { x: number; y: number; w: number }) {
  return <line x1={x} x2={x + w} y1={y} y2={y} className="stroke-foreground/25" />
}

function Pins({ x }: { x: number }) {
  const labels = ["q", "qd", "qdd"]
  return (
    <>
      {labels.map((label, index) => (
        <g key={label}>
          <circle cx={x} cy={196 + index * 16} r="3.5" fill={ink} />
          <text x={x + 10} y={200 + index * 16} className="fill-muted-foreground font-mono" fontSize="10">
            {label}
          </text>
        </g>
      ))}
    </>
  )
}
