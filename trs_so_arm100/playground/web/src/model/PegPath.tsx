import { useRef } from "react"

import { Hit, Trace, useTrace } from "@/model/trace"

export type PegMode = "overview" | "layout" | "yaw" | "reach" | "grasp" | "carry" | "seat"

type Pt = [number, number]

const hover: Pt = [0.22, 0.55]
const grasp: Pt = [0.22, 0.47]
const lift: Pt = [0.22, 0.59]
const over: Pt = [0.48, 0.59]
const slow: Pt = [0.48, 0.53]
const seat: Pt = [0.48, 0.46]

const reach = [hover, grasp, lift]
const columnFast = [over, slow]
const columnSlow = [slow, seat]

function sx(x: number) {
  return 64 + ((x - 0.05) / 0.62) * 590
}
function sy(z: number) {
  return 336 - ((z - 0.32) / 0.4) * 292
}
function line(pts: Pt[]) {
  return pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("")
}

const ink = "rgba(244, 236, 214, 0.92)"
const aimInk = "#e879f9"

export function PegPath({
  mode,
  onSelect,
}: {
  mode: PegMode
  onSelect?: (id: string) => void
}) {
  if (mode === "layout") return <Layout />
  if (mode === "yaw") return <Yaw />
  if (mode === "grasp") return <Grasp />
  return <Side mode={mode} onSelect={onSelect} />
}

function Side({ mode, onSelect }: { mode: PegMode; onSelect?: (id: string) => void }) {
  const root = useRef<HTMLDivElement>(null)
  useTrace(root, [mode])
  const overview = mode === "overview"
  const lit = (part: PegMode) => overview || mode === part

  return (
    <div ref={root} className="size-full">
      <svg viewBox="0 0 720 380" className="size-full" role="img" aria-label="Insert plan, side view">
        <title>Insert plan, side view</title>
        <line x1={sx(0.06)} x2={sx(0.64)} y1={sy(0.4)} y2={sy(0.4)} className="stroke-foreground/30" strokeWidth="2" />
        <g opacity={lit("layout") ? 1 : 0.3}>
          <rect x={sx(0.2) - 6} y={sy(0.49)} width="12" height={sy(0.4) - sy(0.49)} fill="#d8c4a0" />
          <rect x={sx(0.44)} y={sy(0.46)} width={sx(0.52) - sx(0.44)} height={sy(0.4) - sy(0.46)} fill="none" stroke="#7aa2e3" strokeWidth="1.5" />
          <line x1={sx(0.455)} x2={sx(0.505)} y1={sy(0.46)} y2={sy(0.46)} stroke="#7aa2e3" strokeWidth="1" strokeDasharray="2 3" />
        </g>
        {(overview || mode === "reach") && (
          <g opacity={lit("reach") ? 1 : 0.3}>
            <Trace id={`peg-reach-${mode}`} d={line(reach)} stroke={ink} />
            {overview && <Hit d={line(reach)} id="reach" label="Reach" onSelect={onSelect} />}
            {[hover, grasp, lift].map((p) => (
              <circle key={p.join()} cx={sx(p[0])} cy={sy(p[1])} r="3" fill={ink} />
            ))}
          </g>
        )}
        {(overview || mode === "carry") && (
          <g opacity={lit("carry") ? 1 : 0.35}>
            <Trace id={`peg-carry-${mode}`} d={line([lift, over])} stroke={ink} />
            {overview && <Hit d={line([lift, over])} id="carry" label="Carry" onSelect={onSelect} />}
            {mode === "carry" && (
              <>
                <line x1={sx(0.22)} x2={sx(0.48)} y1={sy(0.445)} y2={sy(0.445)} stroke={ink} strokeWidth="1" strokeDasharray="2 4" opacity="0.7" />
                <rect x={sx(0.24)} y={sy(0.49)} width={sx(0.46) - sx(0.24)} height={sy(0.445) - sy(0.49)} fill="#d8c4a0" opacity="0.85" />
              </>
            )}
          </g>
        )}
        {(overview || mode === "seat") && (
          <g opacity={lit("seat") ? 1 : 0.4}>
            {mode === "seat" && (
              <path d={line([[0.52, 0.59], [0.52, 0.46]])} fill="none" stroke={aimInk} strokeWidth="1.2" strokeDasharray="2 5" opacity="0.4" />
            )}
            <Trace id={`peg-fast-${mode}`} d={line(columnFast)} stroke={aimInk} dashes="4 5" />
            <Trace id={`peg-slow-${mode}`} d={line(columnSlow)} stroke={aimInk} dashes="1.2 3.2" />
            {overview && <Hit d={line([over, seat])} id="seat" label="Seat" onSelect={onSelect} />}
            {mode === "seat" && <SeatMarks />}
          </g>
        )}
        {overview && (
          <>
            <Hit d={`M${sx(0.2)},${sy(0.5)} L${sx(0.52)},${sy(0.4)}`} id="layout" label="Peg and socket" onSelect={onSelect} />
            <circle
              cx={sx(hover[0])}
              cy={sy(hover[1])}
              r="14"
              fill="transparent"
              className="cursor-pointer"
              role="button"
              tabIndex={0}
              aria-label="Yaw"
              onClick={() => onSelect?.("yaw")}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault()
                  onSelect?.("yaw")
                }
              }}
            />
            <circle
              cx={sx(grasp[0])}
              cy={sy(grasp[1])}
              r="12"
              fill="transparent"
              className="cursor-pointer"
              role="button"
              tabIndex={0}
              aria-label="Grasp"
              onClick={() => onSelect?.("grasp")}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault()
                  onSelect?.("grasp")
                }
              }}
            />
            <Labels onSelect={onSelect} />
          </>
        )}
      </svg>
    </div>
  )
}

function SeatMarks() {
  const dots: Pt[] = []
  for (let z = 0.59; z >= 0.46 - 0.001; z -= 0.01) dots.push([0.48, Number(z.toFixed(3))])
  return (
    <>
      {dots.map((p) => (
        <circle key={p[1]} cx={sx(p[0])} cy={sy(p[1])} r={p[1] < 0.535 ? 2.2 : 1.4} fill={aimInk} />
      ))}
      <line x1={sx(0.52)} y1={sy(0.52)} x2={sx(0.48)} y2={sy(0.52)} stroke={ink} strokeWidth="1" />
    </>
  )
}

function Labels({ onSelect }: { onSelect?: (id: string) => void }) {
  const marks = [
    { id: "layout", x: 0.34, z: 0.42, text: "layout" },
    { id: "yaw", x: 0.26, z: 0.57, text: "yaw" },
    { id: "reach", x: 0.14, z: 0.52, text: "reach" },
    { id: "grasp", x: 0.26, z: 0.45, text: "grasp" },
    { id: "carry", x: 0.34, z: 0.63, text: "carry" },
    { id: "seat", x: 0.52, z: 0.5, text: "seat" },
  ]
  return (
    <>
      {marks.map((mark) => (
        <text
          key={mark.id}
          x={sx(mark.x)}
          y={sy(mark.z)}
          className="cursor-pointer fill-muted-foreground font-mono"
          fontSize="10"
          role="button"
          tabIndex={0}
          onClick={() => onSelect?.(mark.id)}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault()
              onSelect?.(mark.id)
            }
          }}
        >
          {mark.text}
        </text>
      ))}
    </>
  )
}

function Layout() {
  const x0 = 90
  const y0 = 50
  const x1 = 550
  const y1 = 300
  const legal = { x: 180, y: 90, w: 250, h: 160 }
  const peg = { x: 250, y: 140 }
  const sock = { x: 360, y: 210 }
  return (
    <svg viewBox="0 0 640 360" className="size-full" role="img" aria-label="Legal table layout, from above">
      <title>Legal table layout, from above</title>
      <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" className="stroke-foreground/25" />
      <rect x={legal.x} y={legal.y} width={legal.w} height={legal.h} fill="none" stroke={ink} strokeWidth="1.2" strokeDasharray="3 5" />
      <circle cx={peg.x} cy={peg.y} r="10" fill="#d8c4a0" />
      <circle cx={sock.x} cy={sock.y} r="18" fill="none" stroke="#7aa2e3" strokeWidth="2" />
      <line x1={peg.x} y1={peg.y} x2={sock.x} y2={sock.y} stroke={aimInk} strokeWidth="1.2" strokeDasharray="2 4" />
      <circle cx={(peg.x + sock.x) / 2} cy={(peg.y + sock.y) / 2} r="54" fill="none" stroke={aimInk} strokeDasharray="2 5" opacity="0.7" />
    </svg>
  )
}

function Yaw() {
  const cx = 280
  const cy = 180
  const r = 120
  const prefer = -2.09
  const ticks = Array.from({ length: 24 }, (_, i) => {
    const k = Math.floor((i + 1) / 2)
    const sign = i % 2 === 0 ? 1 : -1
    return prefer + sign * (k * Math.PI) / 12
  })
  return (
    <svg viewBox="0 0 640 360" className="size-full" role="img" aria-label="Wrist yaw search">
      <title>Wrist yaw search</title>
      <circle cx={cx} cy={cy} r="14" fill="#d8c4a0" />
      {ticks.map((angle, index) => {
        const fit = index === 0
        return (
          <line
            key={angle}
            x1={cx + Math.cos(angle) * (r - (fit ? 22 : 10))}
            y1={cy + Math.sin(angle) * (r - (fit ? 22 : 10))}
            x2={cx + Math.cos(angle) * r}
            y2={cy + Math.sin(angle) * r}
            stroke={fit ? ink : "rgba(244,236,214,0.4)"}
            strokeWidth={fit ? 2.4 : 1}
          />
        )
      })}
      <line
        x1={cx}
        y1={cy}
        x2={cx + Math.cos(prefer) * r}
        y2={cy + Math.sin(prefer) * r}
        stroke={aimInk}
        strokeWidth="1.3"
      />
    </svg>
  )
}

function Grasp() {
  return (
    <svg viewBox="0 0 640 320" className="size-full" role="img" aria-label="Open fingers clear the peg. A partial opening does not.">
      <title>Open fingers clear the peg. A partial opening does not.</title>
      <FingerPair cx={170} gap={78} peg={36} />
      <FingerPair cx={470} gap={28} peg={36} />
    </svg>
  )
}

function FingerPair({ cx, gap, peg }: { cx: number; gap: number; peg: number }) {
  const top = 70
  const bot = 230
  const fits = gap > peg + 8
  return (
    <g>
      <rect x={cx - gap / 2 - 10} y={top} width="8" height={bot - top} rx="2" fill={ink} opacity="0.9" />
      <rect x={cx + gap / 2 + 2} y={top} width="8" height={bot - top} rx="2" fill={ink} opacity="0.9" />
      <circle cx={cx} cy={168} r={peg / 2} fill={fits ? "#d8c4a0" : "none"} stroke="#d8c4a0" strokeWidth="1.5" strokeDasharray={fits ? undefined : "3 4"} />
    </g>
  )
}
