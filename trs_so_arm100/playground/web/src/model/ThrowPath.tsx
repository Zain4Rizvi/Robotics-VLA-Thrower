import { useGSAP } from "@gsap/react"
import gsap from "gsap"
import { useRef } from "react"

import { Hit, Trace, useTrace } from "@/model/trace"

gsap.registerPlugin(useGSAP)

export type ThrowMode = "overview" | "poses" | "yaw" | "waypoints" | "release" | "shooting"

type Pt = [number, number]

const home: Pt = [0.02, 0.22]
const elbow: Pt = [0.07, 0.58]
const ready: Pt = [0.15, 0.5]
const hover: Pt = [0.24, 0.53]
const grasp: Pt = [0.24, 0.43]
const lift: Pt = [0.24, 0.63]
const wind: Pt = [0.18, 0.64]
const rel: Pt = [0.34, 0.7]
const aim: Pt = [0.62, 0.16]

const approach = [home, elbow, ready, hover, grasp, lift]
const windup = [lift, wind, rel]

function sx(x: number) {
  return 52 + ((x + 0.04) / 0.96) * 640
}
function sy(z: number) {
  return 340 - ((z + 0.02) / 0.9) * 304
}
function line(pts: Pt[]) {
  return pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("")
}

function flight(x1: number, z1 = 0.16, n = 28): Pt[] {
  const tFlight = 0.4
  const g = 9.81
  const vx = (x1 - rel[0]) / tFlight
  const vz = (z1 - rel[1]) / tFlight + 0.5 * g * tFlight
  const pts: Pt[] = []
  for (let i = 0; i <= n; i += 1) {
    const t = (tFlight * i) / n
    pts.push([rel[0] + vx * t, rel[1] + vz * t - 0.5 * g * t * t])
  }
  return pts
}

const arc = flight(aim[0])
const misses = [flight(0.5), flight(0.56)]

const ink = "rgba(244, 236, 214, 0.92)"
const dimInk = "rgba(244, 236, 214, 0.35)"
const aimInk = "#e879f9"

export function ThrowPath({
  mode,
  onSelect,
}: {
  mode: ThrowMode
  onSelect?: (id: string) => void
}) {
  if (mode === "yaw") return <ThrowYaw />
  return <ThrowSide mode={mode} onSelect={onSelect} />
}

function ThrowSide({
  mode,
  onSelect,
}: {
  mode: ThrowMode
  onSelect?: (id: string) => void
}) {
  const root = useRef<HTMLDivElement>(null)
  useTrace(root, [mode])
  const overview = mode === "overview"
  const showApproach = overview || mode === "waypoints"
  const showWind = overview || mode === "waypoints" || mode === "release"
  const showArc = overview || mode === "release" || mode === "shooting"
  const showPlane = overview || mode === "poses" || mode === "shooting" || mode === "release"
  const lit = (part: ThrowMode) => overview || mode === part

  return (
    <div ref={root} className="size-full">
      <svg viewBox="0 0 720 380" className="size-full" role="img" aria-label="Throw plan, side view">
        <title>Throw plan, side view</title>
        <defs>
          <marker id="throw-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7" fill={ink} />
          </marker>
        </defs>
        <Floor />
        <g opacity={lit("poses") ? 1 : 0.28}>
          <rect x={sx(0.08)} y={sy(0.4)} width={sx(0.42) - sx(0.08)} height={sy(0.36) - sy(0.4)} className="fill-foreground/10" />
          <circle cx={sx(0.18)} cy={sy(0.45)} r="7" fill="#e8a05a" />
          <circle cx={sx(0.24)} cy={sy(0.45)} r="7" fill="#d85a4a" />
          <circle cx={sx(0.31)} cy={sy(0.45)} r="7" fill="#6a9e7a" />
          <rect x={sx(0.56)} y={sy(0.13)} width={sx(0.68) - sx(0.56)} height={sy(0) - sy(0.13)} fill="none" stroke="#7aa2e3" strokeWidth="1.4" />
          <line x1={sx(0.56)} x2={sx(0.68)} y1={sy(0)} y2={sy(0)} stroke="#7aa2e3" strokeWidth="3" />
        </g>
        {showPlane && (
          <g opacity={lit("poses") || mode === "shooting" || mode === "release" ? 1 : 0.45}>
            <line
              x1={sx(-0.02)}
              x2={sx(0.86)}
              y1={sy(0.16)}
              y2={sy(0.16)}
              stroke={aimInk}
              strokeWidth="1"
              strokeDasharray="3 6"
            />
            <circle cx={sx(aim[0])} cy={sy(aim[1])} r="3.5" fill={aimInk} />
          </g>
        )}
        {showApproach && (
          <g opacity={lit("waypoints") ? 1 : 0.3}>
            <Trace id={`throw-approach-${mode}`} d={line(approach)} stroke={lit("waypoints") ? ink : dimInk} />
            {overview && <Hit d={line(approach)} id="waypoints" label="Waypoints" onSelect={onSelect} />}
          </g>
        )}
        {showWind && (
          <g opacity={lit("waypoints") || mode === "release" ? 1 : 0.3}>
            <Trace id={`throw-wind-${mode}`} d={line(windup)} stroke={ink} dashes="1.2 4" />
            {overview && <Hit d={line(windup)} id="quintic" label="Quintic between two waypoints" onSelect={onSelect} />}
          </g>
        )}
        {(overview || mode === "waypoints") &&
          [elbow, ready, hover, grasp, lift, wind, rel].map((p) => (
            <circle key={p.join()} cx={sx(p[0])} cy={sy(p[1])} r="3" fill={ink} />
          ))}
        {showArc && mode !== "shooting" && (
          <g opacity={lit("release") ? 1 : 0.35}>
            <Trace id={`throw-arc-${mode}`} d={line(arc)} stroke={aimInk} />
            {overview && <Hit d={line(arc)} id="release" label="Release arc" onSelect={onSelect} />}
            {mode === "release" && <Velocity />}
          </g>
        )}
        {mode === "shooting" && <Misses />}
        {overview && (
          <>
            <Hit d={line([[-0.02, 0.16], [0.86, 0.16]])} id="poses" label="Ball, bin, and aim plane" onSelect={onSelect} />
            <circle
              cx={sx(grasp[0])}
              cy={sy(grasp[1])}
              r="14"
              fill="transparent"
              className="cursor-pointer"
              role="button"
              tabIndex={0}
              aria-label="Grasp yaw"
              onClick={() => onSelect?.("yaw")}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault()
                  onSelect?.("yaw")
                }
              }}
            />
            <circle
              cx={sx(aim[0])}
              cy={sy(aim[1])}
              r="12"
              fill="transparent"
              className="cursor-pointer"
              role="button"
              tabIndex={0}
              aria-label="Shooting"
              onClick={() => onSelect?.("shooting")}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault()
                  onSelect?.("shooting")
                }
              }}
            />
          </>
        )}
        {overview && <Labels onSelect={onSelect} />}
        {mode === "waypoints" && <WaypointNames />}
      </svg>
    </div>
  )
}

function Floor() {
  return <line x1={sx(-0.04)} x2={sx(0.9)} y1={sy(0)} y2={sy(0)} className="stroke-foreground/25" strokeWidth="1" />
}

function Velocity() {
  const vx = (aim[0] - rel[0]) / 0.4
  const vz = (aim[1] - rel[1]) / 0.4 + 0.5 * 9.81 * 0.4
  const scale = 0.09
  const x2 = rel[0] + vx * scale
  const z2 = rel[1] + vz * scale
  return (
    <line
      x1={sx(rel[0])}
      y1={sy(rel[1])}
      x2={sx(x2)}
      y2={sy(z2)}
      stroke={ink}
      strokeWidth="1.6"
      markerEnd="url(#throw-arrow)"
    />
  )
}

function Misses() {
  return (
    <g>
      {misses.map((pts, index) => (
        <path
          key={index}
          d={line(pts)}
          fill="none"
          stroke={aimInk}
          strokeWidth="1.2"
          strokeDasharray="2 6"
          opacity={0.28 + index * 0.15}
        />
      ))}
      <Trace id="throw-hit" d={line(arc)} stroke={aimInk} />
      {misses.map((pts) => {
        const end = pts[pts.length - 1]
        return <circle key={end[0]} cx={sx(end[0])} cy={sy(0.16)} r="3" fill={aimInk} opacity="0.45" />
      })}
      <AimDot />
    </g>
  )
}

function AimDot() {
  const dot = useRef<SVGGElement>(null)
  useGSAP(() => {
    const dx = sx(aim[0]) - sx(0.5)
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches
    if (reduce) {
      gsap.set(dot.current, { x: dx })
      return
    }
    gsap.fromTo(dot.current, { x: 0 }, { x: dx, duration: 1.5, ease: "power2.inOut", delay: 0.35 })
  })
  return (
    <g ref={dot}>
      <circle cx={sx(0.5)} cy={sy(0.16)} r="4.5" fill={ink} />
    </g>
  )
}

function Labels({ onSelect }: { onSelect?: (id: string) => void }) {
  const marks: { id: string; x: number; z: number; text: string }[] = [
    { id: "poses", x: 0.78, z: 0.2, text: "poses" },
    { id: "yaw", x: 0.36, z: 0.5, text: "yaw" },
    { id: "waypoints", x: 0.0, z: 0.64, text: "waypoints" },
    { id: "quintic", x: 0.12, z: 0.74, text: "quintic" },
    { id: "release", x: 0.44, z: 0.76, text: "release" },
    { id: "shooting", x: 0.72, z: 0.22, text: "shooting" },
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

function WaypointNames() {
  const names: { p: Pt; text: string }[] = [
    { p: elbow, text: "elbow" },
    { p: ready, text: "ready" },
    { p: hover, text: "hover" },
    { p: grasp, text: "grasp" },
    { p: lift, text: "lift" },
    { p: wind, text: "wind" },
    { p: rel, text: "release" },
  ]
  return (
    <>
      {names.map((item) => (
        <text key={item.text} x={sx(item.p[0]) + 8} y={sy(item.p[1]) - 6} className="fill-muted-foreground font-mono" fontSize="10">
          {item.text}
        </text>
      ))}
    </>
  )
}

function ThrowYaw() {
  const cx = 210
  const cy = 180
  const r = 108
  const ticks = Array.from({ length: 36 }, (_, i) => (i / 36) * Math.PI * 2)
  const finger = -0.4
  return (
    <svg viewBox="0 0 640 360" className="size-full" role="img" aria-label="Grasp yaw, from above">
      <title>Grasp yaw, from above</title>
      <circle cx={cx} cy={cy} r="16" fill="#d85a4a" />
      <circle cx={cx - 70} cy={cy - 36} r="14" fill="#e8a05a" />
      <circle cx={cx + 62} cy={cy + 48} r="14" fill="#6a9e7a" />
      <circle cx={cx} cy={cy} r={r} fill="none" className="stroke-foreground/20" />
      {ticks.map((angle) => {
        const chosen = Math.abs(angle - (finger + Math.PI / 2)) < 0.09 || Math.abs(angle - (finger - Math.PI / 2)) < 0.09
        const inner = chosen ? r - 16 : r - 8
        return (
          <line
            key={angle}
            x1={cx + Math.cos(angle) * inner}
            y1={cy + Math.sin(angle) * inner}
            x2={cx + Math.cos(angle) * r}
            y2={cy + Math.sin(angle) * r}
            stroke={chosen ? ink : "rgba(244,236,214,0.45)"}
            strokeWidth={chosen ? 2 : 1}
          />
        )
      })}
      <line
        x1={cx + Math.cos(finger) * 28}
        y1={cy + Math.sin(finger) * 28}
        x2={cx + Math.cos(finger) * 150}
        y2={cy + Math.sin(finger) * 150}
        stroke={aimInk}
        strokeWidth="1.4"
        strokeDasharray="3 5"
      />
      <circle cx={520} cy={250} r="18" fill="none" stroke="#7aa2e3" strokeWidth="1.5" />
      <text x={496} y={278} className="fill-muted-foreground font-mono" fontSize="10">
        bin
      </text>
    </svg>
  )
}
