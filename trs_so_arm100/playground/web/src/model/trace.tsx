import { useGSAP } from "@gsap/react"
import gsap from "gsap"
import type { RefObject } from "react"

gsap.registerPlugin(useGSAP)

/** Reveal dotted strokes by drawing a mask. Reduced motion shows the finished path. */
export function useTrace(scope: RefObject<Element | null>, deps: unknown[]) {
  useGSAP(
    () => {
      const nodes = scope.current?.querySelectorAll<SVGPathElement>("[data-trace]")
      if (!nodes?.length) return
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      nodes.forEach((path) => {
        const length = path.getTotalLength()
        gsap.set(path, {
          strokeDasharray: length,
          strokeDashoffset: reduce ? 0 : length,
        })
      })
      if (reduce) return
      gsap.to(nodes, {
        strokeDashoffset: 0,
        duration: 1.35,
        stagger: 0.16,
        ease: "power2.inOut",
      })
    },
    { scope, dependencies: deps },
  )
}

export function Trace({
  id,
  d,
  stroke,
  width = 1.75,
  dashes = "2.2 5.5",
}: {
  id: string
  d: string
  stroke: string
  width?: number
  dashes?: string | null
}) {
  const pattern = dashes ?? "3 6"
  return (
    <>
      <path d={d} fill="none" stroke={stroke} strokeWidth={width} strokeDasharray={pattern} strokeLinecap="round" opacity={0.28} />
      <mask id={id}>
        <path data-trace d={d} fill="none" stroke="white" strokeWidth={width + 3} strokeLinecap="round" />
      </mask>
      <path
        d={d}
        fill="none"
        stroke={stroke}
        strokeWidth={width}
        strokeDasharray={pattern}
        strokeLinecap="round"
        mask={`url(#${id})`}
      />
    </>
  )
}

export function Hit({
  d,
  id,
  label,
  onSelect,
}: {
  d: string
  id: string
  label: string
  onSelect?: (id: string) => void
}) {
  if (!onSelect) return null
  return (
    <path
      d={d}
      fill="none"
      stroke="transparent"
      strokeWidth={22}
      className="cursor-pointer"
      role="button"
      tabIndex={0}
      aria-label={label}
      onClick={() => onSelect(id)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault()
          onSelect(id)
        }
      }}
    />
  )
}
