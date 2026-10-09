import { useGSAP } from "@gsap/react"
import gsap from "gsap"
import { useEffect, useRef, useState } from "react"

import { PegPath, type PegMode } from "@/model/PegPath"
import { Quintic } from "@/model/Quintic"
import { ThrowPath, type ThrowMode } from "@/model/ThrowPath"
import { attachFeed } from "@/use-playground"

gsap.registerPlugin(useGSAP)

type Stage = { id: string; title: string; meta: string; caption: string }

const openarm: Stage[] = [
  { id: "poses", title: "Poses", meta: "Ball and bin", caption: "The aim is the bin centre, on the plane the ball has to cross." },
  { id: "yaw", title: "Grasp yaw", meta: "36 candidates", caption: "The fingers sit clear of the other balls. The open side faces the bin." },
  { id: "waypoints", title: "Waypoints", meta: "Elbow up first", caption: "Elbow up, then hover, grasp, lift, wind-up, release." },
  { id: "quintic", title: "Quintic", meta: "Six pins", caption: "Six pins, six coefficients, one curve. The cubic misses zero acceleration." },
  { id: "release", title: "Release", meta: "One flight", caption: "The arrow is the release velocity. The arc meets the aim plane." },
  { id: "shooting", title: "Shooting", meta: "The miss", caption: "Faint arcs miss. The dot is the aim, sliding onto the bin." },
]

const peg: Stage[] = [
  { id: "layout", title: "Layout", meta: "On the table", caption: "The dashed box is the reach. The ring is the smallest gap that still fits." },
  { id: "yaw", title: "Yaw", meta: "Around one heading", caption: "Ticks walk away from the preferred heading. The long mark is the one that fits." },
  { id: "reach", title: "Reach", meta: "Hover to carry", caption: "Hover, down to the peg, up, then across to the socket." },
  { id: "grasp", title: "Grasp", meta: "Fully open", caption: "Fully open clears the peg. The partial opening does not." },
  { id: "carry", title: "Carry", meta: "Level", caption: "The peg stays level, its bottom above the table." },
  { id: "seat", title: "Seat", meta: "Down the hole", caption: "Dots every centimetre, tighter near the hole. The ghost column is the miss." },
]

const specs = {
  openarm: {
    title: "Throw expert",
    lede: "The grasp point through the throw, then the ball.",
    stages: openarm,
  },
  peg: {
    title: "Insert expert",
    lede: "The grasp point from the peg into the hole.",
    stages: peg,
  },
} as const

export function ExpertView({
  robot,
  instruction,
}: {
  robot: "openarm" | "peg"
  instruction: string
}) {
  const root = useRef<HTMLDivElement>(null)
  const spec = specs[robot]
  const [stageId, setStageId] = useState<string | null>(null)
  const stage = spec.stages.find((item) => item.id === stageId) ?? null

  useEffect(() => {
    if (stageId == null) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setStageId(null)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [stageId])

  useGSAP(
    () => {
      const motion = gsap.matchMedia()
      motion.add("(prefers-reduced-motion: no-preference)", () => {
        gsap.to("[data-model]", { y: 0, duration: 0.45, stagger: 0.05, ease: "power2.out" })
      })
    },
    { scope: root, dependencies: [stageId, robot] },
  )

  return (
    <div ref={root} className="absolute inset-0 z-20 flex flex-col bg-background pt-20 pl-36">
      {stage != null && (
        <div className="flex shrink-0 items-center px-7 pb-2">
          <button
            type="button"
            onClick={() => setStageId(null)}
            className="border border-border px-3 py-1.5 font-mono text-[11px] tracking-[0.14em] text-foreground uppercase hover:bg-card"
          >
            ← Plan
          </button>
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-auto">
        {stage == null ? (
          <Overview robot={robot} spec={spec} instruction={instruction} onOpen={setStageId} />
        ) : (
          <Detail robot={robot} stage={stage} />
        )}
      </div>
    </div>
  )
}

function Overview({
  robot,
  spec,
  instruction,
  onOpen,
}: {
  robot: "openarm" | "peg"
  spec: (typeof specs)["openarm"] | (typeof specs)["peg"]
  instruction: string
  onOpen: (id: string) => void
}) {
  return (
    <div className="flex h-full min-h-[28rem] flex-col gap-3 px-5 py-3 lg:px-7">
      <div data-model className="flex shrink-0 items-end justify-between gap-4">
        <div>
          <h1 className="font-serif text-2xl tracking-tight">{spec.title}</h1>
          <p className="mt-1 max-w-xl text-[13px] text-muted-foreground">{spec.lede}</p>
        </div>
        <p className="hidden font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase sm:block">
          Select a stretch
        </p>
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-[10.5rem_minmax(0,1fr)] items-stretch gap-3">
        <div data-model className="flex min-h-0 flex-col gap-2">
          <Camera feed={0} />
          <Camera feed={1} />
          <p className="line-clamp-2 shrink-0 font-serif text-[15px] leading-snug">{instruction || "Planning the expert"}</p>
        </div>
        <figure data-model className="flex min-h-0 flex-col gap-2">
          <div className="min-h-72 flex-1 border border-border">
            {robot === "openarm" ? (
              <ThrowPath mode="overview" onSelect={onOpen} />
            ) : (
              <PegPath mode="overview" onSelect={onOpen} />
            )}
          </div>
          <figcaption className="text-[13px] text-muted-foreground">Each stretch opens that part of the plan.</figcaption>
        </figure>
      </div>
    </div>
  )
}

function Detail({ robot, stage }: { robot: "openarm" | "peg"; stage: Stage }) {
  return (
    <div className="mx-auto flex min-h-full max-w-6xl flex-col gap-4 px-7 py-6">
      <div data-model className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
        <h1 className="font-serif text-2xl tracking-tight">{stage.title}</h1>
        <p className="text-[13px] text-muted-foreground">{stage.meta}</p>
      </div>
      <figure data-model className="flex flex-col gap-3">
        <div className="h-[min(26rem,58vh)] border border-border">
          <StageFigure robot={robot} id={stage.id} />
        </div>
        <figcaption className="max-w-xl text-[13px] text-muted-foreground">{stage.caption}</figcaption>
      </figure>
    </div>
  )
}

function StageFigure({ robot, id }: { robot: "openarm" | "peg"; id: string }) {
  if (id === "quintic") return <Quintic />
  if (robot === "openarm") return <ThrowPath mode={id as ThrowMode} />
  return <PegPath mode={id as PegMode} />
}

function Camera({ feed }: { feed: 0 | 1 }) {
  const image = useRef<HTMLImageElement>(null)
  useEffect(() => {
    attachFeed(feed, image.current)
  }, [feed])
  return (
    <span className="block min-h-0 flex-1 overflow-hidden border border-foreground/15">
      <img
        ref={image}
        data-feed={feed}
        alt={feed === 0 ? "front camera" : "wrist camera"}
        draggable={false}
        className="size-full object-cover"
      />
    </span>
  )
}
