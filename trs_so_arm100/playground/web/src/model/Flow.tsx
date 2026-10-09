import { useEffect, useRef } from "react"

import { attachFeed } from "@/use-playground"

import type { BlockId } from "@/model/facts"

const cream = "rgba(244, 236, 214, 0.92)"
const magenta = "#e879f9"

export function Flow({
  instruction,
  onOpen,
}: {
  instruction: string
  onOpen: (id: BlockId) => void
}) {
  return (
    <div className="flex h-full min-h-[28rem] flex-col gap-3 px-5 py-3 lg:px-7">
      <div className="flex shrink-0 items-end justify-between gap-4">
        <div>
          <h1 className="font-serif text-2xl tracking-tight">SmolVLA</h1>
          <p className="mt-1 max-w-xl text-[13px] text-muted-foreground">
            Cameras and the sentence are encoded once. The action expert reads that cache and turns noise into joint targets.
          </p>
        </div>
        <p className="hidden font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase sm:block">
          Select a block
        </p>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[10.5rem_4.25rem_minmax(0,1fr)_5.25rem] items-stretch gap-2 lg:gap-3">
        <Inputs instruction={instruction} onOpen={onOpen} />
        <Tokens onOpen={onOpen} />
        <div className="flex min-h-0 min-w-0 flex-col gap-2">
          <Panel id="language" count="16" title="SmolVLM" tone="cool" onOpen={onOpen} />
          <Panel id="expert" count="16" title="Action expert" tone="warm" tokens onOpen={onOpen} />
        </div>
        <Output />
      </div>
    </div>
  )
}

function Inputs({
  instruction,
  onOpen,
}: {
  instruction: string
  onOpen: (id: BlockId) => void
}) {
  return (
    <div className="flex min-h-0 flex-col gap-2">
      <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">Image encoders</p>
      <button type="button" onClick={() => onOpen("vision")} className="flex min-h-0 flex-1 flex-col gap-1.5 text-left">
        <Camera feed={0} />
        <Camera feed={1} />
        <span className="font-mono text-[10px] tracking-[0.14em] text-sky-300/90 uppercase">SigLIP · 12 × 768</span>
      </button>
      <p className="line-clamp-2 font-serif text-[15px] leading-snug">{instruction || "Waiting for the instruction"}</p>
      <Strip label="State" note="32-wide, one token" bar="bg-[linear-gradient(90deg,#f0a8a0_0_38%,#f0a8a066_38%_70%,#f0a8a028_70%)]" />
      <Strip label="Action chunk" note="50 steps of noise" bar="bg-[linear-gradient(90deg,#5eead4,#a3e635,#facc15)]" />
    </div>
  )
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
        alt={feed === 0 ? "camera1" : "camera2"}
        draggable={false}
        className="size-full object-cover"
      />
    </span>
  )
}

function Strip({ label, note, bar }: { label: string; note: string; bar: string }) {
  return (
    <div className="shrink-0">
      <p className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">{label}</p>
      <div className={`mt-1 h-2.5 ${bar}`} />
      <p className="mt-0.5 text-[11px] text-muted-foreground">{note}</p>
    </div>
  )
}

function Tokens({ onOpen }: { onOpen: (id: BlockId) => void }) {
  const kinds = [
    ...Array.from({ length: 9 }, () => "bg-teal-300/85"),
    ...Array.from({ length: 5 }, () => "bg-orange-300/85"),
    "bg-rose-300",
    ...Array.from({ length: 6 }, () => "bg-lime-300/75"),
  ]
  return (
    <button type="button" onClick={() => onOpen("connector")} className="flex h-full min-h-0 items-stretch gap-1.5 py-1">
      <span className="shrink-0 self-center font-mono text-[10px] tracking-[0.12em] text-muted-foreground uppercase [writing-mode:vertical-rl]">
        Embeddings
      </span>
      <span className="flex min-h-0 min-w-0 flex-1 flex-col gap-1">
        <span className="shrink-0 text-center font-mono text-[10px] text-muted-foreground">960</span>
        <span className="flex min-h-0 flex-1 flex-col justify-between gap-px overflow-hidden">
          {kinds.map((tone, index) => (
            <span key={index} className={`min-h-px flex-1 ${tone}`} />
          ))}
        </span>
      </span>
    </button>
  )
}

function Panel({
  id,
  count,
  title,
  tone,
  tokens = false,
  onOpen,
}: {
  id: BlockId
  count: string
  title: string
  tone: "cool" | "warm"
  tokens?: boolean
  onOpen: (id: BlockId) => void
}) {
  const ring = tone === "cool" ? "border-sky-400/55" : "border-amber-300/55"
  const ink = tone === "cool" ? "text-sky-300" : "text-amber-200"
  const labels = tokens ? ["t0", "t1", "t2", "t3", "t4", "t5"] : undefined
  return (
    <button
      type="button"
      onClick={() => onOpen(id)}
      className={`relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-md border ${ring} px-2 pt-3 pb-1.5 text-left hover:bg-foreground/[0.03]`}
    >
      <span className="absolute top-0 left-1/2 -translate-x-1/2 bg-background px-2 font-mono text-[11px] text-foreground">
        {count}
      </span>
      <span className="flex min-h-0 flex-1 items-center justify-around gap-1">
        <Block labels={labels} />
        <Block labels={labels} className="hidden xl:flex" />
        <span className="shrink-0 font-mono text-base tracking-[0.25em] text-muted-foreground">···</span>
        <Block labels={labels} cells={tokens ? crossHeat : undefined} />
      </span>
      <span className={`shrink-0 text-center font-mono text-[11px] tracking-[0.18em] uppercase ${ink}`}>{title}</span>
    </button>
  )
}

function Block({
  labels,
  cells,
  className = "",
}: {
  labels?: string[]
  cells?: number[]
  className?: string
}) {
  return (
    <span className={`min-w-0 flex-col items-center justify-center gap-1 ${className || "flex"}`}>
      {labels && (
        <span className="flex gap-1.5">
          {labels.map((label) => (
            <span key={label} className="font-mono text-[8px] text-muted-foreground">
              {label}
            </span>
          ))}
        </span>
      )}
      <span className="flex items-center gap-1.5">
        <Heat cells={cells ?? selfHeat} />
        <Mlp />
      </span>
      <span className="font-mono text-[8px] tracking-[0.14em] text-muted-foreground uppercase">Transformer block</span>
    </span>
  )
}

const selfHeat = [0.95, 0.55, 0.2, 0.08, 0.15, 0.85, 0.4, 0.12, 0.25, 0.7, 0.9, 0.35, 0.1, 0.45, 0.75, 0.3, 0.2, 0.6, 0.4, 0.85, 0.15, 0.5, 0.25, 0.7]
const crossHeat = [0.8, 0.9, 0.7, 0.4, 0.55, 0.35, 0.75, 0.85, 0.6, 0.45, 0.3, 0.5, 0.9, 0.65, 0.4, 0.7, 0.35, 0.8, 0.55, 0.25, 0.6, 0.85, 0.4, 0.7]

function Heat({ cells }: { cells: number[] }) {
  return (
    <span className="grid h-16 w-11 grid-cols-4 grid-rows-6 gap-px sm:h-20" aria-hidden="true">
      {cells.map((value, index) => (
        <span key={index} style={{ background: magenta, opacity: 0.12 + value * 0.88 }} />
      ))}
    </span>
  )
}

function Mlp() {
  const columns = [5, 4, 6, 4]
  const nodes = columns.flatMap((count, column) =>
    Array.from({ length: count }, (_, row) => ({
      column,
      row,
      y: ((row + 0.5) / count) * 78 + 4,
      x: 8 + column * 22,
    })),
  )
  const edges: { x1: number; y1: number; x2: number; y2: number }[] = []
  for (let column = 0; column < columns.length - 1; column += 1) {
    const left = nodes.filter((node) => node.column === column)
    const right = nodes.filter((node) => node.column === column + 1)
    for (const a of left) {
      for (const b of right) edges.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y })
    }
  }
  return (
    <svg viewBox="0 0 82 88" className="h-16 w-14 sm:h-20" aria-hidden="true">
      {edges.map((edge, index) => (
        <line key={index} {...edge} stroke={cream} strokeWidth="0.45" opacity="0.55" />
      ))}
      {nodes.map((node) => (
        <circle key={`${node.column}-${node.row}`} cx={node.x} cy={node.y} r="2.4" fill="none" stroke={cream} strokeWidth="0.9" />
      ))}
    </svg>
  )
}

function Output() {
  return (
    <div className="flex flex-col items-center justify-center gap-2">
      <span className="text-lg text-muted-foreground" aria-hidden="true">
        →
      </span>
      <div className="h-9 w-full bg-[linear-gradient(#5eead4_0_34%,#a3e635_34%_66%,#facc15_66%)]" />
      <p className="text-center font-mono text-[10px] tracking-[0.16em] uppercase">Actions</p>
      <p className="text-center text-[11px] leading-snug text-muted-foreground">50 × 6</p>
    </div>
  )
}
