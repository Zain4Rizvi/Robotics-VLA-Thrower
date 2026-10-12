import { useEffect, useState } from "react"

import { cn } from "@/lib/utils"

const COLORS = ["#d8b4fe", "#67e8f9", "#f9a8d4", "#fcd34d"]

type Kind = "front" | "wrist" | "word" | "pad" | "state"

type Token = {
  kind: Kind
  label: string
  row?: number
  col?: number
}

type ExpertStep = {
  step: number
  self: number[][]
  scene: number[][]
  cross: number[][]
  chunkShare: number[]
}

type Snapshot = {
  id: string
  instruction: string
  tokens: Token[]
  vlmQueries: number[]
  vlm: number[][][]
  expert: ExpertStep[]
}

type Dump = {
  grid: number
  steps: number[]
  front: string
  wrist: string
  snapshots: Snapshot[]
}

type Mark = {
  n: number
  color: string
  label: string
  value: number
  cell?: number
  token?: number
}

function pct(value: number): string {
  return `${(value / 10).toFixed(1)}%`
}

function ink(value: number, max: number): string {
  const t = max > 0 ? Math.min(value / max, 1) : 0
  const alpha = 0.04 + 0.82 * t * t
  return `rgba(192, 132, 252, ${alpha.toFixed(3)})`
}

function region(row: number, col: number, grid: number): string {
  const vertical = row < grid * 0.4 ? "Upper" : row > grid * 0.6 ? "Lower" : "Middle"
  const horizontal = col < grid * 0.4 ? "left" : col > grid * 0.6 ? "right" : "center"
  return `${vertical} ${horizontal}`
}

function moment(id: string): string {
  return id === "green" ? "Green on red" : "Red on green"
}

function marksFor(tokens: Token[], weights: number[]): Mark[] {
  const groups: { label: string; value: number; cell?: number; token?: number }[] = []
  for (const kind of ["front", "wrist"] as const) {
    let sum = 0
    let best = -1
    let bestAt = -1
    tokens.forEach((token, index) => {
      if (token.kind !== kind) return
      const value = weights[index] ?? 0
      sum += value
      if (value > best) {
        best = value
        bestAt = index
      }
    })
    groups.push({
      label: kind === "front" ? "Front camera" : "Wrist camera",
      value: sum,
      cell: bestAt,
    })
  }
  tokens.forEach((token, index) => {
    if (token.kind !== "word" && token.kind !== "state") return
    groups.push({
      label: token.kind === "state" ? "Joints" : token.label,
      value: weights[index] ?? 0,
      token: index,
    })
  })
  return groups
    .filter((group) => group.value > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, 4)
    .map((group, index) => ({ ...group, n: index + 1, color: COLORS[index] ?? COLORS[0] }))
}

function imagePeak(tokens: Token[], weights: number[] | null): number {
  if (!weights) return 1
  let max = 1
  tokens.forEach((token, index) => {
    if (token.kind === "front" || token.kind === "wrist") max = Math.max(max, weights[index] ?? 0)
  })
  return max
}

function describe(tokens: Token[], index: number, grid: number, value: number, blocked: boolean): string {
  const token = tokens[index]
  if (!token) return ""
  if (token.kind === "front" || token.kind === "wrist") {
    const where = token.row != null && token.col != null ? region(token.row, token.col, grid) : token.label
    const camera = token.kind === "front" ? "Front" : "Wrist"
    return `${camera} · ${where} · ${pct(value)}`
  }
  if (token.kind === "state") {
    return blocked ? "Joints · 0% — the pictures and the words cannot see the joints" : `Joints · ${pct(value)}`
  }
  return `${token.label} · ${pct(value)}`
}

export function PrefixTrace({ block, layer }: { block: "language" | "expert"; layer: number }) {
  const [dump, setDump] = useState<Dump | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [id, setId] = useState("red")
  const [query, setQuery] = useState<number | null>(null)
  const [step, setStep] = useState(42)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    let live = true
    fetch("/attention/prefix.json")
      .then((response) => {
        if (!response.ok) throw new Error(`attention dump ${response.status}`)
        return response.json() as Promise<Dump>
      })
      .then((data) => {
        if (live) setDump(data)
      })
      .catch((reason: unknown) => {
        if (live) setError(reason instanceof Error ? reason.message : "attention dump failed")
      })
    return () => {
      live = false
    }
  }, [])

  useEffect(() => {
    setHover(null)
  }, [layer, id, query, step, block])

  if (error) return <p className="text-[13px] text-muted-foreground">{error}</p>
  if (!dump) return <p className="text-[13px] text-muted-foreground">Loading the two sentences.</p>

  const snap = dump.snapshots.find((item) => item.id === id) ?? dump.snapshots[0]
  const layerIndex = Math.min(Math.max(layer, 0), 15)

  return (
    <figure className="flex w-full min-w-0 flex-col gap-5" onMouseLeave={() => setHover(null)}>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="max-w-xl">
          <p className="font-serif text-xl tracking-tight text-foreground">
            {block === "language" ? "Click a word. See which tokens it reads." : "Click a step. See what that motion attends to."}
          </p>
          <p className="mt-2 text-[13px] leading-relaxed text-muted-foreground">
            {block === "language"
              ? "Both cameras, the sentence, and one joint token are a single row. Pick a token. The color is its attention at this layer, and the same color is the same score."
              : "The chunk is 50 steps of noise. Eight of them were measured. Even layers can also see earlier steps. Odd layers only read the scene SmolVLM already cached."}
          </p>
        </div>
        <div className="flex gap-1">
          {dump.snapshots.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setId(item.id)}
              className={
                item.id === snap.id
                  ? "bg-foreground px-2.5 py-1.5 font-mono text-[10px] text-background"
                  : "bg-foreground/10 px-2.5 py-1.5 font-mono text-[10px] text-muted-foreground"
              }
            >
              {moment(item.id)}
            </button>
          ))}
        </div>
      </div>

      {block === "language" ? (
        <Language
          dump={dump}
          snap={snap}
          layer={layerIndex}
          query={query}
          hover={hover}
          onQuery={setQuery}
          onHover={setHover}
        />
      ) : (
        <Expert
          dump={dump}
          snap={snap}
          layer={layerIndex}
          step={step}
          hover={hover}
          onStep={setStep}
          onHover={setHover}
        />
      )}
    </figure>
  )
}

function Language({
  dump,
  snap,
  layer,
  query,
  hover,
  onQuery,
  onHover,
}: {
  dump: Dump
  snap: Snapshot
  layer: number
  query: number | null
  hover: number | null
  onQuery: (index: number) => void
  onHover: (index: number | null) => void
}) {
  const slot = query == null ? -1 : snap.vlmQueries.indexOf(query)
  const weights = slot >= 0 ? snap.vlm[slot]?.[layer] : null
  const chips = snap.tokens
    .map((token, index) => ({ token, index }))
    .filter((item) => item.token.kind === "word" || item.token.kind === "state")
  const marks = weights ? marksFor(snap.tokens, weights) : []
  const peak = imagePeak(snap.tokens, weights)
  const focus = hover ?? query
  const blocked = focus != null && snap.tokens[focus]?.kind === "state" && query !== focus && (weights?.[focus] ?? 0) === 0

  return (
    <>
      <div className="flex flex-wrap gap-1">
        {chips.map(({ token, index }) => (
          <button
            key={index}
            type="button"
            aria-pressed={index === query}
            onClick={() => onQuery(index)}
            onMouseEnter={() => onHover(index)}
            onFocus={() => onHover(index)}
            className={cn(
              "px-2.5 py-1.5 font-mono text-[11px]",
              index === query ? "bg-foreground text-background" : "bg-foreground/10 text-foreground",
            )}
          >
            {token.kind === "state" ? "joints" : token.label}
          </button>
        ))}
      </div>
      <p className="min-h-5 font-mono text-[12px] text-foreground">
        {weights && focus != null ? (
          describe(snap.tokens, focus, dump.grid, weights[focus] ?? 0, blocked)
        ) : (
          <span className="text-muted-foreground">Click a word or the joints.</span>
        )}
      </p>
      <div className="grid min-w-0 items-start gap-4 md:grid-cols-2">
        <Camera
          title="Front camera"
          image={dump.front}
          tokens={snap.tokens}
          kind="front"
          grid={dump.grid}
          weights={weights}
          marks={marks}
          peak={peak}
          hover={hover}
          onHover={onHover}
        />
        <Camera
          title="Wrist camera"
          image={dump.wrist}
          tokens={snap.tokens}
          kind="wrist"
          grid={dump.grid}
          weights={weights}
          marks={marks}
          peak={peak}
          hover={hover}
          onHover={onHover}
        />
      </div>
      {weights && (
        <Bars tokens={snap.tokens} weights={weights} marks={marks} hover={hover} onHover={onHover} onPick={onQuery} query={query} />
      )}
      <Legend marks={marks} onHover={onHover} />
      <p className="max-w-2xl text-[13px] leading-relaxed text-muted-foreground">
        {query != null && snap.tokens[query]?.kind === "state"
          ? "The joint token can read the pictures and the words. They cannot read it back."
          : "The pictures and the words cannot see the joints. Nothing in this block sees the action chunk."}
      </p>
    </>
  )
}

function Expert({
  dump,
  snap,
  layer,
  step,
  hover,
  onStep,
  onHover,
}: {
  dump: Dump
  snap: Snapshot
  layer: number
  step: number
  hover: number | null
  onStep: (step: number) => void
  onHover: (index: number | null) => void
}) {
  const chosen = dump.steps.includes(step) ? step : dump.steps[dump.steps.length - 1]
  const record = snap.expert.find((item) => item.step === chosen) ?? snap.expert[0]
  const half = Math.floor(layer / 2)
  const selfLayer = layer % 2 === 0
  const self = record.self[half] ?? []
  const share = record.chunkShare[half] ?? 0
  const scene = record.scene[half] ?? []
  const rest = 1000 - share
  const weights = selfLayer
    ? scene.map((value) => Math.round((value * rest) / 1000))
    : (record.cross[half] ?? [])
  const marks = marksFor(snap.tokens, weights)
  const peak = imagePeak(snap.tokens, weights)
  const focus = hover

  return (
    <>
      <Chunk
        step={record.step}
        measured={dump.steps}
        self={self}
        causal={selfLayer}
        onStep={onStep}
        onHover={onHover}
        hover={hover}
      />
      <p className="min-h-5 font-mono text-[12px] text-foreground">
        {focus != null && focus < snap.tokens.length ? (
          describe(snap.tokens, focus, dump.grid, weights[focus] ?? 0, false)
        ) : focus != null && focus >= 1000 ? (
          chunkReadout(focus - 1000, record.step, self, selfLayer)
        ) : (
          <span className="text-muted-foreground">Hover a camera square, a word, or a step.</span>
        )}
      </p>
      <div className="grid min-w-0 items-start gap-4 md:grid-cols-2">
        <Camera
          title="Front camera"
          image={dump.front}
          tokens={snap.tokens}
          kind="front"
          grid={dump.grid}
          weights={weights}
          marks={marks}
          peak={peak}
          hover={hover}
          onHover={onHover}
        />
        <Camera
          title="Wrist camera"
          image={dump.wrist}
          tokens={snap.tokens}
          kind="wrist"
          grid={dump.grid}
          weights={weights}
          marks={marks}
          peak={peak}
          hover={hover}
          onHover={onHover}
        />
      </div>
      <Bars tokens={snap.tokens} weights={weights} marks={marks} hover={hover} onHover={onHover} />
      <Legend marks={marks} onHover={onHover} />
      <p className="max-w-2xl text-[13px] leading-relaxed text-muted-foreground">
        {selfLayer
          ? `Step ${record.step + 1} can read itself and earlier steps. That is ${pct(share)} of this layer. The other ${pct(1000 - share)} reads the cached scene above. Later steps are blocked.`
          : `This layer does not read the other steps. All of step ${record.step + 1}’s attention is on the scene SmolVLM cached at layer ${layer + 1}.`}
      </p>
    </>
  )
}

function chunkReadout(index: number, query: number, self: number[], causal: boolean): string {
  if (!causal) return `Step ${index + 1} · this layer does not read other steps`
  if (index > query) return `Step ${index + 1} · blocked. This step cannot see later steps.`
  return `Step ${index + 1} · ${pct(self[index] ?? 0)} of the attention that stays in the chunk`
}

function Chunk({
  step,
  measured,
  self,
  causal,
  hover,
  onStep,
  onHover,
}: {
  step: number
  measured: number[]
  self: number[]
  causal: boolean
  hover: number | null
  onStep: (step: number) => void
  onHover: (index: number | null) => void
}) {
  const known = new Set(measured)
  const peak = Math.max(...self, 1)
  return (
    <div>
      <p className="mb-2 font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
        {causal ? "Inside the chunk" : "Pick a step"}
      </p>
      <div className="grid grid-cols-[repeat(25,minmax(0,1fr))] gap-px sm:grid-cols-[repeat(50,minmax(0,1fr))]">
        {Array.from({ length: 50 }, (_, index) => {
          const live = known.has(index)
          const future = causal && index > step
          const hot = hover === 1000 + index
          const selected = index === step
          return (
            <button
              key={index}
              type="button"
              disabled={!live}
              aria-pressed={selected}
              aria-label={`Step ${index + 1}`}
              onClick={() => onStep(index)}
              onMouseEnter={() => onHover(1000 + index)}
              onFocus={() => onHover(1000 + index)}
              className={cn("h-8 min-w-0", !live && "cursor-default opacity-30")}
              style={{
                backgroundColor: causal && live ? ink(future ? 0 : (self[index] ?? 0), peak) : selected ? "rgba(244,240,232,0.9)" : "rgba(244,240,232,0.08)",
                backgroundImage: future
                  ? "repeating-linear-gradient(-45deg, transparent, transparent 2px, rgba(244,240,232,0.45) 2px, rgba(244,240,232,0.45) 3px)"
                  : undefined,
                outline: selected || hot ? "2px solid #f4f0e8" : undefined,
                outlineOffset: "-2px",
              }}
            />
          )
        })}
      </div>
      <p className="mt-1 font-mono text-[10px] text-muted-foreground">
        {causal
          ? "Steps 1–50. Filled buttons were measured. Hatched steps are in the future."
          : "Steps 1–50. Filled buttons were measured. This layer does not read them."}
      </p>
    </div>
  )
}

function Camera({
  title,
  image,
  tokens,
  kind,
  grid,
  weights,
  marks,
  peak,
  hover,
  onHover,
}: {
  title: string
  image: string
  tokens: Token[]
  kind: "front" | "wrist"
  grid: number
  weights: number[] | null
  marks: Mark[]
  peak: number
  hover: number | null
  onHover: (index: number | null) => void
}) {
  const cells = tokens
    .map((token, index) => ({ token, index }))
    .filter((item) => item.token.kind === kind)
  return (
    <div className="min-w-0">
      <p className="mb-2 font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">{title}</p>
      <div className="relative aspect-square border border-border bg-black">
        <img src={`/attention/${image}`} alt="" draggable={false} className="absolute inset-0 size-full" />
        {weights && (
          <div
            className="pointer-events-none absolute inset-0 grid"
            style={{ gridTemplateColumns: `repeat(${grid}, minmax(0, 1fr))` }}
          >
            {cells.map(({ index }) => {
              const mark = marks.find((item) => item.cell === index)
              const hot = index === hover
              return (
                <span
                  key={index}
                  style={{
                    backgroundColor: mark ? mark.color : ink(weights[index] ?? 0, peak),
                    opacity: mark ? 0.82 : 1,
                    outline: hot ? "2px solid #f4f0e8" : undefined,
                    outlineOffset: "-2px",
                  }}
                />
              )
            })}
          </div>
        )}
        {weights && (
          <svg className="pointer-events-none absolute inset-0 size-full" viewBox={`0 0 ${grid} ${grid}`}>
            {marks.map((mark) => {
              const token = mark.cell != null ? tokens[mark.cell] : null
              if (!token || token.kind !== kind || token.row == null || token.col == null) return null
              return (
                <g key={mark.n}>
                  <rect x={token.col} y={token.row} width={1} height={1} fill={mark.color} fillOpacity={0.9} />
                  <text
                    x={token.col + 0.5}
                    y={token.row + 0.72}
                    textAnchor="middle"
                    fontSize="0.62"
                    fontFamily="ui-monospace, monospace"
                    fill="#1c1424"
                  >
                    {mark.n}
                  </text>
                </g>
              )
            })}
          </svg>
        )}
        <div className="absolute inset-0 grid" style={{ gridTemplateColumns: `repeat(${grid}, minmax(0, 1fr))` }}>
          {cells.map(({ token, index }) => (
            <button
              key={index}
              type="button"
              aria-label={`${title} ${token.row != null && token.col != null ? region(token.row, token.col, grid) : ""}`}
              className="size-full"
              onMouseEnter={() => onHover(index)}
              onFocus={() => onHover(index)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function Bars({
  tokens,
  weights,
  marks,
  hover,
  onHover,
  onPick,
  query,
}: {
  tokens: Token[]
  weights: number[]
  marks: Mark[]
  hover: number | null
  onHover: (index: number | null) => void
  onPick?: (index: number) => void
  query?: number | null
}) {
  const chips = tokens
    .map((token, index) => ({ token, index }))
    .filter((item) => item.token.kind === "word" || item.token.kind === "state")
  const peak = Math.max(...chips.map(({ index }) => weights[index] ?? 0), 1)
  return (
    <div className="flex items-end gap-1 overflow-x-auto">
      {chips.map(({ token, index }) => {
        const value = weights[index] ?? 0
        const blocked = token.kind === "state" && value === 0 && query !== index
        const mark = marks.find((item) => item.token === index)
        const hot = hover === index
        return (
          <button
            key={index}
            type="button"
            onClick={() => onPick?.(index)}
            onMouseEnter={() => onHover(index)}
            onFocus={() => onHover(index)}
            className={cn("flex w-12 shrink-0 flex-col items-center gap-1", !onPick && "cursor-default")}
          >
            <span className="font-mono text-[10px] text-muted-foreground tabular-nums">{pct(value)}</span>
            <span
              className="flex w-full items-end border border-border"
              style={{ height: "4.5rem" }}
            >
              <span
                className="block w-full"
                style={{
                  height: blocked ? "100%" : `${Math.max((value / peak) * 100, value > 0 ? 8 : 0)}%`,
                  backgroundColor: blocked ? "transparent" : mark ? mark.color : "rgba(192, 132, 252, 0.85)",
                  backgroundImage: blocked
                    ? "repeating-linear-gradient(-45deg, transparent, transparent 3px, rgba(244,240,232,0.35) 3px, rgba(244,240,232,0.35) 4px)"
                    : undefined,
                  outline: hot ? "2px solid #f4f0e8" : undefined,
                  outlineOffset: "-2px",
                }}
              />
            </span>
            <span className={cn("font-mono text-[10px]", index === query ? "text-foreground" : "text-muted-foreground")}>
              {mark ? `${mark.n} ` : ""}
              {token.kind === "state" ? "joints" : token.label}
            </span>
          </button>
        )
      })}
    </div>
  )
}

function Legend({ marks, onHover }: { marks: Mark[]; onHover: (index: number | null) => void }) {
  if (marks.length === 0) return null
  return (
    <ol className="grid gap-2 sm:grid-cols-2">
      {marks.map((mark) => (
        <li key={`${mark.n}-${mark.label}-${mark.cell ?? mark.token}`}>
          <button
            type="button"
            onMouseEnter={() => onHover(mark.cell ?? mark.token ?? null)}
            onFocus={() => onHover(mark.cell ?? mark.token ?? null)}
            className="flex w-full items-start gap-3 text-left"
          >
            <span
              className="mt-0.5 flex size-5 shrink-0 items-center justify-center font-mono text-[10px] text-neutral-950"
              style={{ backgroundColor: mark.color }}
            >
              {mark.n}
            </span>
            <span className="text-[13px] leading-snug text-muted-foreground">
              <span className="text-foreground">{mark.label}</span>
              {" · "}
              {pct(mark.value)}
            </span>
          </button>
        </li>
      ))}
    </ol>
  )
}
