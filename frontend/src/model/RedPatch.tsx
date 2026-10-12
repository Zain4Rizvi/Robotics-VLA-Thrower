import { useEffect, useState } from "react"

const MARKS = 5
const COLORS = ["#d8b4fe", "#67e8f9", "#f9a8d4", "#fcd34d", "#86efac"]

type Frame = {
  id: string
  seed: number
  image: string
  query: [number, number]
  layers: number[][]
}

type Snapshot = Frame & { weights: number[] }

type Dump = {
  grid: number
  patch: number
  heads: string
  snapshots: Frame[]
}

type Spot = { index: number; value: number; color: string; n: number }

function ink(value: number, max: number): string {
  const t = max > 0 ? Math.min(value / max, 1) : 0
  const alpha = 0.04 + 0.78 * t * t
  return `rgba(192, 132, 252, ${alpha.toFixed(3)})`
}

function region(index: number, grid: number): string {
  const row = Math.floor(index / grid)
  const col = index % grid
  const vertical = row < grid * 0.4 ? "Upper" : row > grid * 0.6 ? "Lower" : "Middle"
  const horizontal = col < grid * 0.4 ? "left" : col > grid * 0.6 ? "right" : "center"
  return `${vertical} ${horizontal}`
}

function momentLabel(query: [number, number], grid: number): string {
  return query[1] < grid / 2 ? "Cube on the left" : "Cube on the right"
}

function spotsFor(snap: Snapshot, grid: number): Spot[] {
  const query = snap.query[0] * grid + snap.query[1]
  const ranked = snap.weights
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
  const picked: { value: number; index: number }[] = []
  const self = ranked.find((item) => item.index === query)
  if (self) picked.push(self)
  for (const item of ranked) {
    if (picked.length >= MARKS) break
    if (item.index === query) continue
    picked.push(item)
  }
  return picked.map((item, n) => ({ ...item, color: COLORS[n] ?? COLORS[0], n: n + 1 }))
}

function stableIndexes(other: Snapshot, grid: number): Set<number> {
  const query = other.query[0] * grid + other.query[1]
  return new Set(
    other.weights
      .map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8)
      .map((item) => item.index)
      .filter((index) => index !== query),
  )
}

export function RedPatch({ layer }: { layer: number }) {
  const [dump, setDump] = useState<Dump | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [id, setId] = useState("a")
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    setHover(null)
  }, [layer])

  useEffect(() => {
    let live = true
    fetch("/attention/red-patch.json")
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

  if (error) return <p className="text-[13px] text-muted-foreground">{error}</p>
  if (!dump) return <p className="text-[13px] text-muted-foreground">Loading the two frames.</p>

  const frame = dump.snapshots.find((item) => item.id === id) ?? dump.snapshots[0]
  const otherFrame = dump.snapshots.find((item) => item.id !== frame.id) ?? frame
  const layerIndex = Math.min(Math.max(layer, 0), frame.layers.length - 1)
  const snap: Snapshot = { ...frame, weights: frame.layers[layerIndex] }
  const other: Snapshot = { ...otherFrame, weights: otherFrame.layers[layerIndex] }
  const max = Math.max(...snap.weights, ...other.weights)
  const spots = spotsFor(snap, dump.grid)
  const steady = stableIndexes(other, dump.grid)
  const query = snap.query[0] * dump.grid + snap.query[1]
  const focus = hover ?? query
  const focusSpot = spots.find((spot) => spot.index === focus)

  return (
    <figure className="flex w-full min-w-0 flex-col gap-5" onMouseLeave={() => setHover(null)}>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="max-w-xl">
          <p className="font-serif text-xl tracking-tight text-foreground">
            The square on the red cube, and where it is looking.
          </p>
          <p className="mt-2 text-[13px] leading-relaxed text-muted-foreground">
            The photo is cut into a {dump.grid}×{dump.grid} grid. Violet square 1 covers the cube. Layer{" "}
            {layerIndex + 1} scores how much that square attends to every square. Choose a layer below to change this
            picture. The same score is the same color in both pictures.
          </p>
        </div>
        <div className="flex gap-1">
          {dump.snapshots.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                setHover(null)
                setId(item.id)
              }}
              className={
                item.id === snap.id
                  ? "bg-foreground px-2.5 py-1.5 font-mono text-[10px] text-background"
                  : "bg-foreground/10 px-2.5 py-1.5 font-mono text-[10px] text-muted-foreground"
              }
            >
              {momentLabel(item.query, dump.grid)}
            </button>
          ))}
        </div>
      </div>

      <div className="grid min-w-0 items-start gap-4 md:grid-cols-2">
        <Board
          title="The camera"
          snap={snap}
          grid={dump.grid}
          spots={spots}
          max={max}
          hover={hover}
          photo
          onHover={setHover}
        />
        <Board
          title="The same squares, as attention"
          snap={snap}
          grid={dump.grid}
          spots={spots}
          max={max}
          hover={hover}
          photo={false}
          onHover={setHover}
        />
      </div>

      <p className="min-h-5 font-mono text-[12px] text-foreground">
        {focusSpot
          ? `${focusSpot.n}  ·  ${region(focus, dump.grid)}  ·  ${(focusSpot.value * 100).toFixed(1)}%`
          : `${region(focus, dump.grid)}  ·  ${((snap.weights[focus] ?? 0) * 100).toFixed(1)}%`}
        <span className="text-muted-foreground">
          {" "}
          — {focus === query ? "this square is the red cube" : "this square of the photo"}
        </span>
      </p>

      <ol className="grid gap-2 sm:grid-cols-2">
        {spots.map((spot) => (
          <li key={spot.index}>
            <button
              type="button"
              onMouseEnter={() => setHover(spot.index)}
              onFocus={() => setHover(spot.index)}
              className="flex w-full items-start gap-3 text-left"
            >
              <span
                className="mt-0.5 flex size-5 shrink-0 items-center justify-center font-mono text-[10px] text-neutral-950"
                style={{ backgroundColor: spot.color }}
              >
                {spot.n}
              </span>
              <span className="text-[13px] leading-snug text-muted-foreground">
                <span className="text-foreground">
                  {spot.index === query ? "On the red cube" : region(spot.index, dump.grid)}
                </span>
                {" · "}
                {(spot.value * 100).toFixed(1)}%
                {spot.index === query
                  ? ". The cube’s own square. It keeps the most attention on itself."
                  : steady.has(spot.index)
                    ? ". Still this bright when the cube moves."
                    : ". Bright in this moment."}
              </span>
            </button>
          </li>
        ))}
      </ol>

      <p className="max-w-2xl text-[13px] leading-relaxed text-muted-foreground">
        Switch moments. Square 1 moves with the cube. A color that stays in place is a pattern this layer keeps no
        matter where the cube is. The mean is across all {dump.heads === "mean" ? "12" : dump.heads} heads, and the
        scores on one row add up to 100%.
      </p>
    </figure>
  )
}

function Board({
  title,
  snap,
  grid,
  spots,
  max,
  hover,
  photo,
  onHover,
}: {
  title: string
  snap: Snapshot
  grid: number
  spots: Spot[]
  max: number
  hover: number | null
  photo: boolean
  onHover: (index: number) => void
}) {
  const byIndex = new Map(spots.map((spot) => [spot.index, spot]))
  const query = spots[0]
  return (
    <div className="min-w-0">
      <p className="mb-2 font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">{title}</p>
      <div className="relative aspect-square border border-border bg-black">
        {photo && (
          <img
            src={`/attention/${snap.image}`}
            alt=""
            draggable={false}
            className="absolute inset-0 size-full"
          />
        )}
        <div
          className="pointer-events-none absolute inset-0 grid"
          style={{ gridTemplateColumns: `repeat(${grid}, minmax(0, 1fr))` }}
        >
          {snap.weights.map((value, index) => {
            const spot = byIndex.get(index)
            const hot = index === hover
            return (
              <span
                key={index}
                style={{
                  backgroundColor: spot ? spot.color : ink(value, max),
                  opacity: spot ? (photo ? 0.78 : 0.95) : 1,
                  outline: hot ? "2px solid #f4f0e8" : undefined,
                  outlineOffset: "-2px",
                }}
              />
            )
          })}
        </div>
        <svg className="pointer-events-none absolute inset-0 size-full" viewBox={`0 0 ${grid} ${grid}`}>
          {query &&
            spots.slice(1).map((spot) => (
              <line
                key={spot.index}
                x1={(query.index % grid) + 0.5}
                y1={Math.floor(query.index / grid) + 0.5}
                x2={(spot.index % grid) + 0.5}
                y2={Math.floor(spot.index / grid) + 0.5}
                stroke={spot.color}
                strokeWidth={0.12}
              />
            ))}
          {spots.map((spot) => {
            const col = spot.index % grid
            const row = Math.floor(spot.index / grid)
            return (
              <g key={spot.index}>
                <rect x={col} y={row} width={1} height={1} fill={spot.color} fillOpacity={photo ? 0.82 : 0.96} />
                <text
                  x={col + 0.5}
                  y={row + 0.72}
                  textAnchor="middle"
                  fontSize="0.62"
                  fontFamily="ui-monospace, monospace"
                  fill="#1c1424"
                >
                  {spot.n}
                </text>
              </g>
            )
          })}
          {hover != null && (
            <rect
              x={hover % grid}
              y={Math.floor(hover / grid)}
              width={1}
              height={1}
              fill="none"
              stroke="#f4f0e8"
              strokeWidth={0.14}
            />
          )}
        </svg>
        <div
          className="absolute inset-0 grid"
          style={{ gridTemplateColumns: `repeat(${grid}, minmax(0, 1fr))` }}
        >
          {snap.weights.map((_, index) => (
            <button
              key={index}
              type="button"
              aria-label={`${photo ? "Patch" : "Weight"} ${region(index, grid)}`}
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
