export type BlockId = "vision" | "connector" | "language" | "expert"

export const blocks: { id: BlockId; title: string; meta: string }[] = [
  { id: "vision", title: "SigLIP", meta: "12 layers · 768" },
  { id: "connector", title: "Connector", meta: "64 × 960" },
  { id: "language", title: "SmolVLM", meta: "16 layers · 960" },
  { id: "expert", title: "Action expert", meta: "16 layers · 720" },
]

export function layerCount(block: BlockId): number {
  if (block === "connector") return 0
  if (block === "vision") return 12
  return 16
}

export function layerKind(block: BlockId, index: number): string {
  if (block === "vision") return "Self-attention over this camera’s patches"
  if (block === "language") return "Self-attention over the prefix"
  return index % 2 === 0 ? "Self-attention inside the chunk" : "Cross-attention into the prefix cache"
}

const prefix = ["cam", "stack", "red", "cube", "green", "state"] as const
const steps = ["t0", "t1", "t2", "t3", "t4", "t5"] as const

export function plate(block: BlockId, layer: number): {
  rows: readonly string[]
  cols: readonly string[]
  on: (row: number, col: number) => boolean
  note: string
} {
  if (block === "vision") {
    const n = 8
    const labels = Array.from({ length: n }, (_, i) => String(i))
    return {
      rows: labels,
      cols: labels,
      on: () => true,
      note: "Each camera is 1024 patches. They attend to each other. This 8×8 is a stand-in for that block.",
    }
  }
  if (block === "language") {
    return {
      rows: prefix,
      cols: prefix,
      on: (row, col) => prefix[col] !== "state" || prefix[row] === "state",
      note: "Cameras and the sentence see each other. The state token can see them. They cannot see the state, and nothing here sees the action chunk.",
    }
  }
  const cols = [...prefix, ...steps]
  const cross = layer % 2 === 1
  return {
    rows: steps,
    cols,
    on: (row, col) => {
      if (col < prefix.length) return cross
      return !cross && col - prefix.length <= row
    },
    note: cross
      ? "Odd layers read the cached prefix: both cameras, the sentence, and the state."
      : "Even layers stay inside the chunk. A step sees itself and earlier steps.",
  }
}
