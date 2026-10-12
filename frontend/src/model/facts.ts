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
  return index % 2 === 0
    ? "Self-attention: earlier steps, and the cached scene"
    : "Cross-attention into the prefix cache"
}
