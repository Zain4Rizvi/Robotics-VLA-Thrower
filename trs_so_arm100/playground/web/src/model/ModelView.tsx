import { useGSAP } from "@gsap/react"
import gsap from "gsap"
import { useEffect, useRef, useState } from "react"

import { AttentionPlate } from "@/model/AttentionPlate"
import { blocks, layerCount, layerKind, type BlockId } from "@/model/facts"
import { Flow } from "@/model/Flow"
import { LayerStack } from "@/model/LayerStack"
import { cn } from "@/lib/utils"

gsap.registerPlugin(useGSAP)

const copy: Record<BlockId, { body: string }> = {
  vision: {
    body: "Twelve layers, width 768, twelve heads. Each camera is resized to 512 and split into 1024 patches. This tower stays frozen.",
  },
  connector: {
    body: "A 4×4 pixel shuffle turns the 32×32 patch grid into 64 tokens, then a linear map takes each one to 960. Two cameras make 128 tokens. Frozen.",
  },
  language: {
    body: "The first 16 of SmolVLM2’s 32 text layers. Width 960, 15 query heads and 5 key/value heads. They mix the image tokens, the sentence (at most 48 tokens), and one state token. Frozen. Each layer’s keys and values are cached for the expert.",
  },
  expert: {
    body: "Sixteen layers at width 720, trained. The chunk is 50 slots of noise. Ten Euler steps turn that noise into joint targets. This arm keeps the first 6 of the 32-wide output. The viewer starts a new chunk every 10 control steps.",
  },
}

export function ModelView({ instruction }: { instruction: string }) {
  const root = useRef<HTMLDivElement>(null)
  const [block, setBlock] = useState<BlockId | null>(null)
  const [layer, setLayer] = useState(0)

  useEffect(() => {
    if (block == null) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setBlock(null)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [block])

  useGSAP(
    () => {
      const motion = gsap.matchMedia()
      motion.add("(prefers-reduced-motion: no-preference)", () => {
        gsap.to("[data-model]", { y: 0, duration: 0.45, stagger: 0.05, ease: "power2.out" })
      })
    },
    { scope: root, dependencies: [block] },
  )

  return (
    <div ref={root} className="absolute inset-0 z-20 flex flex-col bg-background pt-20 pl-36">
      {block != null && (
        <div className="flex shrink-0 items-center px-7 pb-2">
          <button
            type="button"
            onClick={() => setBlock(null)}
            className="border border-border px-3 py-1.5 font-mono text-[11px] tracking-[0.14em] text-foreground uppercase hover:bg-card"
          >
            ← Flow
          </button>
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-auto">
        {block == null ? (
          <Flow
            instruction={instruction}
            onOpen={(id) => {
              setLayer(0)
              setBlock(id)
            }}
          />
        ) : (
          <Detail block={block} layer={layer} onLayer={setLayer} />
        )}
      </div>
    </div>
  )
}

function Detail({
  block,
  layer,
  onLayer,
}: {
  block: BlockId
  layer: number
  onLayer: (index: number) => void
}) {
  const spec = blocks.find((item) => item.id === block)
  const count = layerCount(block)
  const title = spec?.title ?? block

  return (
    <div className="mx-auto flex min-h-full max-w-6xl flex-col gap-6 px-7 py-6">
      <div data-model className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
        <h1 className="font-serif text-2xl tracking-tight">{title}</h1>
        <p className="text-[13px] text-muted-foreground">{spec?.meta}</p>
      </div>
      <p data-model className="max-w-3xl text-[14px] leading-relaxed text-muted-foreground">
        {copy[block].body}
      </p>

      {count === 0 ? (
        <Connector />
      ) : (
        <div className="grid min-h-[28rem] flex-1 gap-8 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
          <div data-model className="flex min-h-80 flex-col border border-border">
            <div className="flex items-center justify-between px-4 py-3">
              <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
                Layer {layer + 1} of {count}
              </p>
              <p className="text-[13px] text-muted-foreground">{layerKind(block, layer)}</p>
            </div>
            <LayerStack count={count} selected={layer} onSelect={onLayer} />
            <div className="flex gap-1 overflow-x-auto px-4 py-3" role="listbox" aria-label="Layers">
              {Array.from({ length: count }, (_, index) => (
                <button
                  key={index}
                  type="button"
                  role="option"
                  aria-selected={index === layer}
                  onClick={() => onLayer(index)}
                  className={cn(
                    "h-6 min-w-6 px-1 font-mono text-[10px] tabular-nums",
                    index === layer ? "bg-foreground text-background" : "bg-foreground/10 text-muted-foreground",
                  )}
                >
                  {index + 1}
                </button>
              ))}
            </div>
          </div>
          <div data-model className="border border-border p-5">
            <AttentionPlate block={block} layer={layer} />
          </div>
        </div>
      )}
    </div>
  )
}

function Connector() {
  return (
    <div data-model className="grid max-w-3xl gap-4 border border-border p-5 sm:grid-cols-3">
      <Step k="1024 × 768" v="Patches out of SigLIP" />
      <Step k="64 × 12288" v="Each 4×4 group, packed" />
      <Step k="64 × 960" v="One linear map, no bias" />
    </div>
  )
}

function Step({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <p className="font-mono text-[13px] text-foreground">{k}</p>
      <p className="mt-1 text-[13px] text-muted-foreground">{v}</p>
    </div>
  )
}

