import { useGSAP } from "@gsap/react"
import gsap from "gsap"
import { useEffect, useRef, useState } from "react"

import { attachFeed } from "@/use-playground"

import { AttentionPlate } from "@/model/AttentionPlate"
import { blocks, layerCount, layerKind, type BlockId } from "@/model/facts"
import { LayerStack } from "@/model/LayerStack"
import { cn } from "@/lib/utils"

gsap.registerPlugin(useGSAP)

const hint: Record<BlockId, string> = {
  vision: "Frozen. One pass per camera.",
  connector: "Shrinks each camera to 64 tokens.",
  language: "Frozen. Writes the prefix cache.",
  expert: "Trained. Reads that cache on odd layers.",
}

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
    <div ref={root} className="absolute inset-0 z-20 overflow-auto bg-background pt-20">
      {block == null ? (
        <Overview
          instruction={instruction}
          onOpen={(id) => {
            setLayer(0)
            setBlock(id)
          }}
        />
      ) : (
        <Detail
          block={block}
          layer={layer}
          onLayer={setLayer}
          onBack={() => setBlock(null)}
        />
      )}
    </div>
  )
}

function Overview({ instruction, onOpen }: { instruction: string; onOpen: (id: BlockId) => void }) {
  return (
    <div className="mx-auto flex min-h-full max-w-6xl flex-col gap-8 px-7 py-6">
      <div data-model className="flex items-end justify-between gap-6">
        <div>
          <h1 className="font-serif text-2xl tracking-tight">SmolVLA</h1>
          <p className="mt-1 text-[13px] text-muted-foreground">
            Two cameras, the sentence, and the arm. Select a block to open it.
          </p>
        </div>
        <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
          Schematic
        </p>
      </div>

      <div className="grid flex-1 gap-4 lg:grid-cols-[minmax(0,0.85fr)_minmax(0,0.7fr)_minmax(0,1.15fr)_minmax(0,0.75fr)]">
        <section data-model className="flex flex-col gap-5 border border-border p-5">
          <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">Inputs</p>
          <div className="grid grid-cols-2 gap-3">
            <Frame label="camera1" feed={0} />
            <Frame label="camera2" feed={1} />
          </div>
          <p className="text-[13px] leading-relaxed text-muted-foreground">256², padded to 512².</p>
          <p className="font-mono text-[12px] leading-relaxed text-foreground/90">
            {instruction || "Waiting for the instruction"}
          </p>
          <StateStrip />
        </section>

        <section data-model className="flex flex-col gap-3">
          <StageButton id="vision" onOpen={onOpen} />
          <StageButton id="connector" onOpen={onOpen} />
        </section>

        <section data-model className="flex flex-col gap-3">
          <StageButton id="language" onOpen={onOpen} />
          <p className="text-center font-mono text-[10px] tracking-[0.18em] text-muted-foreground uppercase">
            Prefix cache
          </p>
          <StageButton id="expert" onOpen={onOpen} markOdd />
        </section>

        <section data-model className="flex flex-col border border-border p-5">
          <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">Output</p>
          <p className="mt-3 font-serif text-lg tracking-tight">50 × 6</p>
          <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">
            Joint targets, after 10 denoising steps. A fresh chunk is planned every 10 control steps.
          </p>
        </section>
      </div>
    </div>
  )
}

function Detail({
  block,
  layer,
  onLayer,
  onBack,
}: {
  block: BlockId
  layer: number
  onLayer: (index: number) => void
  onBack: () => void
}) {
  const spec = blocks.find((item) => item.id === block)
  const count = layerCount(block)
  const title = spec?.title ?? block

  return (
    <div className="mx-auto flex min-h-full max-w-6xl flex-col gap-6 px-7 py-6">
      <div data-model className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
        <button
          type="button"
          onClick={onBack}
          className="font-mono text-[11px] tracking-[0.14em] text-muted-foreground uppercase hover:text-foreground"
        >
          SmolVLA
        </button>
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

function StageButton({
  id,
  onOpen,
  markOdd = false,
}: {
  id: BlockId
  onOpen: (id: BlockId) => void
  markOdd?: boolean
}) {
  const spec = blocks.find((item) => item.id === id)
  const count = layerCount(id)
  return (
    <button
      type="button"
      onClick={() => onOpen(id)}
      className="flex flex-1 flex-col gap-4 border border-border p-5 text-left transition-colors hover:bg-card"
    >
      <span className="flex items-baseline justify-between gap-3">
        <span className="font-serif text-xl tracking-tight">{spec?.title}</span>
        <span className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">
          {spec?.meta}
        </span>
      </span>
      {count > 0 && (
        <span className="flex flex-col gap-1" aria-hidden="true">
          {Array.from({ length: count }, (_, index) => (
            <span
              key={index}
              className={cn(
                "h-px w-full",
                markOdd && index % 2 === 1 ? "bg-foreground/70" : "bg-foreground/25",
              )}
            />
          ))}
        </span>
      )}
      <span className="text-[13px] text-muted-foreground">{hint[id]}</span>
    </button>
  )
}

function Frame({ label, feed }: { label: string; feed: 0 | 1 }) {
  const image = useRef<HTMLImageElement>(null)
  useEffect(() => {
    attachFeed(feed, image.current)
  }, [feed])
  return (
    <div>
      <div className="aspect-[4/3] overflow-hidden border border-border bg-foreground/5">
        <img ref={image} data-feed={feed} alt={label} draggable={false} className="size-full object-cover" />
      </div>
      <p className="mt-1.5 font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">{label}</p>
    </div>
  )
}

function StateStrip() {
  return (
    <div>
      <div className="flex gap-px" aria-hidden="true">
        {Array.from({ length: 32 }, (_, index) => (
          <span
            key={index}
            className={cn(
              "h-6 flex-1",
              index < 6 && "bg-foreground/80",
              index >= 6 && index < 10 && "bg-foreground/45",
              index >= 10 && index < 12 && "bg-foreground/25",
              index >= 12 && "bg-foreground/10",
            )}
          />
        ))}
      </div>
      <p className="mt-2 text-[12px] leading-relaxed text-muted-foreground">
        32-wide state. Six joint angles, red and green xy, two probe logits, then zeros.
      </p>
    </div>
  )
}
