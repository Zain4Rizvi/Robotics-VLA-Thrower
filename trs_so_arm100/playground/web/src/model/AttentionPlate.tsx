import { plate, type BlockId } from "@/model/facts"

export function AttentionPlate({ block, layer }: { block: BlockId; layer: number }) {
  const spec = plate(block, layer)
  return (
    <figure className="flex min-w-0 flex-col gap-3">
      <figcaption className="flex items-baseline justify-between gap-4">
        <span className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
          Attention mask
        </span>
        <span className="font-mono text-[10px] text-muted-foreground">queries ↓ · keys →</span>
      </figcaption>
      <div className="overflow-x-auto">
        <div
          className="grid w-max gap-px"
          style={{ gridTemplateColumns: `auto repeat(${spec.cols.length}, minmax(1.4rem, 1fr))` }}
        >
          <span />
          {spec.cols.map((label) => (
            <span
              key={label}
              className="px-0.5 text-center font-mono text-[10px] text-muted-foreground"
            >
              {label}
            </span>
          ))}
          {spec.rows.map((row, r) => (
            <Row key={row} label={row} cols={spec.cols} on={(c) => spec.on(r, c)} />
          ))}
        </div>
      </div>
      <p className="max-w-md text-[13px] leading-relaxed text-muted-foreground">{spec.note}</p>
    </figure>
  )
}

function Row({
  label,
  cols,
  on,
}: {
  label: string
  cols: readonly string[]
  on: (col: number) => boolean
}) {
  return (
    <>
      <span className="pr-2 text-right font-mono text-[10px] text-muted-foreground">{label}</span>
      {cols.map((col, index) => (
        <span
          key={`${label}-${col}`}
          className={on(index) ? "size-5 bg-foreground/80" : "size-5 bg-foreground/10"}
        />
      ))}
    </>
  )
}
