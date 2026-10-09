import { useGSAP } from "@gsap/react"
import gsap from "gsap"
import { lazy, Suspense, useEffect, useRef, useState, type FormEvent, type PointerEvent as ReactPointerEvent } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group"
import { Skeleton } from "@/components/ui/skeleton"
import { usePlayback, type RobotId } from "@/use-playback"

const ModelView = lazy(() => import("@/model/ModelView").then((mod) => ({ default: mod.ModelView })))
const ExpertView = lazy(() => import("@/model/ExpertView").then((mod) => ({ default: mod.ExpertView })))

const ROBOTS = [
  { id: "stack", title: "SO-ARM100", subtitle: "Red on green" },
  { id: "openarm", title: "Multithrow", subtitle: "OpenArm" },
  { id: "peg", title: "Peg", subtitle: "OpenArm" },
] as const

const CAMERAS: Record<RobotId, [string, string]> = {
  stack: ["camera1", "camera2"],
  openarm: ["headcam", "camera_wrist_right"],
  peg: ["tablecam", "camera_wrist_right"],
}

gsap.registerPlugin(useGSAP)

function PolicyCamera({ label, feed }: { label: string; feed: 0 | 1 }) {
  const [shown, setShown] = useState(false)
  return (
    <figure className="flex w-44 flex-col gap-1.5">
      <div className="relative aspect-[4/3] overflow-hidden rounded-md border border-border bg-muted">
        {!shown && <Skeleton className="absolute inset-0 size-full rounded-none" />}
        <img
          data-feed={feed}
          alt={label}
          draggable={false}
          onLoad={() => setShown(true)}
          className="size-full object-cover"
        />
      </div>
      <figcaption className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">{label}</figcaption>
    </figure>
  )
}

export default function App() {
  const root = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const revealed = useRef(false)
  const revealHero = useRef<() => void>(() => {})
  const drag = useRef<{ id: number; x: number; y: number; button: number } | null>(null)
  const [draft, setDraft] = useState("")
  const [focused, setFocused] = useState(false)
  const [page, setPage] = useState<"scene" | "model">("scene")
  const { robot, ready, message, instruction, seed, selectRobot, submitInstruction, nextLayout, nudge } =
    usePlayback(canvasRef)

  useGSAP(
    () => {
      const motion = gsap.matchMedia()
      motion.add("(prefers-reduced-motion: no-preference)", () => {
        gsap.to("[data-enter]", {
          y: 0,
          duration: 0.5,
          stagger: 0.08,
          ease: "power2.out",
        })
      })
      revealHero.current = () => {
        if (revealed.current) return
        revealed.current = true
        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches
        if (reduce) {
          gsap.set("[data-hero]", { autoAlpha: 1 })
          return
        }
        gsap.to("[data-hero]", { autoAlpha: 1, duration: 0.5, ease: "power2.out" })
      }
    },
    { scope: root },
  )

  useEffect(() => {
    if (ready) revealHero.current()
  }, [ready])

  useEffect(() => {
    if (!focused) setDraft(instruction)
  }, [focused, instruction])

  const selected = ROBOTS.find((item) => item.id === robot) ?? ROBOTS[0]
  const cameras = CAMERAS[robot]

  function onSubmit(event: FormEvent) {
    event.preventDefault()
    const text = draft.trim()
    if (!text || !ready) return
    submitInstruction(text)
  }

  function onPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0 && event.button !== 2) return
    drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, button: event.button }
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      // Pointer capture is unavailable for some synthetic events.
    }
  }

  function onPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const current = drag.current
    if (!current || event.pointerId !== current.id) return
    const height = event.currentTarget.clientHeight || 1
    const dx = (event.clientX - current.x) / height
    const dy = (event.clientY - current.y) / height
    current.x = event.clientX
    current.y = event.clientY
    if (dx === 0 && dy === 0) return
    nudge(current.button === 2 ? "pan" : "orbit", dx, dy)
  }

  function onPointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    if (drag.current?.id === event.pointerId) drag.current = null
  }

  return (
    <div ref={root} className="relative h-dvh overflow-hidden bg-background text-foreground">
      <div hidden={page !== "scene"} className={page === "scene" ? "contents" : undefined}>
        <div
          data-scene
          role="application"
          tabIndex={0}
          aria-label="Scene. Drag to orbit, scroll to zoom, right-drag to pan."
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onContextMenu={(event) => event.preventDefault()}
          className="absolute inset-0 touch-none outline-none"
        >
          <canvas
            ref={canvasRef}
            data-hero
            className="pointer-events-none size-full select-none"
          />
        </div>

        <div className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-background/80 to-transparent" />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-background/50 to-transparent" />

        {!ready && (
          <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center px-6">
            <p className="font-mono text-[11px] tracking-[0.22em] text-muted-foreground uppercase">
              {message || "Loading"}
            </p>
          </div>
        )}

        <div className="absolute inset-x-0 bottom-0 z-10 flex flex-col gap-4 px-5 py-5 lg:flex-row lg:items-end lg:gap-8 lg:px-7 lg:py-7">
          <div data-enter className="flex shrink-0 flex-col gap-2.5">
            <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
              {robot === "stack" ? "What the policy sees" : "What the expert sees"}
            </p>
            <div className="flex gap-3">
              <PolicyCamera key={`${robot}-0`} label={cameras[0]} feed={0} />
              <PolicyCamera key={`${robot}-1`} label={cameras[1]} feed={1} />
            </div>
          </div>

          {robot === "stack" ? (
            <form data-enter onSubmit={onSubmit} className="w-full lg:mb-5 lg:ml-auto lg:max-w-xl">
              <FieldGroup>
                <Field data-disabled={ready ? undefined : true}>
                  <FieldLabel htmlFor="instruction" className="sr-only">
                    Instruction
                  </FieldLabel>
                  <div className="flex items-center gap-3">
                    <InputGroup className="h-11 bg-background">
                      <InputGroupInput
                        id="instruction"
                        value={draft}
                        disabled={!ready}
                        placeholder="stack the red cube on the green cube"
                        onFocus={() => setFocused(true)}
                        onBlur={() => setFocused(false)}
                        onChange={(event) => setDraft(event.target.value)}
                      />
                      <InputGroupAddon align="inline-end">
                        <InputGroupButton type="submit" variant="default" size="sm" disabled={!ready}>
                          Set
                        </InputGroupButton>
                      </InputGroupAddon>
                    </InputGroup>
                    <Button
                      type="button"
                      variant="outline"
                      className="h-11 bg-background px-4"
                      disabled={!ready}
                      onClick={nextLayout}
                    >
                      New layout
                    </Button>
                  </div>
                </Field>
              </FieldGroup>
            </form>
          ) : (
            <div data-enter className="flex w-full items-center gap-3 lg:mb-5 lg:ml-auto lg:max-w-xl">
              <p className="min-w-0 flex-1 font-serif text-[17px] leading-snug">{instruction || "Planning the expert"}</p>
              <Button
                type="button"
                variant="outline"
                className="h-11 shrink-0 bg-background px-4"
                disabled={!ready}
                onClick={nextLayout}
              >
                New layout
              </Button>
            </div>
          )}
        </div>
      </div>

      <header className="absolute inset-x-0 top-0 z-30 flex items-start justify-between gap-8 px-7 py-6">
        <div className="flex flex-col gap-1.5">
          <p className="font-serif text-[17px] leading-none tracking-tight">{selected.title}</p>
          <p className="text-[13px] leading-none text-muted-foreground">{selected.subtitle}</p>
        </div>
        <nav className="absolute left-1/2 top-6 flex -translate-x-1/2 items-center gap-5 font-mono text-[11px] tracking-[0.14em] uppercase">
          <button
            type="button"
            aria-current={page === "scene" ? "page" : undefined}
            className={page === "scene" ? "text-foreground" : "text-muted-foreground hover:text-foreground"}
            onClick={() => setPage("scene")}
          >
            Scene
          </button>
          <button
            type="button"
            aria-current={page === "model" ? "page" : undefined}
            className={page === "model" ? "text-foreground" : "text-muted-foreground hover:text-foreground"}
            onClick={() => setPage("model")}
          >
            Model
          </button>
        </nav>
        <div className="flex items-center gap-3 font-mono text-[11px] tracking-wide text-muted-foreground">
          <Badge variant="outline">{ready ? "Replay" : "Loading"}</Badge>
          {seed != null && <span className="tabular-nums">seed {seed}</span>}
          {ready && (
            <>
              <span aria-hidden="true">·</span>
              <span>Idle</span>
            </>
          )}
        </div>
      </header>
      <nav className="absolute top-24 left-7 z-40 flex flex-col items-start gap-2.5 font-mono text-[11px] tracking-[0.14em] uppercase">
        {ROBOTS.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-current={robot === item.id ? "true" : undefined}
            className={robot === item.id ? "text-foreground" : "text-muted-foreground hover:text-foreground"}
            onClick={() => selectRobot(item.id)}
          >
            {item.title}
          </button>
        ))}
      </nav>
      {page === "model" && (
        <Suspense
          fallback={
            <p className="absolute inset-0 z-20 flex items-center justify-center font-mono text-[11px] tracking-[0.22em] text-muted-foreground uppercase">
              Opening
            </p>
          }
        >
          {robot === "stack" ? (
            <ModelView instruction={instruction} />
          ) : (
            <ExpertView key={robot} robot={robot} instruction={instruction} />
          )}
        </Suspense>
      )}
    </div>
  )
}
