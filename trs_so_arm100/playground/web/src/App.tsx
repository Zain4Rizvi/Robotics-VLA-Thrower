import { useGSAP } from "@gsap/react"
import gsap from "gsap"
import { lazy, Suspense, useEffect, useRef, useState, type FormEvent, type PointerEvent as ReactPointerEvent, type RefObject } from "react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group"
import { Skeleton } from "@/components/ui/skeleton"
import { usePlayground, type Link, type Phase } from "@/use-playground"

const ModelView = lazy(() => import("@/model/ModelView").then((mod) => ({ default: mod.ModelView })))

gsap.registerPlugin(useGSAP)

function linkBadge(link: Link, phase: Phase): { text: string; variant: "outline" | "destructive" } {
  if (phase === "error") return { text: "Error", variant: "destructive" }
  if (link !== "open") return { text: "Connecting", variant: "outline" }
  if (phase === "loading") return { text: "Loading", variant: "outline" }
  return { text: "Live", variant: "outline" }
}

function PolicyCamera({
  label,
  feed,
  imgRef,
}: {
  label: string
  feed: 0 | 1
  imgRef: RefObject<HTMLImageElement | null>
}) {
  const [shown, setShown] = useState(false)
  return (
    <figure className="flex w-44 flex-col gap-1.5">
      <div className="relative aspect-[4/3] overflow-hidden rounded-md border border-border bg-muted">
        {!shown && <Skeleton className="absolute inset-0 size-full rounded-none" />}
        <img
          ref={imgRef}
          data-feed={feed}
          alt={label}
          draggable={false}
          onLoad={() => setShown(true)}
          className="size-full object-cover"
        />
      </div>
      <figcaption className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
        {label}
      </figcaption>
    </figure>
  )
}

export default function App() {
  const root = useRef<HTMLDivElement>(null)
  const revealed = useRef(false)
  const revealHero = useRef<() => void>(() => {})
  const drag = useRef<{ id: number; x: number; y: number; button: number } | null>(null)
  const [draft, setDraft] = useState("")
  const [focused, setFocused] = useState(false)
  const [page, setPage] = useState<"scene" | "model">("scene")

  useGSAP(
    (_context, contextSafe) => {
      const motion = gsap.matchMedia()
      motion.add("(prefers-reduced-motion: no-preference)", () => {
        gsap.to("[data-enter]", {
          y: 0,
          duration: 0.5,
          stagger: 0.08,
          ease: "power2.out",
        })
      })
      if (!contextSafe) return
      revealHero.current = contextSafe(() => {
        if (revealed.current) return
        revealed.current = true
        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches
        if (reduce) {
          gsap.set("[data-hero]", { autoAlpha: 1 })
          return
        }
        gsap.to("[data-hero]", { autoAlpha: 1, duration: 0.5, ease: "power2.out" })
      })
    },
    { scope: root },
  )

  const { status, link, send, nudge, heroRef, cam1Ref, cam2Ref } = usePlayground(() => {
    revealHero.current()
  })

  useEffect(() => {
    if (!focused) setDraft(status.instruction)
  }, [focused, status.instruction])

  const ready = status.phase === "ready" && link === "open"
  const badge = linkBadge(link, status.phase)

  function onSubmit(event: FormEvent) {
    event.preventDefault()
    const text = draft.trim()
    if (!text || !ready) return
    send({ type: "instruction", text })
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
        <img
          ref={heroRef}
          data-hero
          alt=""
          draggable={false}
          className="pointer-events-none size-full object-cover select-none"
        />
      </div>

      <div className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-background/80 to-transparent" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-background/50 to-transparent" />

      {status.phase !== "ready" && (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center px-6">
          {status.phase === "error" ? (
            <Alert variant="destructive" className="pointer-events-auto w-full max-w-md bg-background/90 backdrop-blur-md">
              <AlertTitle>Not running</AlertTitle>
              <AlertDescription>{status.message}</AlertDescription>
            </Alert>
          ) : (
            <p className="font-mono text-[11px] tracking-[0.22em] text-muted-foreground uppercase">
              {link === "open" ? status.message || "Loading policy" : "Connecting"}
            </p>
          )}
        </div>
      )}

      <div className="absolute inset-x-0 bottom-0 z-10 flex flex-col gap-4 px-5 py-5 lg:flex-row lg:items-end lg:gap-8 lg:px-7 lg:py-7">
        <div data-enter className="flex shrink-0 flex-col gap-2.5">
          <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
            What the policy sees
          </p>
          <div className="flex gap-3">
            <PolicyCamera label="camera1" feed={0} imgRef={cam1Ref} />
            <PolicyCamera label="camera2" feed={1} imgRef={cam2Ref} />
          </div>
        </div>

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
                  onClick={() => send({ type: "layout" })}
                >
                  New layout
                </Button>
              </div>
            </Field>
          </FieldGroup>
        </form>
      </div>
      </div>

      <header className="absolute inset-x-0 top-0 z-30 flex items-start justify-between gap-8 px-7 py-6">
        <div className="flex flex-col gap-1.5">
          <p className="font-serif text-[17px] leading-none tracking-tight">SO-ARM100</p>
          <p className="text-[13px] leading-none text-muted-foreground">Red on green</p>
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
          <Badge variant={badge.variant}>{badge.text}</Badge>
          {status.seed != null && <span className="tabular-nums">seed {status.seed}</span>}
          {status.phase === "ready" && (
            <>
              <span aria-hidden="true">·</span>
              <span>{status.computing ? "Planning" : "Idle"}</span>
            </>
          )}
        </div>
      </header>
      {page === "model" && (
        <Suspense
          fallback={
            <p className="absolute inset-0 z-20 flex items-center justify-center font-mono text-[11px] tracking-[0.22em] text-muted-foreground uppercase">
              Opening
            </p>
          }
        >
          <ModelView instruction={status.instruction} />
        </Suspense>
      )}
    </div>
  )
}
