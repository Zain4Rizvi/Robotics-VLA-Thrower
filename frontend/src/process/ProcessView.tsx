import { useEffect, useRef, useState } from "react"

import { runs, type Clip, type Plot } from "@/process/runs"

export function ProcessView() {
  const root = useRef<HTMLDivElement>(null)
  const [active, setActive] = useState(runs[0].id)

  useEffect(() => {
    const container = root.current
    if (!container) return
    const sections = [...container.querySelectorAll<HTMLElement>("section[id]")]
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0]
        if (visible?.target.id) setActive(visible.target.id)
      },
      { root: container, rootMargin: "-20% 0px -55% 0px", threshold: [0.15, 0.4] },
    )
    for (const section of sections) observer.observe(section)
    return () => observer.disconnect()
  }, [])

  function jump(id: string) {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches
    document.getElementById(id)?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" })
  }

  return (
    <div className="absolute inset-0 z-20 flex flex-col bg-background pt-20 pl-36">
      <div ref={root} className="min-h-0 flex-1 overflow-auto">
        <nav className="sticky top-0 z-10 flex flex-wrap gap-x-5 gap-y-2 border-b border-border bg-background/95 px-7 py-3 font-mono text-[11px] tracking-[0.14em] uppercase backdrop-blur-sm">
          {runs.map((run) => (
            <button
              key={run.id}
              type="button"
              aria-current={active === run.id ? "true" : undefined}
              className={`shrink-0 whitespace-nowrap ${active === run.id ? "text-foreground" : "text-muted-foreground hover:text-foreground"}`}
              onClick={() => jump(run.id)}
            >
              {run.index} {run.nav}
            </button>
          ))}
        </nav>

        <header className="max-w-2xl px-7 pt-10 pb-2">
          <h1 className="font-serif text-3xl tracking-tight">How this arm was trained</h1>
          <p className="mt-4 text-[15px] leading-relaxed text-foreground/90">
            Six runs, in the order they were tried. Each one starts from what the last measurement got wrong. The task is a
            sentence: stack one cube on the other. A vision-language-action model is meant to do that from cameras and the
            sentence, without being given the cube positions.
          </p>
          <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">
            The model is SmolVLA. Its vision tower and its language model stay frozen, so training does not change them.
            What trains is the action expert, the part that outputs joint targets. The training data is 50 successful stacks
            from a scripted expert that is allowed to see the cubes. A result is the learned policy tried on 10 layouts it
            did not train on. A stack counts only if the cube rests on the other for half a second.
          </p>
          <p className="mt-3 text-[15px] leading-relaxed text-muted-foreground">
            Every clip is that test, at the training step named on it. The small view in the corner is the wrist camera. These
            are not the demonstrations the model trained on.
          </p>
        </header>

        {runs.map((run) => (
          <section key={run.id} id={run.id} className="scroll-mt-12 border-t border-border px-7 py-10">
            <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">{run.index}</p>
            <h2 className="mt-2 font-serif text-2xl tracking-tight">{run.title}</h2>
            <p className="mt-3 max-w-2xl text-[15px] leading-relaxed">{run.skim}</p>
            <p className="mt-5 font-serif text-4xl tracking-tight tabular-nums">{run.count}</p>
            <p className="mt-1 max-w-xl text-[13px] text-muted-foreground">{run.countLabel}</p>

            {run.clips.length > 0 && (
              <div className="mt-6">
                {run.clipNote && <p className="mb-3 max-w-2xl text-[13px] text-muted-foreground">{run.clipNote}</p>}
                <div
                  className={
                    run.clips.length === 3
                      ? "grid gap-4 lg:grid-cols-3"
                      : run.clips.length === 2
                        ? "grid max-w-4xl gap-4 sm:grid-cols-2"
                        : "grid max-w-xl gap-4"
                  }
                >
                  {run.clips.map((clip) => (
                    <Rollout key={clip.src} clip={clip} />
                  ))}
                </div>
              </div>
            )}

            {run.plots.length > 0 && (
              <div className={run.plots.length > 1 ? "mt-6 grid gap-4 lg:grid-cols-2" : "mt-6 grid max-w-3xl gap-4"}>
                {run.plots.map((plot) => (
                  <Figure key={plot.src} plot={plot} />
                ))}
              </div>
            )}

            <div className="mt-6 max-w-2xl space-y-3">
              {run.story.map((paragraph) => (
                <p key={paragraph} className="text-[15px] leading-relaxed text-foreground/85">
                  {paragraph}
                </p>
              ))}
            </div>

            {run.limit && (
              <div className="mt-6 max-w-2xl">
                <p className="text-[15px] leading-relaxed text-foreground/85">{run.limit.text}</p>
                <div className="mt-4 max-w-md">
                  <Figure plot={run.limit.plot} />
                </div>
              </div>
            )}

            {run.next && (
              <p className="mt-5 max-w-2xl border-l border-border pl-4 text-[15px] leading-relaxed text-foreground/90">
                {run.next}
              </p>
            )}
          </section>
        ))}
      </div>
    </div>
  )
}

function Rollout({ clip }: { clip: Clip }) {
  const ref = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    const video = ref.current
    if (!video) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry) return
        if (entry.isIntersecting) void video.play().catch(() => {})
        else video.pause()
      },
      { threshold: 0.45 },
    )
    observer.observe(video)
    return () => observer.disconnect()
  }, [])

  return (
    <figure className="flex flex-col gap-2">
      <div className="relative">
        <p className="absolute top-2 left-2 z-10 bg-background/85 px-2 py-1 font-mono text-[10px] tracking-[0.14em] text-foreground uppercase">
          Policy rollout
        </p>
        <video
          ref={ref}
          src={clip.src}
          muted
          loop
          playsInline
          controls
          preload="metadata"
          className="aspect-video w-full bg-black"
        />
      </div>
      <figcaption className="text-[13px] leading-snug">
        <span className="text-foreground">{clip.step}.</span> <span className="text-muted-foreground">{clip.detail}</span>
      </figcaption>
    </figure>
  )
}

function Figure({ plot }: { plot: Plot }) {
  return (
    <figure className="flex flex-col bg-white text-neutral-900">
      <figcaption className="px-4 pt-4 pb-2">
        <p className="text-[15px] leading-snug font-medium">{plot.title}</p>
        <p className="mt-1 text-[12px] leading-relaxed text-neutral-600">{plot.axes}</p>
      </figcaption>
      <img src={plot.src} alt={`${plot.title}. ${plot.axes}`} className="w-full" />
    </figure>
  )
}
