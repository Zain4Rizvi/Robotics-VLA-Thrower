import { useCallback, useEffect, useRef, useState } from "react"

export type Phase = "loading" | "ready" | "error"

export type Status = {
  type: "status"
  phase: Phase
  instruction: string
  seed: number | null
  computing: boolean
  message: string
}

export type Link = "connecting" | "open" | "closed"

type Action = "orbit" | "pan" | "zoom"

const latestFeeds: (string | null)[] = [null, null, null]

export function attachFeed(index: number, image: HTMLImageElement | null) {
  const url = latestFeeds[index]
  if (image && url) image.src = url
}

const empty: Status = {
  type: "status",
  phase: "loading",
  instruction: "",
  seed: null,
  computing: false,
  message: "Connecting",
}

export function usePlayground(onHero: () => void) {
  const [status, setStatus] = useState<Status>(empty)
  const [link, setLink] = useState<Link>("connecting")
  const wsRef = useRef<WebSocket | null>(null)
  const heroRef = useRef<HTMLImageElement>(null)
  const cam1Ref = useRef<HTMLImageElement>(null)
  const cam2Ref = useRef<HTMLImageElement>(null)
  const onHeroRef = useRef(onHero)
  useEffect(() => {
    onHeroRef.current = onHero
  }, [onHero])

  const send = useCallback((message: object) => {
    const socket = wsRef.current
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message))
    }
  }, [])

  const pending = useRef({
    orbitX: 0,
    orbitY: 0,
    panX: 0,
    panY: 0,
    zoom: 0,
    frame: 0,
  })

  const nudge = useCallback(
    (action: Action, dx: number, dy: number) => {
      const bucket = pending.current
      if (action === "orbit") {
        bucket.orbitX += dx
        bucket.orbitY += dy
      } else if (action === "pan") {
        bucket.panX += dx
        bucket.panY += dy
      } else {
        bucket.zoom += dy
      }
      if (bucket.frame) return
      bucket.frame = window.requestAnimationFrame(() => {
        bucket.frame = 0
        if (bucket.orbitX || bucket.orbitY) {
          send({ type: "camera", action: "orbit", dx: bucket.orbitX, dy: bucket.orbitY })
          bucket.orbitX = 0
          bucket.orbitY = 0
        }
        if (bucket.panX || bucket.panY) {
          send({ type: "camera", action: "pan", dx: bucket.panX, dy: bucket.panY })
          bucket.panX = 0
          bucket.panY = 0
        }
        if (bucket.zoom) {
          send({ type: "camera", action: "zoom", dx: 0, dy: bucket.zoom })
          bucket.zoom = 0
        }
      })
    },
    [send],
  )
  const nudgeRef = useRef(nudge)
  useEffect(() => {
    nudgeRef.current = nudge
  }, [nudge])

  useEffect(() => {
    let stopped = false
    let socket: WebSocket | null = null
    let timer = 0
    const urls: (string | null)[] = [null, null, null]
    const images = [cam1Ref, cam2Ref, heroRef]

    const connect = () => {
      if (stopped) return
      const protocol = location.protocol === "https:" ? "wss" : "ws"
      socket = new WebSocket(`${protocol}://${location.host}/ws`)
      wsRef.current = socket
      socket.binaryType = "arraybuffer"
      socket.onopen = () => {
        if (!stopped) setLink("open")
      }
      socket.onclose = (event) => {
        if (stopped || event.code === 4000) return
        setLink("closed")
        timer = window.setTimeout(connect, 800)
      }
      socket.onmessage = (event) => {
        if (typeof event.data === "string") {
          setStatus(JSON.parse(event.data) as Status)
          return
        }
        const bytes = new Uint8Array(event.data as ArrayBuffer)
        const index = bytes[0]
        if (index > 2) return
        const url = URL.createObjectURL(new Blob([bytes.subarray(1)], { type: "image/jpeg" }))
        const previous = urls[index]
        urls[index] = url
        latestFeeds[index] = url
        const image = images[index].current
        if (image) image.src = url
        document.querySelectorAll<HTMLImageElement>(`img[data-feed="${index}"]`).forEach((node) => {
          if (node !== image) node.src = url
        })
        if (previous) URL.revokeObjectURL(previous)
        if (index === 2) onHeroRef.current()
      }
    }

    connect()
    const hero = document.querySelector<HTMLElement>("[data-scene]")
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const height = (event.currentTarget as HTMLElement).clientHeight || 1
      nudgeRef.current("zoom", 0, event.deltaY / height)
    }
    hero?.addEventListener("wheel", onWheel, { passive: false })

    return () => {
      stopped = true
      window.clearTimeout(timer)
      if (pending.current.frame) window.cancelAnimationFrame(pending.current.frame)
      hero?.removeEventListener("wheel", onWheel)
      socket?.close()
      wsRef.current = null
      for (const url of urls) {
        if (url) URL.revokeObjectURL(url)
      }
      latestFeeds.fill(null)
    }
  }, [])

  return { status, link, send, nudge, heroRef, cam1Ref, cam2Ref }
}
