import { Canvas, useFrame } from "@react-three/fiber"
import { useEffect, useMemo, useRef, type RefObject } from "react"
import * as THREE from "three"

const bone = new THREE.Color("#f4f0e8")
const dim = new THREE.Color("#a39e94")

function Slabs({
  count,
  selected,
  yaw,
  onSelect,
}: {
  count: number
  selected: number
  yaw: RefObject<number>
  onSelect: (index: number) => void
}) {
  const group = useRef<THREE.Group>(null)
  const geometry = useMemo(() => new THREE.BoxGeometry(1.7, 0.95, 0.08), [])
  const texture = useMemo(() => {
    const canvas = document.createElement("canvas")
    canvas.width = 256
    canvas.height = 128
    const draw = canvas.getContext("2d")
    if (!draw) return null
    draw.fillStyle = "#ffffff"
    draw.fillRect(0, 0, canvas.width, canvas.height)
    draw.strokeStyle = "rgba(20, 19, 15, 0.22)"
    draw.lineWidth = 1
    for (let i = 0; i < 14; i += 1) {
      const x = 14 + i * 17
      draw.beginPath()
      draw.moveTo(x, 10)
      draw.lineTo(x, 118)
      draw.stroke()
    }
    const map = new THREE.CanvasTexture(canvas)
    map.colorSpace = THREE.SRGBColorSpace
    return map
  }, [])

  useEffect(() => {
    return () => {
      geometry.dispose()
      texture?.dispose()
    }
  }, [geometry, texture])

  useFrame(() => {
    if (group.current) group.current.rotation.y = yaw.current
  })

  const span = Math.max(count - 1, 1)

  return (
    <group ref={group} rotation={[0.12, 0, 0]}>
      {Array.from({ length: count }, (_, index) => {
        const active = index === selected
        return (
          <mesh
            key={index}
            geometry={geometry}
            position={[0, 0, (index - span / 2) * 0.16]}
            onClick={(event) => {
              event.stopPropagation()
              onSelect(index)
            }}
          >
            <meshStandardMaterial
              color={active ? bone : dim}
              map={texture ?? undefined}
              roughness={0.72}
              metalness={0}
            />
          </mesh>
        )
      })}
    </group>
  )
}

export function LayerStack({
  count,
  selected,
  onSelect,
}: {
  count: number
  selected: number
  onSelect: (index: number) => void
}) {
  const yaw = useRef(0.62)
  const drag = useRef<{ x: number; yaw: number } | null>(null)
  const moved = useRef(false)

  return (
    <div
      className="relative h-80 flex-1 cursor-grab touch-none active:cursor-grabbing"
      onPointerDown={(event) => {
        drag.current = { x: event.clientX, yaw: yaw.current }
        moved.current = false
        const move = (next: PointerEvent) => {
          const current = drag.current
          if (!current) return
          const dx = next.clientX - current.x
          if (Math.abs(dx) > 4) moved.current = true
          yaw.current = current.yaw + dx * 0.006
        }
        const up = () => {
          drag.current = null
          window.removeEventListener("pointermove", move)
          window.removeEventListener("pointerup", up)
        }
        window.addEventListener("pointermove", move)
        window.addEventListener("pointerup", up)
      }}
    >
      <Canvas camera={{ position: [1.7, 0.85, 7.4], fov: 24 }} gl={{ alpha: true, antialias: true }}>
        <ambientLight intensity={1.1} />
        <directionalLight position={[4, 5, 3]} intensity={1.6} />
        <Slabs
          count={count}
          selected={selected}
          yaw={yaw}
          onSelect={(index) => {
            if (moved.current) return
            onSelect(index)
          }}
        />
      </Canvas>
    </div>
  )
}

