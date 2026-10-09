const latestFeeds: (string | null)[] = [null, null]

export function attachFeed(index: number, image: HTMLImageElement | null) {
  const url = latestFeeds[index]
  if (image && url) image.src = url
}

export function publishFeed(index: number, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const previous = latestFeeds[index]
  latestFeeds[index] = url
  document.querySelectorAll<HTMLImageElement>(`img[data-feed="${index}"]`).forEach((node) => {
    node.src = url
  })
  if (previous) URL.revokeObjectURL(previous)
}
