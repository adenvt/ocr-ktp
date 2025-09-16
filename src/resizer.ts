import type CV from '@techstark/opencv-js'
import { useCV } from './image'
import { clamp } from './utils'

enum HorizontalAlign {
  LEFT = 0x00,
  CENTER = 0x01,
  RIGHT = 0x02,
}

enum VerticalAlign {
  TOP = 0x00,
  MID = 0x04,
  BOTTOM = 0x08,
}

export enum ResizeAlign {
  TOP_LEFT = VerticalAlign.TOP | HorizontalAlign.LEFT,
  TOP_CENTER = VerticalAlign.TOP | HorizontalAlign.CENTER,
  TOP_RIGHT = VerticalAlign.TOP | HorizontalAlign.RIGHT,
  MID_LEFT = VerticalAlign.MID | HorizontalAlign.LEFT,
  MID_CENTER = VerticalAlign.MID | HorizontalAlign.CENTER,
  MID_RIGHT = VerticalAlign.MID | HorizontalAlign.RIGHT,
  BOTTOM_LEFT = VerticalAlign.BOTTOM | HorizontalAlign.LEFT,
  BOTTOM_CENTER = VerticalAlign.BOTTOM | HorizontalAlign.CENTER,
  BOTTOM_RIGHT = VerticalAlign.BOTTOM | HorizontalAlign.RIGHT,
}

export function useResizer (srcSize: CV.Size, dstSize: CV.Size, align: ResizeAlign = ResizeAlign.MID_CENTER) {
  const ratio = Math.min(dstSize.width / srcSize.width, dstSize.height / srcSize.height)

  const width  = Math.floor(srcSize.width * ratio)
  const height = Math.floor(srcSize.height * ratio)

  const dw = dstSize.width - width
  const dh = dstSize.height - height

  let left = 0
  let top  = 0

  if (align & HorizontalAlign.CENTER)
    left = Math.floor(dw / 2)
  else if (align & HorizontalAlign.RIGHT)
    left = dw

  if (align & VerticalAlign.MID)
    top = Math.floor(dh / 2)
  else if (align & VerticalAlign.BOTTOM)
    top = dh

  async function scaleMat (src: CV.Mat, dst: CV.Mat, padColor?: CV.Scalar, interpolation?: CV.int) {
    const cv = await useCV()

    cv.resize(src, dst, new cv.Size(width, height), 0, 0, interpolation ?? cv.INTER_CUBIC)
    cv.copyMakeBorder(dst, dst, top, dh - top, left, dw - left, cv.BORDER_CONSTANT, padColor ?? new cv.Scalar(0, 0, 0, 255))
  }

  async function scaleRect (rect: CV.Rect) {
    const cv  = await useCV()
    const pt1 = await scalePoint(new cv.Point(rect.x, rect.y))
    const pt2 = await scalePoint(new cv.Point(rect.x + rect.width, rect.y + rect.height), true)

    return new cv.Rect(
      clamp(pt1.x, 0, width),
      clamp(pt1.y, 0, height),
      clamp(pt2.x - pt1.x, 0, width),
      clamp(pt2.y - pt1.y, 0, height),
    )
  }

  async function scalePoint (pt: CV.Point, roundUp = false) {
    const cv = await useCV()
    const x  = roundUp ? Math.ceil(pt.x * ratio) : Math.floor(pt.x * ratio)
    const y  = roundUp ? Math.ceil(pt.y * ratio) : Math.floor(pt.y * ratio)

    return new cv.Point(
      clamp(x, 0, width),
      clamp(y, 0, height),
    )
  }

  async function revertMat (src: CV.Mat, dst: CV.Mat, interpolation?: CV.int) {
    const cv   = await useCV()
    const rect = new cv.Rect(left, top, width, height)
    const roi  = src.roi(rect)

    cv.resize(roi, dst, srcSize, 0, 0, interpolation ?? cv.INTER_CUBIC)

    roi.delete()
  }

  async function revertRect (rect: CV.Rect) {
    const cv  = await useCV()
    const pt1 = await revertPoint(new cv.Point(rect.x, rect.y))
    const pt2 = await revertPoint(new cv.Point(rect.x + rect.width, rect.y + rect.height), true)

    return new cv.Rect(
      pt1.x,
      pt1.y,
      pt2.x - pt1.x,
      pt2.y - pt1.y,
    )
  }

  async function revertPoint (pt: CV.Point, roundUp = false) {
    const cv = await useCV()
    const x  = roundUp ? Math.ceil((pt.x - left) / ratio) : Math.floor((pt.x - left) / ratio)
    const y  = roundUp ? Math.ceil((pt.y - top) / ratio) : Math.floor((pt.y - top) / ratio)

    return new cv.Point(
      clamp(x, 0, srcSize.width),
      clamp(y, 0, srcSize.height),
    )
  }

  return {
    scaleMat,
    scaleRect,
    scalePoint,
    revertMat,
    revertRect,
    revertPoint,
  }
}
