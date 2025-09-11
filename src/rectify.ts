/* eslint-disable @typescript-eslint/restrict-plus-operands */
import type CV from '@techstark/opencv-js'
import { angle, distance } from './utils'
import { useCV } from './image'

type Line = [number, number, number, number]

function intersection (l1: Line, l2: Line): [number, number] | undefined {
  const [
    x1,
    y1,
    x2,
    y2,
  ] = l1

  const [
    x3,
    y3,
    x4,
    y4,
  ] = l2

  const A1 = y2 - y1
  const B1 = x1 - x2
  const C1 = A1 * x1 + B1 * y1

  const A2 = y4 - y3
  const B2 = x3 - x4
  const C2 = A2 * x3 + B2 * y3

  const det = A1 * B2 - A2 * B1

  if (det === 0)
    return

  const x = (B2 * C1 - B1 * C2) / det
  const y = (A1 * C2 - A2 * C1) / det

  return [x, y]
}

function fitLine (points: CV.Point[], size: { width: number, height: number }, vertical = false): Line | undefined {
  // points: [[x1,y1],[x2,y2],...]
  const n = points.length

  if (n < 2)
    return

  if (vertical) {
    // fit x = cy + d
    let sumX  = 0
    let sumY  = 0
    let sumXY = 0
    let sumYY = 0

    for (const { x, y } of points) {
      sumX  += x
      sumY  += y
      sumXY += x * y
      sumYY += y * y
    }

    const c  = (n * sumXY - sumY * sumX) / (n * sumYY - sumY * sumY + 1e-6)
    const d  = (sumX - c * sumY) / n
    const y1 = 0
    const x1 = c * 0 + d
    const y2 = size.height
    const x2 = c * size.height + d

    return [
      x1,
      y1,
      x2,
      y2,
    ]
  } else {
    // fit y = ax + b
    let sumX  = 0
    let sumY  = 0
    let sumXY = 0
    let sumXX = 0

    for (const { x, y } of points) {
      sumX  += x
      sumY  += y
      sumXY += x * y
      sumXX += x * x
    }

    const a = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX + 1e-6)
    const b = (sumY - a * sumX) / n
    // extend line across image width
    const x1 = 0
    const y1 = a * 0 + b
    const x2 = size.width
    const y2 = a * size.width + b // adjust 500 to img width

    return [
      x1,
      y1,
      x2,
      y2,
    ]
  }
}

export async function getCorners (contour: CV.Mat): Promise<[CV.Point, CV.Point, CV.Point, CV.Point]> {
  const cv     = await useCV()
  const rect   = cv.minAreaRect(contour)
  const center = rect.center

  let tl     = new cv.Point(0, 0)
  let tlDist = 0

  let tr     = new cv.Point(0, 0)
  let trDist = 0

  let bl     = new cv.Point(0, 0)
  let blDist = 0

  let br     = new cv.Point(0, 0)
  let brDist = 0

  for (let i = 0; i < contour.data32S.length; i += 2) {
    const point = new cv.Point(contour.data32S[i], contour.data32S[i + 1])
    const dist  = distance(point, center)

    if (point.x < center.x && point.y < center.y) {
      if (dist > tlDist) {
        tl     = point
        tlDist = dist
      }
    } else if (point.x > center.x && point.y < center.y) {
      if (dist > trDist) {
        tr     = point
        trDist = dist
      }
    } else if (point.x < center.x && point.y > center.y) {
      if (dist > blDist) {
        bl     = point
        blDist = dist
      }
    } else if (point.x > center.x && point.y > center.y && dist > brDist) {
      br     = point
      brDist = dist
    }
  }

  const da = angle(center, tr)
  const db = angle(center, tl)
  const dc = angle(center, bl)
  const dd = angle(center, br)

  const top: CV.Point[]    = [tl, tr]
  const bottom: CV.Point[] = [bl, br]
  const left: CV.Point[]   = [tl, bl]
  const right: CV.Point[]  = [tr, br]

  for (let i = 0; i < contour.data32S.length; i += 2) {
    const point = new cv.Point(contour.data32S[i], contour.data32S[i + 1])
    const d     = angle(center, point)

    if (d > da && d < db)
      top.push(point)
    else if (d > db && d < dc)
      left.push(point)
    else if (d > dc && d < dd)
      bottom.push(point)
    else if (d > dd || d < da)
      right.push(point)
  }

  const t = fitLine(top, rect.size)
  const b = fitLine(bottom, rect.size)
  const l = fitLine(left, rect.size, true)
  const r = fitLine(right, rect.size, true)

  if (!t || !b || !l || !r) {
    return [
      tl,
      tr,
      bl,
      br,
    ]
  }

  const itl = intersection(t, l)
  const itr = intersection(t, r)
  const ibr = intersection(b, r)
  const ibl = intersection(b, l)

  if (!itl || !itr || !ibr || !ibl) {
    return [
      tl,
      tr,
      bl,
      br,
    ]
  }

  return [
    new cv.Point(itl[0], itl[1]),
    new cv.Point(itr[0], itr[1]),
    new cv.Point(ibl[0], ibl[1]),
    new cv.Point(ibr[0], ibr[1]),
  ]
}

async function findLargest (contours: CV.MatVector) {
  let idx     = -1
  let maxArea = Number.NEGATIVE_INFINITY

  const cv = await useCV()

  for (let i = 0; i < contours.size(); i++) {
    const cnt  = contours.get(i)
    const area = cv.contourArea(cnt)

    if (area > maxArea) {
      maxArea = area
      idx     = i
    }

    cnt.delete()
  }

  return idx
}

async function getIoU (a: CV.Mat, b: CV.Mat) {
  const cv           = await useCV()
  const intersection = new cv.Mat()
  const union        = new cv.Mat()

  cv.bitwise_and(a, b, intersection)
  cv.bitwise_or(a, b, union)

  const intersectionArea = cv.countNonZero(intersection)
  const unionArea        = cv.countNonZero(union)
  const iou              = unionArea > 0 ? intersectionArea / unionArea : 0

  intersection.delete()
  union.delete()

  return iou
}

export async function rectify (src: CV.Mat, dst: CV.Mat, area: CV.Rect, mask: CV.Mat) {
  const cv        = await useCV()
  const contours  = new cv.MatVector()
  const hierarchy = new cv.Mat()
  const crop      = src.roi(area)

  cv.cvtColor(crop, crop, cv.COLOR_RGBA2GRAY)

  for (let i = 0; i < crop.rows; i++) {
    for (let j = 0; j < crop.cols; j++)
      crop.ucharPtr(i, j)[0] = Math.floor(crop.ucharPtr(i, j)[0] * (mask.ucharPtr(i, j)[0] / 255))
  }

  cv.threshold(crop, crop, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
  cv.findContours(crop, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  let idx = await findLargest(contours)

  if (idx > -1) {
    const contour = contours.get(idx)
    const hull    = new cv.Mat()

    cv.convexHull(contour, hull, true, true)
    contours.set(idx, hull)
    contour.delete()

    const cMask = cv.Mat.zeros(mask.rows, mask.cols, cv.CV_8UC1)

    cv.drawContours(cMask, contours, idx, new cv.Scalar(255), cv.FILLED)

    const iou = await getIoU(cMask, mask)

    if (iou < 0.9) {
      cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

      idx = await findLargest(contours)

      if (idx > -1) {
        const contour = contours.get(idx)

        cv.convexHull(contour, hull, true, true)
        contours.set(idx, hull)

        contour.delete()
      }
    }

    cMask.delete()

    const corners    = await getCorners(hull)
    const isPortrait = area.height > area.width
    const dstWidth   = isPortrait ? 404 : 640
    const dstHeight  = isPortrait ? 640 : 404

    const srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, corners.flatMap(({ x, y }) => [area.x + x, area.y + y]))
    const dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
      0, // x1
      0, // y1
      dstWidth, // x2
      0, // y2
      0, // x3
      dstHeight, // y3
      dstWidth, // x4
      dstHeight, // y4
    ])

    const M = cv.getPerspectiveTransform(srcTri, dstTri)

    cv.warpPerspective(
      src,
      dst,
      M,
      new cv.Size(dstWidth, dstHeight),
      cv.INTER_CUBIC,
      cv.BORDER_CONSTANT,
      new cv.Scalar(),
    )

    if (isPortrait)
      cv.rotate(dst, dst, cv.ROTATE_90_COUNTERCLOCKWISE)

    srcTri.delete()
    dstTri.delete()
    hull.delete()
  }

  contours.delete()
  hierarchy.delete()
  crop.delete()
}
