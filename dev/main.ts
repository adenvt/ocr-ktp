/* eslint-disable @typescript-eslint/no-misused-promises */
import MODEL from '../model/yolov11n.onnx?url'
import { useYolo } from '../src/yolo'
import { rectify } from '../src/rectify'
import {
  autoContrast,
  openImage,
  useCV,
} from '../src/image'

document.addEventListener('DOMContentLoaded', async () => {
  const cv   = await useCV()
  const yolo = useYolo(MODEL, [
    'kartu',
    'ktp',
    'ktp-fc',
  ])

  const colors = [
    new cv.Scalar(0, 255, 0, 255),
    new cv.Scalar(0, 0, 255, 255),
    new cv.Scalar(255, 0, 0, 255),
  ]

  const dur       = document.querySelector('#duration') as HTMLInputElement
  const fileinput = document.querySelector('#fileinput') as HTMLInputElement
  const input     = document.querySelector('#input') as HTMLCanvasElement
  const crop      = document.querySelector('#crop') as HTMLCanvasElement
  const output    = document.querySelector('#output') as HTMLCanvasElement
  const filter    = document.querySelector('#filter') as HTMLCanvasElement
  const grayed    = document.querySelector('#grayed') as HTMLCanvasElement

  let startTime = 0

  fileinput.addEventListener('input', async () => {
    const file = fileinput.files?.[0]

    if (!file)
      return

    startTime = performance.now()

    const src = await openImage(file)

    cv.imshow(input, src)

    const results = await yolo.predict(src)
    const dst     = new cv.Mat()

    for (const result of results) {
      if (result.classIdx !== 1)
        continue

      const rect = new cv.Rect(
        result.bbox[0],
        result.bbox[1],
        result.bbox[2] - result.bbox[0],
        result.bbox[3] - result.bbox[1],
      )

      const w = Math.max(rect.width, rect.height)
      const h = Math.min(rect.width, rect.height)
      const r = w / h

      if (w < 100 || r < 1 || r > 2)
        continue

      const mask = cv.matFromArray(rect.height, rect.width, cv.CV_8UC1, result.mask)
      const roi  = src.roi(rect)
      const temp = new cv.Mat()
      const chn  = new cv.MatVector()

      cv.split(roi, chn)
      chn.set(3, mask)
      cv.merge(chn, temp)

      await rectify(src, dst, rect, mask)

      cv.rectangle(src, new cv.Point(rect.x, rect.y), new cv.Point(rect.x + rect.width, rect.y + rect.height), colors[result.classIdx], 2)
      cv.imshow(crop, temp)

      roi.delete()
      chn.delete()
      mask.delete()
      temp.delete()

      break
    }

    const contrast = new cv.Mat()
    const gray     = new cv.Mat()

    await autoContrast(dst, contrast, 5)

    cv.cvtColor(contrast, gray, cv.COLOR_RGBA2GRAY)
    // cv.threshold(gray, gray, 0, 255, cv.THRESH_OTSU)

    cv.imshow(input, src)
    cv.imshow(output, dst)
    cv.imshow(filter, contrast)
    cv.imshow(grayed, gray)

    src.delete()
    dst.delete()
    contrast.delete()
    gray.delete()

    dur.textContent = `${((performance.now() - startTime) / 1000).toFixed(2)}s`
  })
})
