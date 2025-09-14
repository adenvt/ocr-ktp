/* eslint-disable @typescript-eslint/no-misused-promises */
import MODEL from '../model/yolov11n.ort?url'
import { useYolo } from '../src/yolo'
import { rectify } from '../src/rectify'
import {
  autoContrast,
  drawOverlay,
  openImage,
  useCV,
} from '../src/image'

function toSeconds (ms: number) {
  return `${(ms / 1000).toFixed(2)}s`
}

document.addEventListener('DOMContentLoaded', async () => {
  const note      = document.querySelector('#note') as HTMLInputElement
  const fileinput = document.querySelector('#fileinput') as HTMLInputElement
  const input     = document.querySelector('#input') as HTMLCanvasElement
  const crop      = document.querySelector('#crop') as HTMLCanvasElement
  const output    = document.querySelector('#output') as HTMLCanvasElement
  const filter    = document.querySelector('#filter') as HTMLCanvasElement
  const grayed    = document.querySelector('#grayed') as HTMLCanvasElement

  note.textContent = 'Initiating OpenCV ...'

  const cv = await useCV()

  note.textContent = 'Initiating ONNX Runtime ...'

  const ort = await import('onnxruntime-web')

  note.textContent = 'Initiating YOLO Model ...'

  const model = await ort.default.InferenceSession.create(MODEL)
  const yolo  = useYolo({
    model : model,
    labels: [
      'kartu',
      'ktp',
      'ktp-fc',
    ],
  })

  note.textContent   = 'Ready!'
  fileinput.disabled = false

  const colors = [
    new cv.Scalar(0, 255, 0, 255),
    new cv.Scalar(0, 0, 255, 255),
    new cv.Scalar(255, 0, 0, 255),
  ]

  let startTime  = 0
  let detectTime = 0

  fileinput.addEventListener('input', async () => {
    const file = fileinput.files?.[0]

    if (!file)
      return

    startTime = performance.now()

    const src = await openImage(file)

    cv.imshow(input, src)

    const results = await yolo.predict(src)

    detectTime = performance.now() - startTime

    const result = results.find((item) => {
      const width  = item.bbox[2] - item.bbox[0]
      const height = item.bbox[3] - item.bbox[1]

      const w = Math.max(width, height)
      const h = Math.min(width, height)
      const r = w / h

      return item.classId === 1
        && w > 100
        && r > 1
        && r < 2
    })

    if (result) {
      const dst  = new cv.Mat()
      const rect = new cv.Rect(
        result.bbox[0],
        result.bbox[1],
        result.bbox[2] - result.bbox[0],
        result.bbox[3] - result.bbox[1],
      )

      const mask = cv.matFromArray(rect.height, rect.width, cv.CV_8UC1, result.mask)
      const roi  = src.roi(rect)
      const temp = new cv.Mat()
      const chn  = new cv.MatVector()

      cv.split(roi, chn)
      chn.set(3, mask)
      cv.merge(chn, temp)

      cv.imshow(crop, temp)

      await rectify(src, dst, rect, mask)

      const contrast = new cv.Mat()
      const gray     = new cv.Mat()

      await autoContrast(dst, contrast, 5)

      drawOverlay(roi, mask, colors[result.classId], 0.4)

      cv.cvtColor(contrast, gray, cv.COLOR_RGBA2GRAY)
      cv.rectangle(src, new cv.Point(rect.x, rect.y), new cv.Point(rect.x + rect.width, rect.y + rect.height), colors[result.classId], 2)
      cv.putText(src, result.label, new cv.Point(rect.x + 10, rect.y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255, 255, 255, 255), 1, cv.LINE_AA, false)

      cv.imshow(input, src)
      cv.imshow(output, dst)
      cv.imshow(filter, contrast)
      cv.imshow(grayed, gray)

      roi.delete()
      chn.delete()
      mask.delete()
      temp.delete()

      contrast.delete()
      gray.delete()
      dst.delete()

      note.textContent = `Success, Time: ${toSeconds(performance.now() - startTime)} (Detect: ${toSeconds(detectTime)})`
    } else {
      const blank = cv.Mat.zeros(404, 640, cv.CV_8UC4)

      cv.imshow(crop, blank)
      cv.imshow(output, blank)
      cv.imshow(filter, blank)
      cv.imshow(grayed, blank)

      blank.delete()

      note.textContent = 'Failed, Cannot find KTP in this image'
    }

    src.delete()
  })
})
