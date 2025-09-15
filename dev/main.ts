/* eslint-disable @typescript-eslint/no-misused-promises */
import OBJECT_DETECT_MODEL from '../model/object-detect.onnx?url'
import TEXT_DETECT_MODEL from '../model/text-detect.onnx?url'

import { useYolo } from '../src/yolo'
import { rectify } from '../src/rectify'
import {
  autoContrast,
  drawOverlay,
  openImage,
  useCV,
} from '../src/image'
import { useDBNet } from '../src/dbnet'

function toSeconds (ms: number) {
  return `${(ms / 1000).toFixed(2)}s`
}

function log (message: string) {
  (document.querySelector('#note') as HTMLParagraphElement).textContent = message
}

async function main () {
  const fileinput = document.querySelector('#fileinput') as HTMLInputElement
  const input     = document.querySelector('#input') as HTMLCanvasElement
  const crop      = document.querySelector('#crop') as HTMLCanvasElement
  const output    = document.querySelector('#output') as HTMLCanvasElement
  const filter    = document.querySelector('#filter') as HTMLCanvasElement
  const grayed    = document.querySelector('#grayed') as HTMLCanvasElement
  const textArea  = document.querySelector('#text-mask') as HTMLCanvasElement

  log('Initiating OpenCV ...')

  const cv = await useCV()

  log('Initiating ONNX Runtime ...')

  const ort = await import('onnxruntime-web')

  log('Initiating YOLO Model ...')

  const yolo = useYolo({
    model : await ort.default.InferenceSession.create(OBJECT_DETECT_MODEL),
    labels: [
      'kartu',
      'ktp',
      'ktp-fc',
    ],
  })

  fileinput.disabled = false

  const dbnet = useDBNet({ model: await ort.InferenceSession.create(TEXT_DETECT_MODEL) })

  log('Ready!')

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

      await rectify(src, dst, rect, mask, new cv.Size(1024, 646))

      const contrast = new cv.Mat()
      const gray     = new cv.Mat()

      await autoContrast(dst, contrast, 5)

      drawOverlay(roi, mask, colors[result.classId], 0.4)

      cv.cvtColor(contrast, gray, cv.COLOR_RGBA2GRAY)
      cv.cvtColor(gray, gray, cv.COLOR_GRAY2RGBA)
      cv.rectangle(src, new cv.Point(rect.x, rect.y), new cv.Point(rect.x + rect.width, rect.y + rect.height), colors[result.classId], 2)
      cv.putText(src, result.label, new cv.Point(rect.x + 10, rect.y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, new cv.Scalar(255, 255, 255, 255), 1, cv.LINE_AA, false)

      const boxes = await dbnet.predict(gray)
      const text  = new cv.Mat(gray.rows, gray.cols, dst.type(), new cv.Scalar(255, 255, 255, 255))

      for (const box of boxes) {
        const rect = new cv.Rect(
          box[0],
          box[1],
          box[2] - box[0],
          box[3] - box[1],
        )

        const srcRoi = gray.roi(rect)
        const dstRoi = text.roi(rect)

        srcRoi.copyTo(dstRoi)

        srcRoi.delete()
        dstRoi.delete()
      }

      cv.imshow(input, src)
      cv.imshow(output, dst)
      cv.imshow(filter, contrast)
      cv.imshow(grayed, gray)
      cv.imshow(textArea, text)

      roi.delete()
      chn.delete()
      mask.delete()
      temp.delete()

      const totalTime = performance.now() - startTime

      log(`Success, Time: ${toSeconds(totalTime)} (Detect: ${toSeconds(detectTime)})`)

      contrast.delete()
      gray.delete()
      dst.delete()
    } else {
      const blank = cv.Mat.zeros(404, 640, cv.CV_8UC4)

      cv.imshow(crop, blank)
      cv.imshow(output, blank)
      cv.imshow(filter, blank)
      cv.imshow(grayed, blank)

      blank.delete()

      log('Failed, Cannot find KTP in this image')
    }

    src.delete()
  })
}

document.addEventListener('DOMContentLoaded', () => {
  main()
    .catch((error) => {
      const message: string = error instanceof Error ? error.message : error

      log(`Error, ${message}`)
    })
})
