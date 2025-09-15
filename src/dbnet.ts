import type CV from '@techstark/opencv-js'
import { Tensor, type InferenceSession } from 'onnxruntime-common'
import { openImage, useCV } from './image'
import { useResizer } from './resizer'
import defu from 'defu'
import { sigmoid } from './utils'

interface DBNetOption {
  model: InferenceSession,
}

interface DBNetPredictOption {
  /**
   * Confidence threshold
   * @default 0.3
   */
  confidence: number,
  /**
   * Unclip ratio
   * @default 2
   */
  unclipRatio: number,
  /**
   * Box min size
   * @default 64
   */
  boxMinArea: number,
}

type DBNetResult = Array<[number, number, number, number]>

export function useDBNet ({ model }: DBNetOption) {
  const inputMeta  = model.inputMetadata[0] as InferenceSession.TensorValueMetadata // ['batch_size', 3, 1024, 1024]
  const outputMeta = model.outputMetadata[0] as InferenceSession.TensorValueMetadata // ['batch_size', 1, 1024, 1024]

  async function predict (input: string | File | CV.Mat, options?: Partial<DBNetPredictOption>): Promise<DBNetResult> {
    const config = defu<DBNetPredictOption, [DBNetPredictOption]>(options, {
      confidence : 0.3,
      unclipRatio: 2,
      boxMinArea : 64,
    })

    // Preprocess
    const cv        = await useCV()
    const orig      = await openImage(input)
    const size      = orig.size()
    const inputSize = new cv.Size(inputMeta.shape[2] as number, inputMeta.shape[3] as number)
    const resizer   = useResizer(size, inputSize)

    const src = new cv.Mat()
    const raw = new cv.Mat()

    await resizer.scaleMat(orig, src)

    cv.cvtColor(src, raw, cv.COLOR_RGBA2BGR)

    const blob = cv.blobFromImage(
      raw,
      1 / 255,
      raw.size(),
      new cv.Scalar(0, 0, 0),
      true, // swapRB
      false, // crop
    )

    const tensor = new Tensor('float32', blob.data32F, [1, ...inputMeta.shape.slice(1)] as number[])
    const result = await model.run({ [inputMeta.name]: tensor })

    const maskResult = result[outputMeta.name] // [1, 3, 1024, 1024]
    const maskData   = maskResult.data as Float32Array

    // Postprocess
    const mask = cv.matFromArray(
      outputMeta.shape[2] as number,
      outputMeta.shape[3] as number,
      cv.CV_8UC1,
      Uint8ClampedArray.from(maskData, (c) => sigmoid(c) >= config.confidence ? 255 : 0),
    )

    const contours  = new cv.MatVector()
    const hierarchy = new cv.Mat()

    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    const boxes: Array<[number, number, number, number]> = []

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i)
      const area    = cv.contourArea(contour)
      const length  = cv.arcLength(contour, true)

      if (area > config.boxMinArea && length > 0) {
        const d    = Math.ceil(area * config.unclipRatio / length)
        const box  = cv.boundingRect(contour)
        const rect = await resizer.revertRect(new cv.Rect(
          box.x - d,
          box.y - d,
          box.width + d * 2,
          box.height + d * 2,
        ))

        boxes.push([
          rect.x,
          rect.y,
          rect.x + rect.width,
          rect.y + rect.height,
        ])
      }

      contour.delete()
    }

    tensor.dispose()
    maskResult.dispose()

    mask.delete()
    blob.delete()
    src.delete()
    raw.delete()
    orig.delete()

    return boxes
  }

  return { predict }
}
