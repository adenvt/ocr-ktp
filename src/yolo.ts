import type { InferenceSession } from 'onnxruntime-common'
import { Tensor } from 'onnxruntime-common'
import type CV from '@techstark/opencv-js'
import { clamp, sigmoid } from './utils'
import { defu } from 'defu'
import {
  openImage,
  useCV,
} from './image'
import { useResizer } from './resizer'

interface YoloOption {
  model: InferenceSession,
  /**
   * Labels
   */
  labels: string[],
}

interface YoloResult {
  /**
   * Label
   */
  label: string,
  /**
   * Class index
   */
  classId: number,
  /**
   * Confidence
   */
  confidence: number,
  /**
   * Bounding box
   * Format: [x1, y1, x2, y2]
   */
  bbox: [number, number, number, number],
  /**
   * Segmentation mask
   */
  mask: Uint8ClampedArray,
}

interface YoloPredictOption {
  /**
   * Confidence threshold
   */
  confidence: number,
}

export function useYolo ({ model, labels }: YoloOption) {
  const inputMeta = model.inputMetadata[0] as InferenceSession.TensorValueMetadata
  const boxMeta   = model.outputMetadata[0] as InferenceSession.TensorValueMetadata
  const maskMeta  = model.outputMetadata[1] as InferenceSession.TensorValueMetadata

  async function predict (input: string | File | CV.Mat, options?: Partial<YoloPredictOption>) {
    const config = defu<YoloPredictOption, [YoloPredictOption]>(options, { confidence: 0.8 })

    // Preprocess
    const cv        = await useCV()
    const orig      = await openImage(input)
    const size      = orig.size()
    const inputSize = new cv.Size(inputMeta.shape[3] as number, inputMeta.shape[2] as number)
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

    // Inference
    const tensor = new Tensor('float32', blob.data32F, inputMeta.shape as number[])
    const result = await model.run({ [inputMeta.name]: tensor })

    const boxResult  = result[boxMeta.name] // [1, 300, 38]
    const maskResult = result[maskMeta.name] // [1, 32, 160, 160]

    // Postprocess
    const maxDetections = boxMeta.shape[1] as number
    const stride        = boxMeta.shape[2] as number

    const boxData  = boxResult.data as Float32Array // x1, y1, x2, y2, score, class, 32 maskCoeffs
    const maskData = maskResult.data as Float32Array

    const maskNum  = maskMeta.shape[1] as number
    const maskSize = new cv.Size(maskMeta.shape[3] as number, maskMeta.shape[2] as number)
    const maskDims = maskSize.width * maskSize.height

    const results: YoloResult[] = []

    for (let i = 0; i < maxDetections; i++) {
      const offset  = i * stride
      const score   = boxData[offset + 4]
      const classId = boxData[offset + 5]

      if (score < config.confidence || classId < 0 || classId >= labels.length)
        continue

      const x1 = clamp(Math.floor(boxData[offset]), 0, inputSize.width)
      const y1 = clamp(Math.floor(boxData[offset + 1]), 0, inputSize.height)
      const x2 = clamp(Math.ceil(boxData[offset + 2]), 0, inputSize.width)
      const y2 = clamp(Math.ceil(boxData[offset + 3]), 0, inputSize.height)

      const maskCoeffs  = boxData.slice(offset + 6, offset + 6 + maskNum)
      const maskData32F = Float32Array.from({ length: maskDims }, (_, i) => {
        let sum = 0

        for (let k = 0; k < maskNum; k++)
          sum += maskCoeffs[k] * maskData[k * maskDims + i]

        return sigmoid(sum)
      })

      const masRect = new cv.Rect(x1, y1, x2 - x1, y2 - y1)
      const maskRaw = cv.matFromArray(maskSize.height, maskSize.width, cv.CV_8UC1, Uint8ClampedArray.from(maskData32F, (c) => c * 255))

      // Because mask size only 160x160, we need to resize it equal to input size
      cv.resize(maskRaw, maskRaw, inputSize, 0, 0, cv.INTER_CUBIC)

      // Upscale to original size
      await resizer.revertMat(maskRaw, maskRaw)

      const rectObject = await resizer.revertRect(masRect)
      const maskObject = maskRaw.roi(rectObject)

      results.push({
        classId   : classId,
        label     : labels[classId],
        confidence: score,
        bbox      : [
          rectObject.x,
          rectObject.y,
          rectObject.x + rectObject.width,
          rectObject.y + rectObject.height,
        ],
        mask: Uint8ClampedArray.from({ length: maskObject.rows * maskObject.cols }, (_, idx) => {
          const i = Math.floor(idx / maskObject.cols)
          const j = idx % maskObject.cols

          return maskObject.ucharPtr(i, j)[0]
        }),
      })

      maskRaw.delete()
      maskObject.delete()
    }

    tensor.dispose()
    boxResult.dispose()
    maskResult.dispose()

    orig.delete()
    src.delete()
    raw.delete()
    blob.delete()

    return results
  }

  return { predict }
}
