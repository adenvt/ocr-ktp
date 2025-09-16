import { Tensor, type InferenceSession } from 'onnxruntime-common'
import type CV from '@techstark/opencv-js'
import { openImage, useCV } from './image'
import { ResizeAlign, useResizer } from './resizer'
import { argmax, softmax } from './utils'

interface CRNNOption {
  model: InferenceSession,
  vocab?: string,
}

const DEFAULT_VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ"

export function useCRNN ({ model, vocab = DEFAULT_VOCAB }: CRNNOption) {
  const inputMeta  = model.inputMetadata[0] as InferenceSession.TensorValueMetadata // ['batch_size', 3, 1024, 1024]
  const outputMeta = model.outputMetadata[0] as InferenceSession.TensorValueMetadata // ['batch_size', 1, 1024, 1024]

  async function predictBatch (inputs: Array<string | File | CV.Mat>) {
    const cv    = await useCV()
    const blobs = await Promise.all(inputs.map(async (input) => {
      const orig      = await openImage(input)
      const size      = orig.size()
      const inputSize = new cv.Size(inputMeta.shape[3] as number, inputMeta.shape[2] as number)
      const resizer   = useResizer(size, inputSize, ResizeAlign.TOP_LEFT)

      const src = new cv.Mat()
      const raw = new cv.Mat()

      await resizer.scaleMat(orig, src, new cv.Scalar(255, 255, 255, 255))

      cv.cvtColor(src, raw, cv.COLOR_RGBA2BGR)

      const blob = cv.blobFromImage(
        raw,
        1 / 255,
        raw.size(),
        new cv.Scalar(0, 0, 0),
        true, // swapRB
        false, // crop
      )

      src.delete()
      raw.delete()
      orig.delete()

      return blob
    }))

    const tensorShape = [blobs.length, ...inputMeta.shape.slice(1)] as number[]
    const tensorData  = new Float32Array(tensorShape.reduce((a, b) => a * b, 1))

    for (const blob of blobs) {
      const offset = blobs.indexOf(blob) * tensorShape.slice(1).reduce((a, b) => a * b, 1)

      tensorData.set(blob.data32F, offset)
      blob.delete()
    }

    // Inference
    const tensor = new Tensor('float32', tensorData, tensorShape)
    const result = await model.run({ [inputMeta.name]: tensor })

    const logitsResult = result[outputMeta.name]
    const logits       = logitsResult.data as Float32Array

    const [
      N,
      T,
      C,
    ] = logitsResult.dims

    const blank           = C - 1
    const texts: string[] = []

    for (let n = 0; n < N; n++) {
      const bestPath: number[] = []
      const stride             = n * T * C

      let minProb = 1

      for (let t = 0; t < T; t++) {
        const offset            = t * C + stride
        const logitsT: number[] = logits.slice(offset, offset + C) as unknown as number[]

        const probs = softmax(logitsT)
        const idx   = argmax(probs)

        bestPath.push(idx)

        if (probs[idx] < minProb)
          minProb = probs[idx]
      }

      // Collapse repeats and remove blanks
      const decoded: number[] = []

      let prev: number = -1

      for (const idx of bestPath) {
        if (idx !== prev && idx !== blank)
          decoded.push(idx)

        prev = idx
      }

      texts.push(decoded.map((idx) => vocab[idx]).join(''))
    }

    tensor.dispose()
    logitsResult.dispose()

    return texts
  }

  async function predict (input: string | File | CV.Mat) {
    const batch = await predictBatch([input])

    return batch[0]
  }

  return { predict, predictBatch }
}
