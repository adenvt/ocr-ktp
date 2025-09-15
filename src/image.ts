import type CV from '@techstark/opencv-js'

let cv: typeof CV

export async function useCV (): Promise<typeof cv> {
  if (!cv) {
    const { default: CV } = await import('@techstark/opencv-js')

    cv = CV instanceof Promise
      ? await CV
      : CV
  }

  return cv
}

export async function openImage (input: string | Blob | CV.Mat) {
  const cv = await useCV()

  if (input instanceof cv.Mat)
    return input.clone()

  const image = new Image()

  image.src = input instanceof Blob
    ? URL.createObjectURL(input)
    : input

  await image.decode()

  if (input instanceof Blob)
    URL.revokeObjectURL(image.src)

  return cv.imread(image)
}

export async function autoContrast (src: CV.Mat, dst: CV.Mat, clipHistPercent = 10) {
  const cv = await useCV()
  // Convert to grayscale
  const gray   = new cv.Mat()
  const srcVec = new cv.MatVector()

  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY)
  srcVec.push_back(gray)

  // Calculate grayscale histogram
  const histSize = 256
  const ranges   = [0, 256]
  const hist     = new cv.Mat()
  const mask     = new cv.Mat()

  cv.calcHist(srcVec, [0], mask, hist, [histSize], ranges)

  // Calculate cumulative distribution from the histogram
  const accumulator = [hist.data32F[0]]

  for (let i = 1; i < histSize; i++)
    accumulator[i] = accumulator[i - 1] + hist.data32F[i]

  // Locate points to clip
  const maximum            = accumulator[histSize - 1]
  const clipHistPercentAbs = (clipHistPercent * maximum) / 100 / 2

  // Locate left cut
  let minimumGray = 0

  while (accumulator[minimumGray] < clipHistPercentAbs)
    minimumGray++

  // Locate right cut
  let maximumGray = histSize - 1

  while (accumulator[maximumGray] >= (maximum - clipHistPercentAbs))
    maximumGray--

  // Calculate alpha and beta values
  const alpha = 255 / (maximumGray - minimumGray)
  const beta  = -minimumGray * alpha

  // Apply alpha and beta to adjust brightness/contrast
  src.convertTo(dst, -1, alpha, beta)

  // Clean up
  srcVec.delete()
  gray.delete()
  hist.delete()
  mask.delete()
}

export function drawOverlay (img: CV.Mat, mask: CV.Mat, color: CV.Scalar, transparency = 0.5) {
  for (let i = 0; i < img.rows; i++) {
    for (let j = 0; j < img.cols; j++) {
      const a     = mask.ucharPtr(i, j)[0]
      const beta  = transparency * a / 255
      const alpha = 1 - beta
      const ptr   = img.ucharPtr(i, j)

      for (let k = 0; k < img.channels(); k++)
        ptr[k] = ptr[k] * alpha + color[k] * beta
    }
  }
}
