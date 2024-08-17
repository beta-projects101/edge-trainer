import * as imgly from '@imgly/background-removal'

const userImage = document.getElementById('image-upload');

const canvas = document.getElementById('canvas');
const fullCanvas = document.getElementById('full-canvas');
const ctx = canvas.getContext('2d');
const fullCtx = fullCanvas.getContext('2d')
const pixelCanvas = document.createElement('canvas');
const pixelCtx = pixelCanvas.getContext('2d');

const img = document.getElementById('hidden-image');
const cutoutImg = document.getElementById('cutout-image');

const removeBtn = document.getElementById('remove')
const styleSelect = document.getElementById('style')
const status = document.getElementById('status')
const cooldown = document.getElementById('cooldown')
const uplaodContainer = document.getElementById('upload-container')
const speed = document.getElementById('speed-select')

const state = {
  allPixels: [],
  intervalId: null,
  t: {},
  model: null,
  images: [],
  imageIndex: 0,
}

const options = {
  modelPath: '../models/default-f16/model.json',
  imagePath: '../samples/example.jpg',
  minScore: 0.38,
  maxResults: 50,
  iouThreshold: 0.5,
  outputNodes: ['output1', 'output2', 'output3'],
  resolution: [0, 0],
  user: {
      map: {
          pixel
      },
      pixelSize: 3.5,
      censorType: 'pixel'
  },
  DIFFICULTY: 5,
  SPEED: 500,
  ALPHA_THRESHOLD: 90,
  R: 0,
  G: 0,
  B: 0,
  A: 255,
  removeStyle: 'lines',
  started: false,
  cooldown: 8000
};

const labels = [
  'exposed anus',  //0
  'exposed armpits',  //1
  'belly',  //2
  'exposed belly',  //3
  'buttocks',  //4
  'exposed buttocks',  //5
  'female face',  //6
  'male face',  //7
  'feet',  //8
  'exposed feet',  //9
  'breast',  //10
  'exposed breast',  //11
  'vagina',  //12
  'exposed vagina',  //13
  'male breast',  //14
  'exposed male breast',  //15
];

const betaSettings = {
  pathetic: {
      person: [1,2,3,4,6,7,8,9,14,15],
      sexy: [5,10,12],
      nude: [0,11,13],
  },
  original: {
      person: [1,2,3,4,6,7,8,9,14,15],
      sexy: [1,3,5,9,10,12,14,15],
      nude: [0,11,13],
  },
  ultimate: {
      person: [7, 6],
      sexy: [1,2,4,8,10,14,15],
      nude: [0,3,5,9,11,12,13],
  }
}

let composite = betaSettings.ultimate

async function processPrediction(boxesTensor, scoresTensor, classesTensor, inputTensor) {
  const boxes = await boxesTensor.array();
  const scores = await scoresTensor.data();
  const classes = await classesTensor.data();
  const nmsT = await tf.image.nonMaxSuppressionAsync(boxes[0], scores, options.maxResults, options.iouThreshold, options.minScore); // sort & filter results
  const nms = await nmsT.data();
  tf.dispose(nmsT);
  const parts = [];
  for (const i in nms) { // create body parts object
      const id = parseInt(i);
      parts.push({
          score: scores[i],
          id: classes[id],
          class: labels[classes[id]],
          box: [
              Math.trunc(boxes[0][id][0]),
              Math.trunc(boxes[0][id][1]),
              Math.trunc((boxes[0][id][3] - boxes[0][id][1])),
              Math.trunc((boxes[0][id][2] - boxes[0][id][0])),
          ],
      });
  }
  const result = {
      input: { width: inputTensor.shape[2], height: inputTensor.shape[1] },
      person: parts.filter((a) => composite.person.includes(a.id)).length > 0,
      sexy: parts.filter((a) => composite.sexy.includes(a.id)).length > 0,
      nude: parts.filter((a) => composite.nude.includes(a.id)).length > 0,
      parts,
  };
  return result;
}

function pixel({left=0, top=0, width=0, height=0}) {
  if (width === 0 || height === 0 || canvas.width === 0 || canvas.height === 0)
      return;
  pixelCanvas.width = width
  pixelCanvas.height = height

  pixelCtx.drawImage(fullCanvas, left, top, width, height, 0, 0, width ,height);

  let size = options.user.pixelSize / 100,
  w = pixelCanvas.width * size,
  h = pixelCanvas.height * size;

  pixelCtx.drawImage(pixelCanvas, 0, 0, w, h);

  pixelCtx.msImageSmoothingEnabled = false;
  pixelCtx.mozImageSmoothingEnabled = false;
  pixelCtx.webkitImageSmoothingEnabled = false;
  pixelCtx.imageSmoothingEnabled = false;

  pixelCtx.drawImage(pixelCanvas, 0, 0, w, h, 0, 0, pixelCanvas.width, pixelCanvas.height);

  fullCtx.drawImage(pixelCanvas, left, top, width, height);

  pixelCtx.msImageSmoothingEnabled = true;
  pixelCtx.mozImageSmoothingEnabled = true;
  pixelCtx.webkitImageSmoothingEnabled = true;
  pixelCtx.imageSmoothingEnabled = true;
}

function processParts(res) {
  for (const obj of res.parts) { // draw all detected objects
      if (composite.nude.includes(obj.id))
          options.user.map[options.user.censorType]({ left: obj.box[0], top: obj.box[1], width: obj.box[2], height: obj.box[3] });
      if (composite.sexy.includes(obj.id))
          options.user.map[options.user.censorType]({ left: obj.box[0], top: obj.box[1], width: obj.box[2], height: obj.box[3] });
  }
}

async function processLoop() {
  if (canvas.width > 0 && state.model) {
      status.innerText = 'censoring image'
      state.t.buffer = await tf.browser.fromPixelsAsync(img);
      state.t.resize = (options.resolution[0] > 0 && options.resolution[1] > 0 && (options.resolution[0] !== img.width || options.resolution[1] !== img.height)) // do we need to resize
          ? tf.image.resizeNearestNeighbor(state.t.buffer, [options.resolution[1], options.resolution[0]])
          : state.t.buffer;
      state.t.cast = tf.cast(state.t.resize, 'float32');
      state.t.batch = tf.expandDims(state.t.cast, 0);

      [state.t.boxes, state.t.scores, state.t.classes] = await state.model.executeAsync(state.t.batch, options.outputNodes);

      const res = await processPrediction(state.t.boxes, state.t.scores, state.t.classes, state.t.cast);
      await tf.browser.toPixels(state.t.resize, fullCanvas);
      processParts(res);
  }
}

styleSelect.addEventListener('change', () => {
  options.removeStyle = styleSelect.value
})

function resizeCanvas() {
  canvas.width = parseInt(window.getComputedStyle(img).width);
  canvas.height = parseInt(window.getComputedStyle(img).height);
  fullCanvas.width = canvas.width
  fullCanvas.height = canvas.height

  let styles = `
    width: ${canvas.width};
    height: ${canvas.height};
    left: 0; 
    right: 0; 
    margin-left: auto; 
    margin-right: auto; 
    position: absolute;
  `
  canvas.setAttribute('style', styles+'z-index: 3;')
  fullCanvas.setAttribute('style', styles+'z-index: 2;')
}

async function removeBg() {
  uplaodContainer.style.display = 'none'
  status.innerText = 'Removing background\nThis could take a few moments'
  try {
    const blob = await imgly.removeBackground(img.src)
  
    const cutout = URL.createObjectURL(blob);
  
    cutoutImg.src = cutout
    status.innerText = ''
  } catch (error) {
    status.innerText = 'An error occured,\nplease try again'
    console.log(error);
  }
}

function replaceColors() {
  const imgData = ctx.getImageData(0,0,canvas.width,canvas.height)

  for (let i = 0; i < imgData.data.length; i+=4) {

    if(imgData.data[i+3] >= options.ALPHA_THRESHOLD) {
      imgData.data[i] = options.R
      imgData.data[i+1] = options.G
      imgData.data[i+2] = options.B
      imgData.data[i+3] = options.A

      state.allPixels.push(i)
    }
  }

  ctx.putImageData(imgData, 0,0)
}

function isCanvasBlank() {
  const pixelBuffer = new Uint32Array(
    ctx.getImageData(0, 0, canvas.width, canvas.height).data.buffer
  );

  return !pixelBuffer.some(color => color !== 0);
}

function wait() {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve()
    }, options.cooldown)
  })
}

async function checkCanvas(imgData1, imgData2, callback) {
  for (let i = 0; i < imgData1.data.length; ++i) {
    if (imgData1.data[i] != imgData2.data[i]) {
      return
    }
  }

  if (!isCanvasBlank()) {
    console.log('canvas not blank, trying to remove pixels again');
    callback()
  } else {
    clearInterval(state.intervalId)
    state.intervalId = null
    state.imageIndex++
    if (state.imageIndex == state.images.length) {
      resetCanvas()
      resetState()
      toggleVisibility()
      status.style.display = 'none'
    } else {
      cooldown.style.visibility = 'visible'
      await wait()
      cooldown.style.visibility = 'hidden'
      runImageProcessing()
    }
  }
}

function removeBoxes() {
  const imgData1 = ctx.getImageData(0,0,canvas.width,canvas.height)

  let boxWidth = (canvas.width / 100) * (options.DIFFICULTY * 2)
  let boxHeight = (canvas.height / 100) * (options.DIFFICULTY * 2)

  let x = Math.floor(Math.random()*canvas.width)
  let y = Math.floor(Math.random()*canvas.height)

  let xOffset = Math.round(x-(boxWidth/2))
  let yOffset = Math.round(y-(boxHeight/2))
  
  ctx.clearRect(xOffset, yOffset, boxWidth, boxHeight)

  const imgData2 = ctx.getImageData(0,0,canvas.width,canvas.height)
  
  checkCanvas(imgData1, imgData2, removeBoxes)
}

const getTopBottomContent = () => {
  console.log(state.allPixels[0]);
  let top = (state.allPixels[0] / 4) / canvas.width
  let bottom = (state.allPixels[state.allPixels.length-1] / 4) / canvas.width
  
  ctx.fillStyle = 'green'
  ctx.fillRect(0, top, canvas.width, 1)
  ctx.fillRect(0, bottom, canvas.width, 1)
}

function removeLines() {
  const imgData1 = ctx.getImageData(0,0,canvas.width,canvas.height)

  let y = Math.floor(Math.random()*canvas.height)
  
  let boxHeight = (canvas.height / 100) * options.DIFFICULTY

  ctx.clearRect(0, y, canvas.width, boxHeight)

  const imgData2 = ctx.getImageData(0,0,canvas.width,canvas.height)

  checkCanvas(imgData1, imgData2, removeLines)
}

function removeRandomArea() {
  console.log(options.removeStyle);
  switch (options.removeStyle) {
    case 'boxes':
      removeBoxes()
      break;
    case 'lines':
      removeLines()
      break;
  }
}

function drawCanvas (imgToDraw, context) {
  console.log('drawing canvas');
  context.drawImage(imgToDraw, 0,0, canvas.width, canvas.height);
}

function resetCanvas() {
  ctx.clearRect(0,0,canvas.width,canvas.height)
  fullCtx.clearRect(0,0,canvas.width,canvas.height)
  pixelCtx.clearRect(0,0,canvas.width,canvas.height)
  state.allPixels = []
  uplaodContainer.style.display = 'block'
  if (state.intervalId) {
    clearInterval(state.intervalId)
    state.intervalId = null
  }
}

function resetState() {
  state.allPixels = []
  state.images = []
  state.imageIndex = 0
}

function toggleVisibility() {
  if (canvas.style.display == 'none') {
    canvas.style.display = 'block'
    fullCanvas.style.display = 'block'
  } else {
    canvas.style.display = 'none'
    fullCanvas.style.display = 'none'
  }
}

function runImageProcessing () {
  status.style.display = 'block'
  img.src = URL.createObjectURL(state.images[state.imageIndex]);
  resetCanvas()
  img.onload = async() => {
    resizeCanvas()
    await removeBg()

    options.resolution[0] = img.width
    options.resolution[1] = img.height

    cutoutImg.onload = async() => {
      drawCanvas(cutoutImg, ctx)
      replaceColors()
      drawCanvas(img, fullCtx)  
      toggleVisibility()
      await processLoop()
      toggleVisibility()
      if (state.imageIndex != 0) {
        state.intervalId = setInterval(removeRandomArea, options.SPEED)
      }
    }
  }
}

userImage.addEventListener('change', (e) => {
  resetState()
  resetCanvas()
  state.images = [...e.target.files]
  runImageProcessing()
})

removeBtn.addEventListener('click', () => {
  state.intervalId = setInterval(removeRandomArea, options.SPEED)
})

speed.addEventListener('change', (e) => {
  options.SPEED = Math.abs(e.target.value * 500)
})

async function main() {
  if (tf.engine().registryFactory.webgpu && navigator?.gpu)
      await tf.setBackend('webgpu');
  else
      await tf.setBackend('webgl');
  tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true); // doubles the performance
  await tf.ready();

  state.model = await tf.loadGraphModel(options.modelPath);
}
main()