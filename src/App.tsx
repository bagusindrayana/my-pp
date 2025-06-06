import './App.css'

import React, { useState, useRef, useEffect, type ChangeEvent, type CSSProperties, useCallback } from 'react';
import * as faceapi from 'face-api.js';
import * as ort from 'onnxruntime-web';
import { BeforeAfterSlider } from './components/BeforeAfterSlider';

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    // Set up a timer to update the debounced value after the specified delay
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    // Clean up the timer if the value changes (e.g., user keeps typing)
    // or if the component unmounts.
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]); // Only re-call effect if value or delay changes

  return debouncedValue;
}

const FACE_API_MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights';

interface EyePoint {
  x: number;
  y: number;
  _x?: number; // face-api.js might have internal properties
  _y?: number;
}

interface EyeMetrics {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  width: number;
  height: number;
  centerX: number;
  centerY: number;
}

// --- Constants ---
const MODEL_PATH = "/models/Anzhc_Eyes_seg_hd (2) (2).onnx"; // Ensure this model is in your `public` folder
const MODEL_INPUT_SIZE = { width: 1024, height: 1024 };
const CONF_SCORE_THRESHOLD = 0.25;
const MASK_VALUE_THRESHOLD = 0.5;
const IOU_NMS_THRESHOLD = 0.45;

// --- Interfaces ---
interface OrtTensor {
  dims: readonly number[];
  type: string;
  data: Float32Array | Uint8Array | Int32Array; // Add other types if your model uses them
  size: number;
}


interface OriginalImageShape {
  width: number;
  height: number;
}

interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface ModelCoords {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

interface DetectionCandidate {
  boxForNMS: Box;
  score: number;
  modelCoords: ModelCoords;
  maskCoeffs: Float32Array;
}

interface BoundingBoxDisplay {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: string;
}


function App() {

  const [uploadFile, setUploadFile] = useState<File | null>(null);

  const [message, setMessage] = useState<string>('Please upload an image.');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isFaceApiModelsLoading, setIsFaceApiModelsLoading] = useState<boolean>(true);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [isError, setIsError] = useState<boolean>(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  // --- State ---
  const [ortSession, setOrtSession] = useState<ort.InferenceSession | null>(null);
  const [originalImageElement, setOriginalImageElement] = useState<HTMLImageElement | null>(null);
  const [originalImageShape, setOriginalImageShape] = useState<OriginalImageShape>({ width: 0, height: 0 });
  const [status, setStatus] = useState<string>("Initializing...");
  // const [detectionCount, setDetectionCount] = useState<string>("Detected Eyes: 0");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [errorModal, setErrorModal] = useState<string | null>(null);



  const [contentType, setContentType] = useState<string | null>("real-person");

  const [glowEffect, setGlowEffect] = useState<boolean>(true);
  const [bgColor, setBgColor] = useState<string>("#ff0000");
  const debouncedColor = useDebounce(bgColor, 200);

  const imageTempUrl = useRef<string>(null);

  const barCanvas = useRef<HTMLCanvasElement>(null);

  const resultImageElement = useRef<HTMLImageElement | null>(null);



  // --- Load face-api.js Models ---
  useEffect(() => {


    async function initOrtSession() {
      try {
        if (!ort) {
          setStatus("Error: ONNX Runtime is not available. Ensure it's loaded.");
          console.error("ONNX Runtime (ort) not found. Load it via CDN or install the package.");
          setErrorModal("ONNX Runtime is not available. Please ensure it's correctly loaded in your application (e.g., via a script tag in index.html).");
          return;
        }
        // Set wasm paths before creating the session.
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0-dev.20250206-d981b153d3/dist/";
        // ort.env.wasm.wasmPaths = {
        //   "wasm": "/ort-wasm-simd-threaded.jsep.wasm",
        //   "mjs": "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs"
        // };

        setMessage("Loading anime model...");
        const session = await ort.InferenceSession.create(MODEL_PATH);
        setOrtSession(session);
        setStatus("Model loaded. Ready to segment.");
        setMessage('Models loaded. Please upload an image.');
        console.log("ONNX session created successfully.");
      } catch (e: any) {
        console.log(e);
        // console.error(`Failed to create ONNX session: ${e}`);
        setStatus(`Error loading model: ${e.message}. Check console.`);
        setErrorModal(`Failed to load model: ${e.message}. Ensure '${MODEL_PATH}' is accessible in your public folder and the ONNX runtime is configured correctly.`);
      }
    }


    const loadFaceApiModels = async () => {
      setMessage('Loading face detection models...');
      setIsLoading(true); // Use main loader for this initial step too
      setIsFaceApiModelsLoading(true);
      setDownloadUrl(null);
      setIsError(false);
      try {
        if (typeof faceapi === 'undefined') {
          throw new Error('face-api.js not loaded. Make sure the CDN script is in your HTML.');
        }
        await faceapi.nets.ssdMobilenetv1.loadFromUri(FACE_API_MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(FACE_API_MODEL_URL);

        await initOrtSession();

        setMessage('Models loaded. Please upload an image.');
        setIsFaceApiModelsLoading(false);
      } catch (error: any) {
        console.error("Error loading face-api.js models:", error);
        setMessage(`Error loading face detection models: ${error.message}. Please refresh.`);
        setIsError(true);
      } finally {
        setIsLoading(false); // Stop loading indicator after this phase
      }
    };




    loadFaceApiModels();


  }, []);

  useEffect(() => {
    drawResult(resultImageElement.current);
  }, [glowEffect, debouncedColor])

  async function uploadFileWithGivenFileObject(fileObject: any) {
    const formData = new FormData();
    // Use the actual file name from the File object
    formData.append('file', fileObject, fileObject.name);

    const url = 'https://tmpfiles.org/api/v1/upload';

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // Try to get more error details from the response
        let errorDetails = `HTTP error! status: ${response.status}`;
        try {
          const errorData = await response.json(); // Or response.text()
          errorDetails += ` - ${JSON.stringify(errorData)}`;
        } catch (e) {
          // Ignore if response is not JSON or text
        }
        throw new Error(errorDetails);
      }

      const data = await response.json();
      // tmpfiles.org returns the URL in data.data.url
      // if (data && data.data && data.data.url) {
      //   alert('File uploaded! URL: ' + data.data.url);
      // } else {
      //   alert('File uploaded, but URL not found in response.');
      //   console.log('Full response:', data);
      // }
      return data;
    } catch (error) {
      console.error('Error uploading file:', error);
      return null;

    }
  }

  const getEyeMetrics = (eyePoints: EyePoint[] | undefined): EyeMetrics | null => {
    if (!eyePoints || eyePoints.length === 0) return null;
    const xCoords = eyePoints.map(p => p.x);
    const yCoords = eyePoints.map(p => p.y);
    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);
    return {
      minX, maxX, minY, maxY,
      width: maxX - minX,
      height: maxY - minY,
      centerX: (minX + maxX) / 2,
      centerY: (minY + maxY) / 2
    };
  };

  const drawCombinedSensorBar = (
    ctx: CanvasRenderingContext2D,
    leftEyePoints: EyePoint[],
    rightEyePoints: EyePoint[]
  ) => {
    const leftEyeMetrics = getEyeMetrics(leftEyePoints);
    const rightEyeMetrics = getEyeMetrics(rightEyePoints);

    if (!leftEyeMetrics || !rightEyeMetrics) {
      console.warn("Could not get metrics for one or both eyes.");
      return;
    }

    const midPointX = (leftEyeMetrics.centerX + rightEyeMetrics.centerX) / 2;
    const midPointY = (leftEyeMetrics.centerY + rightEyeMetrics.centerY) / 2;
    const deltaX = rightEyeMetrics.centerX - leftEyeMetrics.centerX;
    const deltaY = rightEyeMetrics.centerY - leftEyeMetrics.centerY;
    const angle = Math.atan2(deltaY, deltaX);
    const distanceBetweenEyeCenters = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    const averageEyeWidth = (leftEyeMetrics.width + rightEyeMetrics.width) / 2;
    const sensorWidth = distanceBetweenEyeCenters + averageEyeWidth * 3;
    const averageEyeHeight = (leftEyeMetrics.height + rightEyeMetrics.height) / 2;
    const sensorHeight = averageEyeHeight * 3.5;

    ctx.save();
    ctx.translate(midPointX, midPointY);
    ctx.rotate(angle);
    ctx.fillStyle = 'black';
    ctx.fillRect(-sensorWidth / 2, -sensorHeight / 2, sensorWidth, sensorHeight);
    ctx.restore();
  };



  const loaderStyle: CSSProperties = {
    border: '8px solid #f3f3f3',
    borderTop: '8px solid #3498db',
    borderRadius: '50%',
    width: '60px',
    height: '60px',
    animation: 'spin 1s linear infinite',
    // margin: '20px auto',
  };

  const keyframes = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;



  // --- Image Preprocessing ---
  const preprocess = useCallback(async (imageElement: HTMLImageElement): Promise<OrtTensor> => {
    const canvas = document.createElement('canvas');
    canvas.width = MODEL_INPUT_SIZE.width;
    canvas.height = MODEL_INPUT_SIZE.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error("Failed to get 2D context for preprocessing canvas");

    ctx.drawImage(imageElement, 0, 0, MODEL_INPUT_SIZE.width, MODEL_INPUT_SIZE.height);

    const imageData = ctx.getImageData(0, 0, MODEL_INPUT_SIZE.width, MODEL_INPUT_SIZE.height);
    const { data, width, height } = imageData;
    const float32Data = new Float32Array(3 * width * height);
    const R: number[] = [], G: number[] = [], B: number[] = [];

    for (let i = 0; i < data.length; i += 4) {
      R.push(data[i] / 255.0);
      G.push(data[i + 1] / 255.0);
      B.push(data[i + 2] / 255.0);
    }
    float32Data.set(R);
    float32Data.set(G, R.length);
    float32Data.set(B, R.length + G.length);

    return new ort.Tensor('float32', float32Data, [1, 3, height, width]);
  }, []);

  // --- IoU Calculation ---
  const calculateIoU = (box1: Box, box2: Box): number => {
    const xA = Math.max(box1.x1, box2.x1);
    const yA = Math.max(box1.y1, box2.y1);
    const xB = Math.min(box1.x2, box2.x2);
    const yB = Math.min(box1.y2, box2.y2);
    const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    if (intersectionArea === 0) return 0;
    const box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    return intersectionArea / (box1Area + box2Area - intersectionArea);
  };

  // --- Non-Maximum Suppression (NMS) ---
  const applyNMS = (detections: DetectionCandidate[], iouThreshold: number): DetectionCandidate[] => {
    detections.sort((a, b) => b.score - a.score);
    const selected: DetectionCandidate[] = [];
    const active = new Array(detections.length).fill(true);
    for (let i = 0; i < detections.length; i++) {
      if (active[i]) {
        selected.push(detections[i]);
        active[i] = false;
        for (let j = i + 1; j < detections.length; j++) {
          if (active[j]) {
            const iou = calculateIoU(detections[i].boxForNMS, detections[j].boxForNMS);
            if (iou > iouThreshold) {
              active[j] = false;
            }
          }
        }
      }
    }
    return selected;
  };

  // --- Postprocessing with NMS ---
  const postprocessSegmentationWithNMS = useCallback((
    detectionOutputData: Float32Array,
    detectionOutputDims: readonly number[],
    maskFeaturesData: Float32Array,
    maskFeaturesDims: readonly number[],
    currentOriginalShape: OriginalImageShape
  ): { mask: Uint8Array; count: number; boxes: BoundingBoxDisplay[] } => {
    const numProposals = detectionOutputDims[2];
    const numMaskCoeffs = maskFeaturesDims[1];
    const maskProtoH = maskFeaturesDims[2];
    const maskProtoW = maskFeaturesDims[3];

    const candidateProposals: DetectionCandidate[] = [];
    for (let i = 0; i < numProposals; i++) {
      const proposalOffset = i;
      const cx = detectionOutputData[0 * numProposals + proposalOffset];
      const cy = detectionOutputData[1 * numProposals + proposalOffset];
      const w = detectionOutputData[2 * numProposals + proposalOffset];
      const h = detectionOutputData[3 * numProposals + proposalOffset];
      const confidence = detectionOutputData[4 * numProposals + proposalOffset];

      if (confidence < CONF_SCORE_THRESHOLD) continue;

      const currentMaskCoeffs = new Float32Array(numMaskCoeffs);
      for (let k = 0; k < numMaskCoeffs; k++) {
        currentMaskCoeffs[k] = detectionOutputData[(5 + k) * numProposals + proposalOffset];
      }

      candidateProposals.push({
        boxForNMS: { x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 },
        score: confidence,
        modelCoords: { cx, cy, w, h },
        maskCoeffs: currentMaskCoeffs
      });
    }

    const finalProposalsAfterNMS = applyNMS(candidateProposals, IOU_NMS_THRESHOLD);

    const detectedEyesCount = finalProposalsAfterNMS.length;
    const finalCombinedMask = new Uint8Array(currentOriginalShape.width * currentOriginalShape.height).fill(0);
    const finalBoundingBoxesForDisplay: BoundingBoxDisplay[] = [];

    for (const proposal of finalProposalsAfterNMS) {
      const { modelCoords, maskCoeffs, score } = proposal;
      const { cx, cy, w, h } = modelCoords;

      const instanceMaskLowRes = new Float32Array(maskProtoH * maskProtoW);
      for (let y_proto = 0; y_proto < maskProtoH; y_proto++) {
        for (let x_proto = 0; x_proto < maskProtoW; x_proto++) {
          let sumVal = 0;
          for (let k = 0; k < numMaskCoeffs; k++) {
            sumVal += maskCoeffs[k] * maskFeaturesData[k * maskProtoH * maskProtoW + y_proto * maskProtoW + x_proto];
          }
          instanceMaskLowRes[y_proto * maskProtoW + x_proto] = sumVal;
        }
      }
      const instanceMaskSigmoid = instanceMaskLowRes.map(val => 1 / (1 + Math.exp(-val)));

      const x1Model = cx - w / 2; const y1Model = cy - h / 2;
      const x2Model = cx + w / 2; const y2Model = cy + h / 2;

      const modelInputW = MODEL_INPUT_SIZE.width; const modelInputH = MODEL_INPUT_SIZE.height;

      const boxXminProtoSlice = Math.max(0, Math.min(Math.floor((x1Model / modelInputW) * maskProtoW), maskProtoW));
      const boxYminProtoSlice = Math.max(0, Math.min(Math.floor((y1Model / modelInputH) * maskProtoH), maskProtoH));
      const boxXmaxProtoSlice = Math.max(0, Math.min(Math.ceil((x2Model / modelInputW) * maskProtoW), maskProtoW));
      const boxYmaxProtoSlice = Math.max(0, Math.min(Math.ceil((y2Model / modelInputH) * maskProtoH), maskProtoH));

      if (boxXminProtoSlice >= boxXmaxProtoSlice || boxYminProtoSlice >= boxYmaxProtoSlice) continue;

      const croppedProtoWidth = boxXmaxProtoSlice - boxXminProtoSlice;
      const croppedProtoHeight = boxYmaxProtoSlice - boxYminProtoSlice;
      const croppedSigmoidMask = new Float32Array(croppedProtoWidth * croppedProtoHeight);

      for (let y_crop = 0; y_crop < croppedProtoHeight; y_crop++) {
        for (let x_crop = 0; x_crop < croppedProtoWidth; x_crop++) {
          croppedSigmoidMask[y_crop * croppedProtoWidth + x_crop] =
            instanceMaskSigmoid[(boxYminProtoSlice + y_crop) * maskProtoW + (boxXminProtoSlice + x_crop)];
        }
      }
      if (croppedSigmoidMask.length === 0) continue;
      const binaryCroppedMask = croppedSigmoidMask.map(val => (val > MASK_VALUE_THRESHOLD ? 1 : 0));

      const originalWImg = currentOriginalShape.width; const originalHImg = currentOriginalShape.height;
      const origImgX1 = Math.max(0, Math.min(Math.floor((x1Model / modelInputW) * originalWImg), originalWImg));
      const origImgY1 = Math.max(0, Math.min(Math.floor((y1Model / modelInputH) * originalHImg), originalHImg));
      const origImgX2 = Math.max(0, Math.min(Math.ceil((x2Model / modelInputW) * originalWImg), originalWImg));
      const origImgY2 = Math.max(0, Math.min(Math.ceil((y2Model / modelInputH) * originalHImg), originalHImg));

      const targetWOnOrig = origImgX2 - origImgX1; const targetHOnOrig = origImgY2 - origImgY1;

      if (targetWOnOrig <= 0 || targetHOnOrig <= 0) continue;

      finalBoundingBoxesForDisplay.push({
        x: origImgX1, y: origImgY1, width: targetWOnOrig, height: targetHOnOrig,
        confidence: score.toFixed(3)
      });

      const tempMaskCanvas = document.createElement('canvas');
      tempMaskCanvas.width = croppedProtoWidth; tempMaskCanvas.height = croppedProtoHeight;
      const tempMaskCtx = tempMaskCanvas.getContext('2d');
      if (!tempMaskCtx) continue;
      const tempMaskImageData = tempMaskCtx.createImageData(croppedProtoWidth, croppedProtoHeight);
      for (let j = 0; j < binaryCroppedMask.length; j++) {
        const val = binaryCroppedMask[j] * 255;
        tempMaskImageData.data[j * 4] = val; tempMaskImageData.data[j * 4 + 1] = val;
        tempMaskImageData.data[j * 4 + 2] = val; tempMaskImageData.data[j * 4 + 3] = 255;
      }
      tempMaskCtx.putImageData(tempMaskImageData, 0, 0);

      const resizedMaskCanvas = document.createElement('canvas');
      resizedMaskCanvas.width = targetWOnOrig; resizedMaskCanvas.height = targetHOnOrig;
      const resizedMaskCtx = resizedMaskCanvas.getContext('2d');
      if (!resizedMaskCtx) continue;
      resizedMaskCtx.imageSmoothingEnabled = false; // Use nearest neighbor for sharp mask
      resizedMaskCtx.drawImage(tempMaskCanvas, 0, 0, targetWOnOrig, targetHOnOrig);
      const resizedMaskImageData = resizedMaskCtx.getImageData(0, 0, targetWOnOrig, targetHOnOrig);

      for (let y_final = 0; y_final < targetHOnOrig; y_final++) {
        for (let x_final = 0; x_final < targetWOnOrig; x_final++) {
          if (resizedMaskImageData.data[(y_final * targetWOnOrig + x_final) * 4] > 128) { // Check alpha channel or intensity
            const finalX = origImgX1 + x_final; const finalY = origImgY1 + y_final;
            if (finalX < originalWImg && finalY < originalHImg) {
              finalCombinedMask[finalY * originalWImg + finalX] = 1;
            }
          }
        }
      }
    }

    return {
      mask: finalCombinedMask.map(val => val * 255) as Uint8Array, // Convert {0,1} to {0,255}
      count: detectedEyesCount,
      boxes: finalBoundingBoxesForDisplay
    };
  }, [MODEL_INPUT_SIZE, CONF_SCORE_THRESHOLD, MASK_VALUE_THRESHOLD, IOU_NMS_THRESHOLD]);


  const handleImageUpload = (event: ChangeEvent<HTMLInputElement>) => {

    if (!event.target.files || !event.target.files[0]) {
      setMessage('No file selected.');
      setIsError(false);
     
      if (canvasRef.current) { // Clear canvas if no file is selected after a previous image
        const ctx = canvasRef.current.getContext('2d');
        ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      return;
    }


    setMessage('Upload image...');
    setIsLoading(true);
    setIsProcessing(true);

    const file = event.target.files?.[0];
    if (file) {
      setDownloadUrl(null);
      setOriginalUrl(null);
      uploadFileWithGivenFileObject(file).then((d) => {
        if (d.data.url) {
          const urlFile = d.data.url.replace("http://tmpfiles.org", "https://tmpfiles.org").replace("https://tmpfiles.org", "https://tmpfiles.org/dl")

          imageTempUrl.current = urlFile;

          setUploadFile(file);
          const reader = new FileReader();
          reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
              setOriginalImageElement(img);
              const shape = { width: img.naturalWidth, height: img.naturalHeight };
              setOriginalImageShape(shape);

              const canvas = canvasRef.current;
              canvas!.width = shape.width;
              canvas!.height = shape.height;
              const ctx = canvas!.getContext('2d');
              ctx?.drawImage(img, 0, 0, shape.width, shape.height);
              setOriginalUrl(canvas!.toDataURL('image/png'));

              setIsLoading(false);
              setIsProcessing(false);
              setMessage('Image ready!');

            }
            img.onerror = () => {
              setStatus("Error loading image file.");
              setErrorModal("Could not load the selected image. Please try a different file.");
              setIsProcessing(false);
            }
            if (e.target?.result) {
              img.src = e.target.result as string;
            } else {
              setStatus("Error reading image file.");
              setErrorModal("Could not read the selected image file.");
              setIsProcessing(false);
              setIsLoading(false);
            }
          };
          reader.onerror = () => {
            setStatus("Error reading file.");
            setErrorModal("There was an error reading the file.");
            setIsProcessing(false);
            setIsLoading(false);
          };
          reader.readAsDataURL(file);
        }
      });

    }

  };

  function getBoundingBoxCenter(boundingBox: any) {
    const { xMin, yMin, xMax, yMax } = boundingBox;

    const centerX = (xMin + xMax) / 2;
    const centerY = (yMin + yMax) / 2;

    return { centerX, centerY };
  }

  const addGlow = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    if (!glowEffect) {
      console.log("No glow");
      return;
    }
    ctx.drawImage(canvas, 0, 0);
    ctx.filter = "brightness(" + 1.5 + ") blur(" + 20 + "px)";   // note: not all browsers support this yet (FF/Chrome OK)
    ctx.globalCompositeOperation = "lighten";
    ctx.globalAlpha = +0.8;
    ctx.drawImage(canvas, 0, 0);
    ctx.filter = "none";  // reset
    ctx.globalCompositeOperation = "source-over";
  }


  const drawResult = (image: any) => {
    if (image == null) {
      return;
    }
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');

    if (!canvas || !ctx) {
      setMessage('Canvas not ready.');
      setIsError(true);
      return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.reset();

    // 1. Fill background with red
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    tempCtx!.drawImage(image, 0, 0, canvas.width, canvas.height);

    try {
      const imageData = tempCtx!.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data; // This is an array of R,G,B,A values

      // --- Pixel manipulation for B&W high contrast ---
      for (let i = 0; i < data.length; i += 4) {
        // Convert to grayscale (average method)
        let r = data[i];
        let g = data[i + 1];
        let b = data[i + 2];


        // 1. Apply Grayscale "Amount"
        const bwAmount = 1; // Convert 0-100 to 0.0-1.0
        // Standard luminosity calculation for grayscale
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;

        r = r * (1 - bwAmount) + gray * bwAmount;
        g = g * (1 - bwAmount) + gray * bwAmount;
        b = b * (1 - bwAmount) + gray * bwAmount;

        const contrastParameter = (80 - 50) * 5.1; // (100-50)*5.1 = 255; (0-50)*5.1 = -255

        // Contrast formula: factor * (value - 128) + 128
        // Factor calculation based on GIMP's levels tool or similar:
        // http://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
        const factor = (259 * (contrastParameter + 255)) / (255 * (259 - contrastParameter));

        r = factor * (r - 128) + 128;
        g = factor * (g - 128) + 128;
        b = factor * (b - 128) + 128;

        // Clamp values to the 0-255 range and round them
        data[i] = Math.max(0, Math.min(255, Math.round(r)));
        data[i + 1] = Math.max(0, Math.min(255, Math.round(g)));
        data[i + 2] = Math.max(0, Math.min(255, Math.round(b)));
      }

      // Put the modified data onto the contrast canvas
      tempCtx!.putImageData(imageData, 0, 0);

      ctx!.drawImage(tempCanvas, 0, 0);



      if (barCanvas.current) {
        ctx!.drawImage(barCanvas.current, 0, 0);
      }

      addGlow(ctx, canvas);

      setDownloadUrl(canvas.toDataURL('image/png'));



    } catch (error) {
      console.error("Error processing image data:", error);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0);
    }
  }

  async function fetchImageAsFile(imageUrl: string, outputFileName: string = "image.png"): Promise<File | null> {
    try {
      // const apiUrl = "https://api.ryzumi.vip/api/ai/removebg?url=";
      const apiUrl = "https://api.ferdev.my.id/tools/removebg?link=";

      const response = await fetch(apiUrl + imageUrl);

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
      }
      let blob: Blob

      const contentType = response.headers.get('content-type');

      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();

        const response2 = await fetch(data.data);

        if (!response2.ok) {
          throw new Error(`API request failed with status ${response2.status}: ${response2.statusText}`);
        }
        blob = await response2.blob();
      } else {
        blob = await response.blob();
      }



      // Try to get the file type from the blob, default to 'image/png' if not available
      // The API is for removing background, so PNG is a likely output.
      const fileType = blob.type || 'image/png';

      // Create a File object
      const imageFile = new File([blob], outputFileName, { type: fileType });

      return imageFile;
    } catch (error) {
      console.error("Error fetching image:", error);
      return null;
    }
  }

  const processReal = async () => {
    if (isFaceApiModelsLoading) {
      setMessage('Face detection models are still loading. Please wait.');
      setIsError(false);
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');

    if (!canvas || !ctx) {
      setMessage('Canvas not ready.');
      setIsError(true);
      return;
    }

    setMessage('Processing image: Initializing...');
    setIsProcessing(true);
    setIsLoading(true);
    setDownloadUrl(null);
    setIsError(false);
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear previous image

    try {
      setMessage('Processing image: Removing background (this may take a moment)...');

      const backgroundRemovedBlob = await fetchImageAsFile(imageTempUrl.current!);


      setMessage('Processing image: Loading foreground...');
      const foregroundImage = new Image();
      const foregroundUrl = URL.createObjectURL(backgroundRemovedBlob!);

      await new Promise<void>((resolve, reject) => {
        foregroundImage.onload = () => {
          URL.revokeObjectURL(foregroundUrl); // Clean up after load
          resolve();
        };
        foregroundImage.onerror = () => {
          URL.revokeObjectURL(foregroundUrl);
          reject(new Error('Failed to load the background-removed image.'));
        };
        foregroundImage.src = foregroundUrl;
      });

      canvas.width = originalImageElement!.width;
      canvas.height = originalImageElement!.height;

      resultImageElement.current = foregroundImage;
      setMessage('Processing image: Detecting faces...');

      const originalCanvas = document.createElement('canvas');
      const originalCtx = originalCanvas.getContext('2d');
      originalCanvas.width = canvas.width;
      originalCanvas.height = canvas.height;
      originalCtx!.drawImage(originalImageElement!, 0, 0, canvas.width, canvas.height);
      const detections = await faceapi.detectAllFaces(originalCanvas).withFaceLandmarks();

      if (detections && detections.length > 0) {
        setMessage(`Found ${detections.length} face(s). Applying sensor bar...`);
        detections.forEach((detection: any) => {
          const landmarks = detection.landmarks;
          const leftEyePoints = landmarks.getLeftEye() as EyePoint[];
          const rightEyePoints = landmarks.getRightEye() as EyePoint[];

          const bc = document.createElement('canvas');
          barCanvas.current = bc
          const barCtx = barCanvas.current!.getContext('2d');
          barCanvas.current!.width = canvas.width;
          barCanvas.current!.height = canvas.height;






          drawCombinedSensorBar(barCtx!, leftEyePoints, rightEyePoints);


          drawResult(foregroundImage);

        });
        setMessage('Processing complete! 🎉');
      } else {
        drawResult(foregroundImage);
        setMessage('Background removed, red background applied. No faces detected for sensor bar. 🤔');
      }
      setDownloadUrl(canvas.toDataURL('image/png'));

    } catch (err: any) {
      console.error("Error during image processing:", err);
      let userMessage = 'An unexpected error occurred during processing.';
      if (err.message) {
        userMessage = err.message;
      }
      // Check for common Imgly asset loading errors (example, adjust based on actual errors)
      if (err.message && (err.message.includes('fetch') || err.message.includes('.wasm'))) {
        userMessage += " This might be due to issues loading @imgly/background-removal assets. Check console and network tab. You may need to configure 'publicPath'.";
      }
      setMessage(`Error: ${userMessage}`);
      setIsError(true);
    } finally {
      setIsLoading(false);
      setIsProcessing(false);
    }
  }

  const processAnime = async () => {

    if (!originalImageElement || !ortSession) {
      setErrorModal("Please upload an image and wait for the model to load before segmenting.");
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');

    if (!canvas || !ctx) {
      setMessage('Canvas not ready.');
      setIsError(true);
      return;
    }

    setMessage('Processing image: Initializing...');
    setIsProcessing(true);
    setIsLoading(true);
    setDownloadUrl(null);
    setIsError(false);


    try {
      setMessage('Processing image: Removing background (this may take a moment)...');

      const backgroundRemovedBlob = await fetchImageAsFile(imageTempUrl.current!);



      setMessage('Processing image: Loading foreground...');
      const foregroundImage = new Image();
      const foregroundUrl = URL.createObjectURL(backgroundRemovedBlob!);


      await new Promise<void>((resolve, reject) => {
        foregroundImage.onload = () => {
          URL.revokeObjectURL(foregroundUrl); // Clean up after load
          resolve();
        };
        foregroundImage.onerror = () => {
          URL.revokeObjectURL(foregroundUrl);
          reject(new Error('Failed to load the background-removed image.'));
        };
        foregroundImage.src = foregroundUrl;
      });

      canvas.width = originalImageElement.width;
      canvas.height = originalImageElement.height;

      resultImageElement.current = foregroundImage;
      setMessage('Processing image: Detecting faces...');

      const originalCanvas = document.createElement('canvas');
      const originalCtx = originalCanvas.getContext('2d');
      originalCanvas.width = canvas.width;
      originalCanvas.height = canvas.height;
      originalCtx!.drawImage(foregroundImage, 0, 0, canvas.width, canvas.height);

      const inputTensor = await preprocess(originalImageElement);
      const inputName = ortSession.inputNames[0];
      const outputNames = ortSession.outputNames;
      const feeds: any = { [inputName]: inputTensor };

      setMessage("Running model inference...");
      const results = await ortSession.run(feeds);

      const detectionOutput = results[outputNames[0]];
      const maskFeatures = results[outputNames[1]];

      if (!detectionOutput || !maskFeatures) {
        throw new Error("Model output is missing expected tensors.");
      }
      if (!(detectionOutput.data instanceof Float32Array) || !(maskFeatures.data instanceof Float32Array)) {
        throw new Error("Model output data is not in Float32Array format as expected.");
      }

      setMessage("Post-processing segmentation (incl. NMS)...");
      const { mask: _, count: detectedCountNum, boxes: boundingBoxes } = postprocessSegmentationWithNMS(
        detectionOutput.data as Float32Array,
        detectionOutput.dims,
        maskFeatures.data as Float32Array,
        maskFeatures.dims,
        originalImageShape
      );
      if (detectedCountNum > 1) {


        const leftEye = getBoundingBoxCenter({
          xMin: boundingBoxes[0].x, yMin: boundingBoxes[0].y, xMax: boundingBoxes[0].x + boundingBoxes[0].width, yMax: boundingBoxes[0].y + boundingBoxes[0].height
        });

        const rightEye = getBoundingBoxCenter({
          xMin: boundingBoxes[1].x, yMin: boundingBoxes[1].y, xMax: boundingBoxes[1].x + boundingBoxes[1].width, yMax: boundingBoxes[1].y + boundingBoxes[1].height
        });

        const midPointX = (leftEye.centerX + rightEye.centerX) / 2;
        const midPointY = (leftEye.centerY + rightEye.centerY) / 2;
        const deltaX = rightEye.centerX - leftEye.centerX;
        const deltaY = rightEye.centerY - leftEye.centerY;
        const angle = Math.atan2(deltaY, deltaX);
        const distanceBetweenEyeCenters = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        const averageEyeWidth = (boundingBoxes[0].width + boundingBoxes[1].width) / 2;
        const sensorWidth = distanceBetweenEyeCenters + averageEyeWidth * 1.5;
        const averageEyeHeight = (boundingBoxes[0].height + boundingBoxes[1].height) / 2;
        const sensorHeight = averageEyeHeight * 1.5;

        const bc = document.createElement('canvas');
        barCanvas.current = bc
        const barCtx = barCanvas.current!.getContext('2d');
        barCanvas.current!.width = canvas.width;
        barCanvas.current!.height = canvas.height;

        //ctx.save();
        barCtx!.translate(midPointX, midPointY);
        barCtx!.rotate(angle);
        barCtx!.fillStyle = 'black';
        barCtx!.fillRect(-sensorWidth / 2, -sensorHeight / 2, sensorWidth, sensorHeight);
        console.log(midPointX, midPointY, -sensorWidth / 2, -sensorHeight / 2, sensorWidth, sensorHeight);


        drawResult(foregroundImage);





        setMessage('Processing complete! 🎉');
      } else {
        drawResult(foregroundImage);
        setMessage('Background removed, red background applied. No faces detected for sensor bar. 🤔');
      }
      setDownloadUrl(canvas.toDataURL('image/png'));

    } catch (err: any) {
      console.error("Error during image processing:", err);
      let userMessage = 'An unexpected error occurred during processing.';
      if (err.message) {
        userMessage = err.message;
      }
      // Check for common Imgly asset loading errors (example, adjust based on actual errors)
      if (err.message && (err.message.includes('fetch') || err.message.includes('.wasm'))) {
        userMessage += " This might be due to issues loading @imgly/background-removal assets. Check console and network tab. You may need to configure 'publicPath'.";
      }
      setMessage(`Error: ${userMessage}`);
      setIsError(true);
    } finally {
      setIsLoading(false);
      setIsProcessing(false);
    }
  }

  // --- Handle Segmentation ---
  const handleProcess = async () => {

    if (contentType == "real-person") {
      processReal();
    } else if (contentType == "anime") {
      processAnime();
    }


  };

  // --- Error Modal Component ---
  const ErrorModal: React.FC<{ message: string; onClose: () => void }> = ({ message, onClose }) => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white p-6 rounded-lg shadow-xl max-w-sm w-full">
        <h3 className="text-lg font-semibold text-red-700 mb-4">Error</h3>
        <p className="text-gray-700 mb-4">{message}</p>
        <button
          onClick={onClose}
          className="w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded-md transition duration-150 ease-in-out"
        >
          Close
        </button>
      </div>
    </div>
  );
  return (
    <div className='bg-gray-100 min-h-screen flex flex-col items-center justify-center p-4'>
      {errorModal && <ErrorModal message={errorModal} onClose={() => setErrorModal(null)} />}
      <style>{keyframes}</style>
      <div className="bg-white p-6 sm:p-8 rounded-xl shadow-2xl w-full max-w-7xl">
        <header className="text-center mb-4">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-800">Generate Your PP</h1>
          <p className="text-gray-600 mt-2">small pp big pp doesnt matter.</p>
        </header>


        <div className='grid grid-cols-1 md:grid-cols-2 gap-4 w-full'>
          <div>
            <div className="mb-4">
              <label htmlFor="imageUpload" className="block mb-2 text-sm font-medium text-gray-700">Upload Image:</label>
              <input
                type="file"
                id="imageUpload"
                accept="image/*"
                onChange={handleImageUpload}
                className="block w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 cursor-pointer focus:outline-none focus:border-blue-500 p-2.5"
                disabled={isFaceApiModelsLoading || isLoading}
              />

            </div>
            <div className='mb-4'>
              <label className="block mb-2 text-sm font-medium text-gray-700">Content Type:</label>
              <ul className="items-center w-full text-sm font-medium text-gray-900 bg-white border border-gray-200 rounded-lg sm:flex ">
                <li className="w-full border-b border-gray-200 sm:border-b-0 sm:border-r ">
                  <div className="flex items-center ps-3">
                    <input id="horizontal-list-radio-license" type="radio" checked={contentType == "real-person"} onChange={(_: any) => {
                      setContentType("real-person");
                    }} disabled={isLoading} name="list-radio" className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500 " />
                    <label htmlFor="horizontal-list-radio-license" className="w-full py-3 ms-2 text-sm font-medium text-gray-900 ">Real Person </label>
                  </div>
                </li>
                <li className="w-full border-b border-gray-200 sm:border-b-0 sm:border-r ">
                  <div className="flex items-center ps-3">
                    <input id="horizontal-list-radio-id" type="radio" checked={contentType == "anime"} onChange={(_: any) => {
                      setContentType("anime");
                    }} disabled={isLoading} name="list-radio" className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 focus:ring-blue-500 " />
                    <label htmlFor="horizontal-list-radio-id" className="w-full py-3 ms-2 text-sm font-medium text-gray-900 ">Anime</label>
                  </div>
                </li>


              </ul>
            </div>
            <div className='mb-4'>
              <label className="block mb-2 text-sm font-medium text-gray-700">Setting:</label>
              <ul className="items-center w-full text-sm font-medium text-gray-900 bg-white border border-gray-200 rounded-lg sm:flex ">

                <li className="w-full border-b border-gray-200 sm:border-b-0 sm:border-r ">
                  <div className="flex items-center ps-3">
                    <input id="glow-effect" type="checkbox" checked={glowEffect} onChange={(_) => {
                      setGlowEffect(!glowEffect);
                    }} disabled={isLoading} className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded-sm focus:ring-blue-500 " />
                    <label htmlFor="glow-effect" className="w-full py-3 ms-2 text-sm font-medium text-gray-900 ">Glow</label>
                  </div>
                </li>
                <li className="w-full border-b border-gray-200 sm:border-b-0 sm:border-r ">
                  <div className="flex items-center ps-3">
                    <input id="bg-color" type="color" value={bgColor} onChange={(e) => {
                      setBgColor(e.target.value);

                    }} disabled={isLoading} className="w-8 h-8 text-blue-600 bg-gray-100 border-gray-300 rounded-sm focus:ring-blue-500 " />
                    <label htmlFor="bg-color" className="w-full py-3 ms-2 text-sm font-medium text-gray-900 ">Color</label>
                  </div>
                </li>

              </ul>
            </div>

            <div className='mb-4 text-center'>
              <button
                onClick={handleProcess} disabled={!uploadFile || isProcessing || !ortSession} className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer">
                {isProcessing && status.startsWith("Processing") ? '🔃 Processing...' : 'Process Image 🔥'}
              </button>
            </div>

            <div id="statusContainer" className="text-center my-4">

              <p
                id="messageArea"
                className={`font-medium min-h-[30px] ${isError ? 'text-red-500' : 'text-gray-700'}`}
              >
                {message}
              </p>
            </div>



          </div>
          <div>
            <div className="w-full bg-gray-200 min-h-96 rounded-lg overflow-hidden shadow-inner relative">
              <canvas ref={canvasRef} id="canvas" className="w-full h-auto hidden rounded-lg"></canvas>
              {originalUrl && <BeforeAfterSlider
                beforeImage={originalUrl!}
                afterImage={downloadUrl ?? originalUrl}
                beforeLabel="Original"
                afterLabel="New"
                className="w-full mx-auto rounded-lg "
              />}
              {isLoading && <div style={loaderStyle} data-testid="loader" className='absolute top-0 left-0 bottom-0 right-0 m-auto'></div>}
            </div>
            <div className="mt-6 text-center">
              {downloadUrl && !isLoading && (
                <a
                  id="downloadLink"
                  href={downloadUrl}
                  className="bg-blue-500 hover:bg-blue-700 block w-full text-white font-bold py-3 px-6 rounded-lg transition duration-150 ease-in-out"
                  download="sensored-image-pp.png"
                >
                  Download Image 🖼️
                </a>
              )}
            </div>
          </div>
        </div>


      </div>

      <footer className="text-center text-gray-500 mt-8 text-sm">
        <p>Powered by tmpfiles.org</p>
      </footer>
    </div>
  )
}

export default App
