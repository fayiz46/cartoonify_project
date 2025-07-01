from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import os
import shutil

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup template directory
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/cartoonify/")
async def cartoonify(request: Request, file: UploadFile = File(...)):
    filename = file.filename
    upload_path = f"static/uploads/{filename}"
    cartoon_path = f"static/uploads/cartoon_{filename}"

    # Save the uploaded image
    os.makedirs("static/uploads", exist_ok=True)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read and cartoonify
    img = cv2.imread(upload_path)

    # Cartoonify logic:
    def edge_mask(img, line_size=7, blur_value=7):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, blur_value)
        edges = cv2.adaptiveThreshold(gray_blur, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY,
                                      blockSize=line_size,
                                      C=blur_value)
        return edges

    def color_quantization(img, k=9):
        data = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        return result.reshape(img.shape)

    def enhance_image(img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        s = cv2.multiply(s, 1.3)
        v = cv2.multiply(v, 1.1)
        img_hsv = cv2.merge([h, np.clip(s, 0, 255), np.clip(v, 0, 255)])
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def apply_gabor_filter(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        accum = np.zeros_like(gray, dtype=np.float32)
        for theta in range(4):
            angle = theta * np.pi / 4
            kernel = cv2.getGaborKernel((11, 11), 4.0, angle, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            np.maximum(accum, filtered, accum)
        accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX)
        gabor_texture = np.uint8(accum)
        return gabor_texture

    def add_shadow(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shadow = cv2.Laplacian(gray, cv2.CV_64F)
        shadow = cv2.convertScaleAbs(shadow)
        shadow = cv2.GaussianBlur(shadow, (9, 9), 0)
        shadow = cv2.normalize(shadow, None, 0, 80, cv2.NORM_MINMAX)
        shadow_bgr = cv2.merge([shadow]*3)
        return cv2.subtract(img, shadow_bgr)

    edges = edge_mask(img)
    quantized = color_quantization(img)
    smoothed = cv2.bilateralFilter(quantized, d=5, sigmaColor=300, sigmaSpace=300)
    enhanced = enhance_image(smoothed)
    gabor_texture = apply_gabor_filter(enhanced)
    gabor_colored = cv2.applyColorMap(gabor_texture, cv2.COLORMAP_BONE)
    enhanced = cv2.addWeighted(enhanced, 0.9, gabor_colored, 0.1, 0)
    shadowed = add_shadow(enhanced)
    cartoon = cv2.bitwise_and(shadowed, shadowed, mask=edges)

    # Save cartoonified image
    cv2.imwrite(cartoon_path, cartoon)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "original_img": f"/{upload_path}",
        "result_img": f"/{cartoon_path}"
    })
