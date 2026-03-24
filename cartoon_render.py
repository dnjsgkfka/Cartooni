import cv2
import numpy as np


def nothing(x):
    pass


def apply_cartoon(img: np.ndarray, color_sigma: int, edge_block: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        edge_block, 3
    )

    color = img.copy()
    sigma = max(1, color_sigma)
    for _ in range(4):
        color = cv2.bilateralFilter(color, d=7, sigmaColor=sigma, sigmaSpace=sigma)

    edge_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edge_3ch)
    cartoon = cv2.convertScaleAbs(cartoon, alpha=1.05, beta=10)
    return cartoon


def resize_for_display(merge: np.ndarray, max_w: int = 1400) -> np.ndarray:
    h, w = merge.shape[:2]
    if w > max_w:
        scale = max_w / w
        merge = cv2.resize(merge, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    return merge


# ── 초기화 ────────────────────────────────────────────────────────────────
img = cv2.imread('002.jpg')
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

h, w = img.shape[:2]
if w > 1000:
    img = cv2.resize(img, (1000, int(h * 1000 / w)), interpolation=cv2.INTER_AREA)

window_name = 'Cartooni : Original | Result'
cv2.namedWindow(window_name)
cv2.createTrackbar('Color Flattening', window_name, 50, 150, nothing)
cv2.createTrackbar('Edge Detail',      window_name, 7,  31,  nothing)

prev_sigma, prev_block = -1, -1
merge = resize_for_display(np.hstack((img, img)))  # 초기 화면도 적용

while True:
    sigma = cv2.getTrackbarPos('Color Flattening', window_name)
    block = cv2.getTrackbarPos('Edge Detail',      window_name)

    if block % 2 == 0:
        block += 1
    block = max(3, block)

    if sigma != prev_sigma or block != prev_block:
        result = apply_cartoon(img, sigma, block)
        merge  = resize_for_display(np.hstack((img, result)))
        prev_sigma, prev_block = sigma, block

    cv2.imshow(window_name, merge)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()