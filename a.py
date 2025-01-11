from fast_alpr import ALPR
import time

# You can also initialize the ALPR with custom plate detection and OCR models.
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

# The "assets/test_image.png" can be found in repo root dit
# You can also pass a NumPy array containing cropped plate image
start_time = time.time()
alpr_results = alpr.predict("./License-Plate-Recognition/test_image/101.jpg")
end_time = time.time()
print(alpr_results)
print("Ex time: ", end_time-start_time)
