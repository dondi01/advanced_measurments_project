import random
import cv2
from pathlib import Path
import paths
import routine
import time
# List of all valid case names from paths.py
case_names = [
    "parmareggio", "parmigiano", "parmareggio_bucato", "parmareggio_ok", "nappies",
    "green_scratched", "green_buco_in_piu", "green_buco_in_meno", "green_lettere_disallineate",
    "green_ok", "nappies_ok",
]

project_root = Path(__file__).resolve().parent

while True:
    case = random.choice(case_names)
    print(f"Testing case: {case}")
    scorre_path, base_shape_path, base_print_path, recomposed_path = paths.define_files(case, project_root)
    # Load images
    recomposed = cv2.imread(recomposed_path, cv2.IMREAD_COLOR)
    base_shape = cv2.imread(base_shape_path, cv2.IMREAD_GRAYSCALE)
    base_print = cv2.imread(base_print_path, cv2.IMREAD_GRAYSCALE)
    # For ORB, use the "ok" version of the base image if available, else use base_shape
    if "green" in case:
        base_image = cv2.imread(paths.define_files("green_ok", project_root)[3], cv2.IMREAD_COLOR)
    elif "parmareggio" in case or "parmigiano" in case:
        base_image = cv2.imread(paths.define_files("parmareggio_ok", project_root)[3], cv2.IMREAD_COLOR)
    elif "nappies" in case:
        base_image = cv2.imread(paths.define_files("nappies_ok", project_root)[3], cv2.IMREAD_COLOR)
    else:
        base_image = recomposed  # fallback
        print(f"Warning: No specific 'ok' image found for case '{case}', using recomposed image.")
    start=time.time()
    result = routine.run_full_analysis(
        recomposed=recomposed,
        base_shape=base_shape,
        base_print=base_print,
        base_image=base_image,
        show_plots=False
    )
    end=time.time()
    average_time = (average_time*iterations+end-start)/(iterations+1)
    iterations += 1
    print(f"Defect detected: {result}")
    if any(result):
        print(f"Defect detected in case: {case}")
        result = routine.run_full_analysis(
        recomposed=recomposed,
        base_shape=base_shape,
        base_print=base_print,
        base_image=base_image,
        show_plots=True
        )