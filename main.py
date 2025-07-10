import random
import cv2
from pathlib import Path
import paths
import routine
import time
import concurrent.futures
import glob
import functions_th as th
# List of all valid case names from paths.py
case_names = [
     "green_buco_in_piu", "green_buco_in_meno", "green_lettere_disallineate",
    "green_ok"
]

project_root = Path(__file__).resolve().parent
average_time = 1
iterations = 0
routine_obj = routine.Routine()

def run_case(case):
    scorre_path, base_shape_path, base_print_path, recomposed_path = paths.define_files(case, project_root)
    recomposed = cv2.imread(recomposed_path, cv2.IMREAD_COLOR)

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
            # Load images from the specified folder path
    
    image_files = sorted(
        glob.glob(scorre_path),
        key=th.alphanum_key)
    frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in image_files]
    frames = [f for f in frames if f is not None and f.shape == frames[0].shape]
    base_shape = cv2.imread(base_shape_path, cv2.IMREAD_GRAYSCALE)
    base_print = cv2.imread(base_print_path, cv2.IMREAD_GRAYSCALE)

    start = time.time()
    result = routine_obj.run_full_analysis(
        frames=frames,
        base_shape=base_shape,
        base_print=base_print,
        base_image=base_image,
        show_plots=False
    )
    end = time.time()
    # if any(result):
    #     print(f"Defect detected in case: {case}")
    #     # Optionally rerun with plots
    #     routine_obj.run_full_analysis(
    #         frames=frames,
    #         base_shape=base_shape,
    #         base_print=base_print,
    #         base_image=base_image,
    #         show_plots=True
    #                     )
    return case, result, end - start

def main():
    global average_time, iterations
    max_cases = len(case_names)
    n_repeats = 2  # Number of times to process all cases
    all_times = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for repeat in range(n_repeats):
            futures = [executor.submit(run_case, case) for case in case_names]
            for future in concurrent.futures.as_completed(futures):
                case, result, elapsed = future.result()
                if case != "green_lettere_disallineate":
                    all_times.append(elapsed)
                    average_time = sum(all_times) / len(all_times)
                    iterations += 1
                print(f"Defect detected: {result}")
                print(f"Case: {case}, Time: {elapsed:.2f}s, Avg Time: {average_time:.2f}s")
    print(f"\nAverage execution time over {iterations} cases: {average_time:.2f} seconds.")

if __name__ == "__main__":
    main()