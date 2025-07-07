import random
import cv2
from pathlib import Path
import paths
import routine
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


#"parmareggio_bucato","nappies_ok","green_scratched"
# List of all valid case names from paths.py
case_names = [
    "parmareggio", "parmigiano","parmareggio_ok", "nappies",
     "green_buco_in_piu", "green_buco_in_meno", "green_lettere_disallineate",
    "green_ok",
]
expected_results = {
    "parmareggio": (False, False, False),
    #"parmareggio_bucato": (False, True, False),
    "parmigiano": (False, False, False),
    "parmareggio_ok": (False, False, False),
    "nappies": (False, True, False),
    #"green_scratched": (True, True, False),
    "green_buco_in_piu": (True, True, False),
    "green_buco_in_meno": (True, True, False),
    "green_lettere_disallineate": (False, False, True),
    "green_ok": (False, False, False),
}

project_root = Path(__file__).resolve().parent

def process_case(case):
    print(f"Processing case: {case}")
    scorre_path, base_shape_path, base_print_path, recomposed_path = paths.define_files(case, project_root)
    recomposed = cv2.imread(recomposed_path, cv2.IMREAD_COLOR)
    base_shape = cv2.imread(base_shape_path, cv2.IMREAD_GRAYSCALE)
    base_print = cv2.imread(base_print_path, cv2.IMREAD_GRAYSCALE)
    if "green" in case:
        base_image = cv2.imread(paths.define_files("green_ok", project_root)[3], cv2.IMREAD_COLOR)
    elif "parmareggio" in case or "parmigiano" in case:
        base_image = cv2.imread(paths.define_files("parmareggio_ok", project_root)[3], cv2.IMREAD_COLOR)
    elif "nappies" in case:
        base_image = cv2.imread(paths.define_files("nappies_ok", project_root)[3], cv2.IMREAD_COLOR)
    else:
        base_image = recomposed  # fallback
        print(f"Warning: No specific 'ok' image found for case '{case}', using recomposed image.")
    start = time.time()
    result = routine.run_full_analysis(
        recomposed=recomposed,
        torecompose_path=scorre_path, 
        base_shape=base_shape,
        base_print=base_print,
        base_image=base_image,
        show_plots=False
    )
    end = time.time()
    return case, result, end - start

if __name__ == "__main__":
    num_threads = 4  # Set this to the number of threads you want
    iterations = 0
    average_time = 0
    all_results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_case, random.choice(case_names)) for _ in range(20)]  # 20 random cases
        for future in as_completed(futures):
            case, result, elapsed = future.result()
            iterations += 1
            average_time = (average_time * (iterations - 1) + elapsed) / iterations
            expected= expected_results.get(case)
            iscorrect= "Right" if result == expected else "Wrong"
            print(f"{iscorrect},Case: {case}, Expected:{expected_results.get(case)}, Defect detected: {result}, Time: {elapsed:.2f}s, Avg time: {average_time:.2f}s", flush=True)
            all_results.append((case, result, elapsed))
    print(f"Final average execution time: {average_time:.2f}s over {iterations} cases", flush=True)