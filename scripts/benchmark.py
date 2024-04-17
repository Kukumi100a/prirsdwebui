from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2


def run_benchmark(image):
    from .main import generate_callback, measure_execution_timeOpenMP, measure_execution_timeMPI, measure_execution_timeCUDAGPU, measure_execution_timeCUDA
    from .openmp import grayscaleOpenMP, sepiaOpenMP, blurOpenMP, contrastOpenMP, edge_detectionOpenMP
    from .MPI import grayscaleMPI, sepiaMPI, blurMPI, contrastMPI, edge_detectionMPI
    from .cudagpu import grayscaleCUDAGPU, sepiaCUDAGPU, blurCUDAGPU, contrastCUDAGPU, edge_detectionCUDAGPU
    from .cuda import grayscaleCUDA, sepiaCUDA, blurCUDA, contrastCUDA, edge_detectionCUDA

    operations = {
        'Wykrywanie kraw\u0119dzi': [edge_detectionOpenMP, edge_detectionMPI, edge_detectionCUDA, edge_detectionCUDAGPU],
        'Czarno-bia\u0142y': [grayscaleOpenMP, grayscaleMPI, grayscaleCUDA, grayscaleCUDAGPU],
        'Sepia': [sepiaOpenMP, sepiaMPI, sepiaCUDA, sepiaCUDAGPU],
        'Rozmycie': [blurOpenMP, blurMPI, blurCUDA, blurCUDAGPU],
        'Kontrast': [contrastOpenMP, contrastMPI, contrastCUDA, contrastCUDAGPU],
    }

    technologies = ['OpenMP', 'MPI', 'CUDA', 'CUDA (GPU)']
    
    num_trials = 3
    num_threads_list = [1, 4, 8, 16, 20, 28]

    execution_times = {operation: {tech: {num_threads: [] for num_threads in num_threads_list} for tech in technologies} for operation in operations}

    processed_images = []
    wykresy_files = []  # Lista ścieżek do plików z wykresami
    
    # Tworzenie foldera dla obrazów
    obrazy_folder = os.path.join(os.getcwd(), "extensions", "prirsdwebui", "obrazy")
    os.makedirs(obrazy_folder, exist_ok=True)

    for operation, ops in operations.items():
        for tech, op_func in zip(technologies, ops):
            for num_threads in num_threads_list:
                if tech == 'CUDA (GPU)' and num_threads > 1:
                    continue
                for _ in range(num_trials):
                    start_time = time.time()
                    
                    # Wywołanie odpowiedniej funkcji do mierzenia czasu i przetwarzania
                    if tech == 'OpenMP':
                        processed_image, execution_time = measure_execution_timeOpenMP(op_func, image, num_threads)
                    elif tech == 'MPI':
                        processed_image, execution_time = measure_execution_timeMPI(op_func, image, num_threads)
                    elif tech == 'CUDA (GPU)':
                        processed_image, execution_time = measure_execution_timeCUDAGPU(op_func, image)
                    elif tech == 'CUDA':
                        processed_image, execution_time = measure_execution_timeCUDA(op_func, image, num_threads)
                    
                    end_time = time.time()

                    time.sleep(1)
                    # Zapisywanie przetworzonego obrazu
                    timestamp = time.strftime('%Y%m%d%H%M%S')
                    image_name = f"obraz_{tech.replace(' ', '_')}_{operation.replace(' ', '_')}_{num_threads}_{timestamp}.png"
                    image_path = os.path.join(obrazy_folder, image_name)

                    if 'Wykrywanie krawędzi' in operation and tech in ['CUDA', 'CUDA (GPU)']:
                        # Znormalizuj obraz do zakresu [0, 255]
                        normalized_image = (processed_image * 255).astype(np.uint8)
                        cv2.imwrite(image_path, normalized_image)
                    else:
                        Image.fromarray(processed_image).save(image_path)

                    processed_images.append(image_path)
                    execution_times[operation][tech][num_threads].append(end_time - start_time)

    # Obliczanie średnich czasów wykonania
    avg_execution_times = {operation: {tech: {num_threads: np.mean(times) for num_threads, times in exec_times.items()} for tech, exec_times in op_times.items()} for operation, op_times in execution_times.items()}

    # Generowanie wykresów
    plt.figure(figsize=(10, 6))
    for operation, op_times in avg_execution_times.items():
        for tech, exec_times in op_times.items():
            plt.plot(list(exec_times.keys()), list(exec_times.values()), label=f"{operation} - {tech}")
    plt.xlabel('Liczba wątków')
    plt.ylabel('Średni czas wykonania (s)')
    plt.title('Porównanie średniego czasu wykonania operacji dla różnych technologii')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    wykresy_folder = os.path.join(os.getcwd(), "extensions", "prirsdwebui", "wykresy")
    os.makedirs(wykresy_folder, exist_ok=True)
    wykres_file1 = os.path.join(wykresy_folder, "porownanie_czasow_wykonania.png")
    plt.savefig(wykres_file1, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    for operation, op_times in avg_execution_times.items():
        for tech, exec_times in op_times.items():
            reference_time = exec_times[1]
            differences = [time - reference_time for time in exec_times.values()]
            plt.plot(list(exec_times.keys()), differences, label=f"{operation} - {tech}")
    plt.xlabel('Liczba wątków')
    plt.ylabel('Różnica czasu wykonania (s)')
    plt.title('Porównanie różnic czasu wykonania względem jednego wątku dla różnych technologii')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    wykres_file2 = os.path.join(wykresy_folder, "porownanie_roznic_czasow_wykonania.png")
    plt.savefig(wykres_file2, bbox_inches='tight')
    plt.close()

    return processed_images, wykresy_files