from scripts.openmp import grayscaleOpenMP, sepiaOpenMP, blurOpenMP, contrastOpenMP, edge_detectionOpenMP
from scripts.MPI import grayscaleMPI, sepiaMPI, blurMPI, contrastMPI, edge_detectionMPI
from scripts.cudagpu import grayscaleCUDAGPU, sepiaCUDAGPU, blurCUDAGPU, contrastCUDAGPU, edge_detectionCUDAGPU
from scripts.cuda import grayscaleCUDA, sepiaCUDA, blurCUDA, contrastCUDA, edge_detectionCUDA
from mpi4py import MPI
from mpi4py import MPI

from modules import script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules.ui_components import ResizeHandleRow
from modules import shared
from concurrent.futures import ThreadPoolExecutor
from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy
from torch.multiprocessing import Pool, Process, set_start_method


import numpy as np
import modules.scripts as scripts
import gradio as gr
import os
import pathlib
import torch
import torch.multiprocessing as mp
import time
import matplotlib.pyplot as plt
import seaborn as sns
import threading



#Inicjalizacja MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def measure_execution_timeOpenMP(operation, input_image, amount_of_threads):
    execution_times = []
    processed_images = []

    # Function to execute the operation and measure execution time for each thread
    def execute_operation_threadOpenMP(thread_id):
        start_time = time.time()
        result = operation(input_image)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times[thread_id] = execution_time
        processed_images[thread_id] = result  # Store processed image for this thread

    # Create lists to store execution times and processed images for each thread
    execution_times = [0.0] * amount_of_threads
    processed_images = [None] * amount_of_threads

    # Create and start threads for each parallel execution
    threads = []
    for i in range(amount_of_threads):
        thread = threading.Thread(target=execute_operation_threadOpenMP, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Assuming output is a single image, return it along with execution times
    return processed_images[0], execution_times

def measure_execution_timeMPI(operation, input_image, amount_of_threads):
    execution_times = []
    processed_images = []

    # Divide the image into chunks for each thread
    chunk_height = input_image.shape[0] // amount_of_threads
    chunks = [input_image[i * chunk_height : (i + 1) * chunk_height] for i in range(amount_of_threads)]

    # Function to execute the operation and measure execution time for each thread
    def execute_operation_threadMPI(thread_id):
        start_time = time.time()
        result = operation(chunks[thread_id])
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times[thread_id] = execution_time
        processed_images[thread_id] = result  # Store processed image for this thread

    # Create lists to store execution times and processed images for each thread
    execution_times = [0.0] * amount_of_threads
    processed_images = [None] * amount_of_threads

    # Create and start threads for each parallel execution
    threads = []
    for i in range(amount_of_threads):
        thread = threading.Thread(target=execute_operation_threadMPI, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Join processed chunks back into a single image
    processed_image = np.vstack(processed_images)

    # Assuming output is a single image, return it along with execution times
    return processed_image, execution_times

def measure_execution_timeCUDAGPU(operation, input_image):
    execution_times = []

    # Function to execute the operation and measure execution time
    def execute_operation():
        start_time = time.time()
        result = operation(input_image)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        return result

    # Execute the operation on the GPU
    with torch.no_grad():
        torch.cuda.empty_cache()  # Clear GPU memory
        torch.cuda.synchronize()  # Synchronize before timing
        processed_image = execute_operation()
        torch.cuda.synchronize()  # Synchronize after timing

    # Return processed image and execution times
    return processed_image, execution_times

def measure_execution_timeCUDA(operation, input_image, amount_of_threads):
    execution_times = []

    # Function to execute the operation and measure execution time for each thread
    def execute_operation_thread(thread_id):
        start_time = time.time()
        with torch.no_grad():
            device = torch.device('cpu')  # Force CPU usage
            input_image_tensor = torch.tensor(input_image, device=device)
            input_image_tensor = input_image_tensor.float()
            input_image_np = input_image_tensor.numpy()  # Convert tensor to numpy array
            result = operation(input_image_np)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times[thread_id] = execution_time

    # Create lists to store execution times for each thread
    execution_times = [0.0] * amount_of_threads

    # Create and start threads for each parallel execution
    threads = []
    for i in range(amount_of_threads):
        thread = threading.Thread(target=execute_operation_thread, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Execute the operation on the CPU
    with torch.no_grad():
        device = torch.device('cpu')
        input_image_tensor = torch.tensor(input_image, device=device)
        processed_image = operation(input_image_tensor)
        torch.cuda.empty_cache()

    # Return processed image and execution times
    return processed_image, execution_times

@torch.inference_mode()
@torch.no_grad()
def predict(input_image, used_technology, type_of_operation, amount_of_threads):
    operation = None

    # Determine which operation to perform based on the type_of_operation input
    if type_of_operation == 'Czarno-bia\u0142y':
        if used_technology == 'OpenMP':
            operation = grayscaleOpenMP
        elif used_technology == 'MPI':
            operation = grayscaleMPI
        elif used_technology == 'CUDA':
            operation = grayscaleCUDA
        elif used_technology == 'CUDA (GPU)':
            operation = grayscaleCUDAGPU
    elif type_of_operation == 'Sepia':
        if used_technology == 'OpenMP':
            operation = sepiaOpenMP
        elif used_technology == 'MPI':
            operation = sepiaMPI
        elif used_technology == 'CUDA':
            operation = sepiaCUDA
        elif used_technology == 'CUDA (GPU)':
            operation = sepiaCUDAGPU
    elif type_of_operation == 'Rozmycie':
        if used_technology == 'OpenMP':
            operation = blurOpenMP
        elif used_technology == 'MPI':
            operation = blurMPI
        elif used_technology == 'CUDA':
            operation = blurCUDA
        elif used_technology == 'CUDA (GPU)':
            operation = blurCUDAGPU
    elif type_of_operation == 'Kontrast':
        if used_technology == 'OpenMP':
            operation = contrastOpenMP
        elif used_technology == 'MPI':
            operation = contrastMPI
        elif used_technology == 'CUDA':
            operation = contrastCUDA
        elif used_technology == 'CUDA (GPU)':
            operation = contrastCUDAGPU
    elif type_of_operation == 'Wykrywanie kraw\u0119dzi':
        if used_technology == 'OpenMP':
            operation = edge_detectionOpenMP
        elif used_technology == 'MPI':
            operation = edge_detectionMPI
        elif used_technology == 'CUDA':
            operation = edge_detectionCUDA
        elif used_technology == 'CUDA (GPU)':
            operation = edge_detectionCUDAGPU
    else:
        return "Error: Invalid operation"

    if used_technology == 'OpenMP':
        print("Using OpenMP with", amount_of_threads, "threads")
        processed_image, execution_times = measure_execution_timeOpenMP(operation, input_image, amount_of_threads)
    elif used_technology == 'MPI':
        os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"] = str(amount_of_threads)
        print("Using MPI with", amount_of_threads, "threads")
        processed_image, execution_times = measure_execution_timeMPI(operation, input_image, amount_of_threads)
    elif used_technology == 'CUDA':
        print("Using CUDA (CPU) with", amount_of_threads, "threads")
        processed_image, execution_times = measure_execution_timeCUDA(operation, input_image, amount_of_threads)
    elif used_technology == 'CUDA (GPU)':
        print("Using CUDA (GPU) with", amount_of_threads, "threads")
        processed_image, execution_times = measure_execution_timeCUDAGPU(operation, input_image)
    
    return processed_image, execution_times

def max_threads():
    try:
        num_cores = os.cpu_count()
        max_threads = num_cores if num_cores else None
        return max_threads
    except NotImplementedError:
        return None
    

def sprawdz(used_technology, amount_of_threads):
    if used_technology == 'CUDA (GPU)':
        amount_of_threads = 1
    return amount_of_threads

def on_ui_tabs():
    with gr.Blocks() as ui_component:
        with ResizeHandleRow():
            with gr.Column():
                input_image = gr.Image(label='Input Image', source='upload', type='numpy', height=400)
                used_technology = gr.Radio(label='Wykorzystanie biblioteki',
                                              choices=['OpenMP', 'MPI', 'CUDA', 'CUDA (GPU)'], value='OpenMP', Interactive=True)
                
                type_of_operation = gr.Radio(label='Rodzaj operacji',
                                              choices=['Czarno-bia\u0142y', 'Sepia', 'Rozmycie', 'Kontrast', 'Wykrywanie kraw\u0119dzi'], value='Czarno-bia\u0142y')
                amount_of_threads = gr.Slider(label='Liczba w\u0105tk\u00f3w (Dla CUDA (GPU) zostaw 1)', minimum=1, maximum=max_threads(), step=1, value=1, Interactive=True)
                generate_button = gr.Button(value="Generate")
                ctrls = [input_image, used_technology, type_of_operation, amount_of_threads]
            with gr.Column():
                output_image = gr.Image(label='Wynik modyfikacji', type='numpy', source='upload', height=400, Interactive=True)
                output_wykresy = gr.Gallery(label='Wykresy', show_label=True, object_fit='contain',
                                            visible=True, columns=4)

            used_technology.change(sprawdz, [used_technology, amount_of_threads], amount_of_threads)

            generate_button.click(generate_callback, inputs=ctrls, outputs=[output_image, output_wykresy])

    return [(ui_component, "PRiR", "PRiR")]
    
def generate_callback(input_image, used_technology, type_of_operation, amount_of_threads):
    processed_image, execution_times = predict(input_image, used_technology, type_of_operation, amount_of_threads)

    # Wybierz paletę kolorów
    palette = sns.color_palette("husl", amount_of_threads)

    plt.figure(figsize=(8, 6), facecolor='#0b0f19')
    plt.gca().set_facecolor('#0b0f19')

    # Obliczenie maksymalnej wartości czasu wykonania
    max_execution_time = max(execution_times)
    max_execution_time_ms = max_execution_time * 1000


    # Tworzenie słupków dla każdego wątku z użyciem wybranej palety kolorów
    for i in range(amount_of_threads):
        plt.bar(i + 1, execution_times[i], color=palette[i])
        

    plt.xlabel('Liczba wątków', color='white')
    plt.ylabel('Czas wykonania (s)', color='white')
    plt.title('Wykres wydajnościowy', color='white')
    plt.grid(False)

    # Ustawienie skali osi Y tak, aby kończyła się w połowie maksymalnej wartości czasu wykonania
    plt.ylim(0, max_execution_time * 2)
    plt.yticks(color='white')

    # Ustawienie etykiet osi X dla każdego wątku
    plt.xticks(range(1, amount_of_threads + 1), color='white')

    #Czas wykonania
    plt.text((amount_of_threads + 1) / 2, max_execution_time * 2 * 0.9, f"Całkowity czas wykonania: {max_execution_time_ms:.2f} ms", fontsize=10, bbox=dict(facecolor='#0b0f19', alpha=0.5), color='white', ha='center')


    plt.tight_layout()

    # Saving the plot 
    wykres = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "wykresy", "wykres.png")
    plt.savefig(wykres)
    plt.close()

    # Returning the result and the plot
    return processed_image, [wykres]
    
script_callbacks.on_ui_tabs(on_ui_tabs)