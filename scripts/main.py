from scripts.openmp import grayscale, sepia, blur, contrast, edge_detection, process_image

import numpy as np
import modules.scripts as scripts
import gradio as gr
import os
import pathlib
import torch
import time



from modules import script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules.ui_components import ResizeHandleRow
from modules import shared
from concurrent.futures import ThreadPoolExecutor


from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy

@torch.inference_mode()
@torch.no_grad()
def predict(input_image, used_technology, type_of_operation, amount_of_threads):
    print("Input Image Shape:", input_image.shape)  # Debugging statement
    operation = None

    if type_of_operation == 'Czarno-bia\u0142y':
        operation = grayscale
    elif type_of_operation == 'Sepia':
        operation = sepia
    elif type_of_operation == 'Rozmycie':
        operation = blur
    elif type_of_operation == 'Kontrast':
        operation = lambda img: contrast(img, alpha=1.5, beta=0)
    elif type_of_operation == 'Wykrywanie kraw\u0119dzi':
        operation = edge_detection

    if used_technology == 'OpenMP':
        print("Using OpenMP with", amount_of_threads, "threads")  # Debugging statement
        with ThreadPoolExecutor(max_workers=amount_of_threads) as executor:
            start_time = time.time()
            processed_image = executor.map(operation, [input_image]*amount_of_threads)
            processed_image = list(processed_image)[0]  # Assuming output is single image
            end_time = time.time()
    elif used_technology == 'MPI' or used_technology == 'CUDA':
        return "B\u0142\u0105d: Operacja niezaimplementowana"
    else:
        print("Using single thread")  # Debugging statement
        start_time = time.time()
        processed_image = operation(input_image)
        end_time = time.time()

    print("Processed Image Shape:", processed_image.shape)  # Debugging statement
    print("Czas wykonania generacji:", end_time - start_time, "sekund")
    return processed_image

def max_threads():
    try:
        num_cores = os.cpu_count()
        max_threads = 2 * num_cores if num_cores else None
        return max_threads
    except NotImplementedError:
        return None

def on_ui_tabs():
    with gr.Blocks() as ui_component:
        with ResizeHandleRow():
            with gr.Column():
                input_image = gr.Image(label='Input Image', source='upload', type='numpy', height=400)
                used_technology = gr.Radio(label='Wykorzystanie biblioteki',
                                              choices=['OpenMP', 'MPI', 'CUDA'], value='CUDA')
                
                type_of_operation = gr.Radio(label='Rodzaj operacji',
                                              choices=['Czarno-bia\u0142y', 'Sepia', 'Rozmycie', 'Kontrast', 'Wykrywanie kraw\u0119dzi'], value='Czarno-bia\u0142y')
                amount_of_threads = gr.Slider(label='Liczba w\u0105tk\u00f3w', minimum=1, maximum=max_threads(), step=1, value=1)
                generate_button = gr.Button(value="Generate")
                ctrls = [input_image, used_technology, type_of_operation, amount_of_threads]
            with gr.Column():
                output_image = gr.Image(label='Wynik modyfikacji', type='numpy', source='upload', height=400, Interactive=True)
                output_wykresy = gr.Gallery(label='Wykresy', show_label=True, object_fit='contain',
                                            visible=True, columns=4)
        generate_button.click(predict, inputs=ctrls, outputs=[output_image])
    return [(ui_component, "PRiR", "PRiR")]

script_callbacks.on_ui_tabs(on_ui_tabs)
