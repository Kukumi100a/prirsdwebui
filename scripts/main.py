import modules.scripts as scripts
import gradio as gr
import os
import pathlib
import torch

from modules import script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules.ui_components import ResizeHandleRow
from modules import shared

from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy

@torch.inference_mode()
@torch.no_grad()

def predict():
    ##Tymczasowa implementacja wyniku
    return outputs


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
                ctrls = [used_technology, type_of_operation, amount_of_threads]
            with gr.Column():
                output_gallery = gr.Gallery(label='Gallery', show_label=True, object_fit='contain',
                                            visible=True, columns=4)
                output_wykresy = gr.Gallery(label='Wykresy', show_label=True, object_fit='contain',
                                            visible=True, columns=4)
        generate_button.click(predict, inputs=ctrls, outputs=[output_gallery])
    return [(ui_component, "PRiR", "PRiR")]

script_callbacks.on_ui_tabs(on_ui_tabs)
