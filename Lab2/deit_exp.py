import torch
import random
import os
import numpy as np

from hqq.utils.patching import recommended_inductor_config_setter
from hqq_utils import AutoHQQTimmModel, get_size_of_model
from utils import prepare_data, evaluate_model
from hqq.core.quantize import BaseQuantizeConfig
from matplotlib import pyplot as plt

def get_quant_config_deit(model, q_layers):
    quant_config = {}
    
    n_blocks = len(model.blocks)
    sp = 1
    q2_config = BaseQuantizeConfig(nbits=2, group_size=64)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64)

    for i in range(n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = q8_config
        quant_config[f'blocks.{i}.attn.proj'] = q8_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q8_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q8_config

    for i in q_layers:
        quant_config[f'blocks.{i}.attn.qkv'] = q4_config
        quant_config[f'blocks.{i}.attn.proj'] = q4_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q4_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q4_config
        
    return quant_config

def main(q_layer=0):
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    recommended_inductor_config_setter()
    
    device = 'cuda:0'
    batch_size = 16
    _, test_loader, _ = prepare_data(batch_size)
    
    model = torch.load('./0.9099_deit3_small_patch16_224.pth', map_location='cpu', weights_only=False)
    model = model.to(device)
    model.eval()
    
    # Config to align HQQ 
    model.device = 'cuda:0'
    model.dtype = torch.float32
    ##################################### 

    # TODO: Quantize
    quant_config = get_quant_config_deit(model, q_layer)
    
    AutoHQQTimmModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float32, device=device)
    
    from hqq.utils.patching import prepare_for_inference
    prepare_for_inference(model)
    torch.cuda.empty_cache()
    
    acc_after_quant = evaluate_model(model, test_loader, 'cuda:0')
    print(f'Accuracy After Quant: {acc_after_quant}%')
    print(f'Model Size (MiB) {get_size_of_model(model)/ (1024 ** 2)} MiB')
    
    score = 20 - max(0, 90 - acc_after_quant) * 10 + (17 - get_size_of_model(model) / (1024 ** 2))
    print(f'Score: {score}')
    return acc_after_quant, get_size_of_model(model) / (1024 ** 2), score

    
if __name__ == '__main__':
    n_layer = 12
    num_permutations = 2**n_layer
    accs = []
    sizes = []
    scores = []
    # Generate all permutations of layer indices
    for i in range(num_permutations):
        q_layers = []
        for j in range(n_layer):
            if (i >> j) & 1:
                q_layers.append(j)
        print(f"Quant to 4 bits: {q_layers}")
        acc, size, score = main(i)
        accs.append(acc)
        sizes.append(size)
        scores.append(score)        
    
    # Plotting
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.plot(range(n_layer), accs, color=color, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(range(n_layer))
    ax1.set_xticklabels(range(n_layer), rotation=45)
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Model Size (MiB)', color=color)
    ax2.plot(range(n_layer), sizes, color=color, label='Model Size')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 20)
    ax2.legend(loc='upper right')
    fig.tight_layout()
    plt.title('Accuracy and Model Size vs Layer (One Layer Quantization)')
    # Save results
    results = {
        'Layer': list(range(n_layer)),
        'Accuracy (%)': accs,
        'Model Size (MiB)': sizes,
        'Score': scores
    }
    results_file = 'results.txt'
    with open(results_file, 'w') as f:
        for i in range(n_layer):
            f.write(f"Layer {i}: Accuracy: {accs[i]}%, Model Size: {sizes[i]} MiB, Score: {scores[i]}\n")
    print(f"Results saved to {results_file}")
    # Save the plot
    fig.savefig('accuracy_model_size_vs_layer.png')
    print("Plot saved as accuracy_model_size_vs_layer.png")