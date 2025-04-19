from hqq.core.quantize import BaseQuantizeConfig

# TODO: Make your own quant config for DeiT-S
def get_quant_config_deit(model):
    quant_config = {}
    
    n_blocks = len(model.blocks)
    sp = 1
    q2_config = BaseQuantizeConfig(nbits=2, group_size=64)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64)

    for i in range(0, 5):
        quant_config[f'blocks.{i}.attn.qkv'] = q4_config
        quant_config[f'blocks.{i}.attn.proj'] = q4_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q4_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q4_config

    quant_config[f'blocks.{1}.attn.qkv'] = q8_config
    quant_config[f'blocks.{1}.attn.proj'] = q8_config
    quant_config[f'blocks.{1}.mlp.fc1'] = q8_config
    quant_config[f'blocks.{1}.mlp.fc2'] = q8_config

    for i in range(5, n_blocks-sp):
        quant_config[f'blocks.{i}.attn.qkv'] = q8_config
        quant_config[f'blocks.{i}.attn.proj'] = q8_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q8_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q8_config

    quant_config[f'blocks.{5}.attn.qkv'] = q4_config
    quant_config[f'blocks.{5}.attn.proj'] = q4_config
    quant_config[f'blocks.{5}.mlp.fc1'] = q4_config
    quant_config[f'blocks.{5}.mlp.fc2'] = q4_config

    quant_config[f'blocks.{7}.attn.qkv'] = q4_config
    quant_config[f'blocks.{7}.attn.proj'] = q4_config
    quant_config[f'blocks.{7}.mlp.fc1'] = q4_config
    quant_config[f'blocks.{7}.mlp.fc2'] = q4_config

    quant_config[f'blocks.{8}.attn.qkv'] = q4_config
    quant_config[f'blocks.{8}.attn.proj'] = q4_config
    quant_config[f'blocks.{8}.mlp.fc1'] = q4_config
    quant_config[f'blocks.{8}.mlp.fc2'] = q4_config

    quant_config[f'blocks.{9}.attn.qkv'] = q4_config
    quant_config[f'blocks.{9}.attn.proj'] = q4_config
    quant_config[f'blocks.{9}.mlp.fc1'] = q4_config
    quant_config[f'blocks.{9}.mlp.fc2'] = q4_config

    for i in range(n_blocks-sp, n_blocks):
        quant_config[f'blocks.{i}.attn.qkv'] = q2_config
        quant_config[f'blocks.{i}.attn.proj'] = q2_config
        quant_config[f'blocks.{i}.mlp.fc1'] = q2_config
        quant_config[f'blocks.{i}.mlp.fc2'] = q2_config
        
    return quant_config

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q2_config = BaseQuantizeConfig(nbits=2, group_size=64)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64)
    sp = 2

    # Score: 10
    for i in range(0, sp):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q8_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q8_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q8_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q8_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q8_config

    for i in range(sp, n_layers-sp):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config
        
    return quant_config