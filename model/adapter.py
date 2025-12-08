import torch
import torch.nn as nn
from ml_collections import config_dict

try:
    from .sdt import SequenceVDTContinuousModelv2
except ImportError:
    print("Error: Could not import SequenceVDTContinuousModelv2. Ensure the new architecture code is saved as 'model_v2.py'")
    raise

class ModelConfigAdapter:
    """
    Translates the old OmegaConf structure (SEDD) to the 
    new ml_collections.ConfigDict structure (SequenceVDT).
    """
    @staticmethod
    def adapt(old_conf):
        # 1. Unpack OmegaConf / Dict
        # Hydra often nests the model config under 'model', but standard access works best
        if hasattr(old_conf, 'model'):
            c_model = old_conf.model
        else:
            c_model = old_conf

        # Data/Graph config might be at root or under 'graph'
        if hasattr(old_conf, 'graph'):
            c_graph = old_conf.graph
        else:
            c_graph = {}

        # 2. Initialize new config
        cfg = config_dict.ConfigDict()
        cfg.framework = "discrete_sedd" 
        
        # --- Pre-calculate Vocab Size ---
        # Critical Fix: 'absorb' graph type adds a mask token.
        # The model output dimension MUST match the full vocabulary size (tokens + mask)
        # otherwise scatter_ operations inside the model will trigger CUDA asserts.
        graph_type = getattr(c_graph, 'type', 'absorb')
        num_tokens = getattr(old_conf, 'tokens', 2)
        
        if graph_type == 'absorb':
            full_vocab_size = num_tokens + 1
        else:
            full_vocab_size = num_tokens

        # --- Model Section ---
        cfg.model = config_dict.ConfigDict()
        
        # Dimensions
        cfg.model.embed_dim = getattr(c_model, 'hidden_size', 128)
        
        # Note: Your config has 'cond_dim' (128). The new VDT model uses 'embed_dim' 
        # for time embeddings internally. Since they match (128==128) in your config, this is fine.
        
        cfg.model.n_blocks = getattr(c_model, 'n_blocks', 2)
        cfg.model.n_heads = getattr(c_model, 'n_heads', 2)
        cfg.model.dropout = getattr(c_model, 'dropout', 0.1)
        
        # Output Dim = Full Vocab Size (including mask if absorb)
        cfg.model.out_dim = full_vocab_size
        
        # Patching
        # Set to 1 to behave like the standard SEDD/DDiT (no compression)
        cfg.model.patch_size = 1 
        
        # Content Dims
        cfg.model.content_dim_discrete = cfg.model.embed_dim
        cfg.model.content_dim_continuous = 1 
        
        # Advanced Head & Attn Config
        cfg.model.head_type = "hybrid_attn_v2"
        cfg.model.use_adaln = True
        cfg.model.use_swiglu = False 
        cfg.model.use_flash_attn = True # Enabled as requested
        
        # Fourier Features (Defaults)
        cfg.model.n_fourier_global = 8
        cfg.model.n_fourier_local = 4
        cfg.model.rpb_max_distance = 64
        cfg.model.dim_ff = getattr(c_model, 'dim_ff', cfg.model.embed_dim * 4)

        # --- Data Section ---
        cfg.data = config_dict.ConfigDict()
        cfg.data.vocab_size = full_vocab_size

        # --- Diffusion Section ---
        cfg.diffusion = config_dict.ConfigDict()
        cfg.diffusion.discrete = config_dict.ConfigDict()
        cfg.diffusion.discrete.q_matrix_type = graph_type
        
        # Scale by sigma
        cfg.model.scale_by_sigma = getattr(c_model, 'scale_by_sigma', True)

        return cfg

class SEDDCompatibilityWrapper(nn.Module):
    """
    Drop-in replacement for the old SEDD class.
    """
    def __init__(self, old_config):
        super().__init__()
        
        # 1. Adapt the config automatically
        self.new_cfg = ModelConfigAdapter.adapt(old_config)
        
        # 2. Initialize the new VDT model
        self.model = SequenceVDTContinuousModelv2(self.new_cfg)
        
        # 3. Print verification
        print(f"\n[Model Adapter] Successfully migrated config '{getattr(old_config, 'name', 'unknown')}'")
        print(f" - Embed Dim: {self.new_cfg.model.embed_dim}")
        print(f" - Patch Size: {self.new_cfg.model.patch_size}")
        print(f" - Vocab Size: {self.new_cfg.data.vocab_size} (Tokens: {self.new_cfg.model.out_dim} + Mask)")
        print(f" - Flash Attn: {self.new_cfg.model.use_flash_attn}\n")

    def forward(self, indices, sigma):
        # Pass through to new model
        # indices: [Batch, Length] (LongTensor)
        # sigma:   [Batch] (FloatTensor)
        
        # FIX: Flash Attention strictly requires fp16 or bf16 inputs. 
        # The training loop appears to be running in fp32, so we enforce 
        # mixed precision context here (just like the original SEDD model did).
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return self.model(indices, sigma)

# Expose as SEDD so your train.py imports work without change
SEDD = SEDDCompatibilityWrapper