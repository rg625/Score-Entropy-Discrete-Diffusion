from ml_collections import config_dict

def get_config():
    cfg = config_dict.ConfigDict()

    # ---- experiment ----
    cfg.experiment = "wikitext103/autoregressive/medium"
    cfg.device = "cuda:0"

    # ---- system ----
    cfg.system = config_dict.ConfigDict()
    cfg.system.distributed = False
    cfg.system.global_rank = 0
    cfg.system.local_rank = 0
    cfg.system.world_size = 1

    # ---- data ----
    cfg.data = config_dict.ConfigDict()
    cfg.data.dataset = "wikitext-103"
    cfg.data.representation = "tokens"
    cfg.data.sequence_len_tokens = 512 + 1

    cfg.data.root = "./datasets/wikitext-103"
    cfg.data.tokenizer_path = "./datasets/wikitext-103/tokenizer_wiki_65k.json"

    cfg.data.num_workers = 8
    cfg.data.prefetch_factor = 4
    cfg.data.pin_memory = True
    cfg.data.drop_last_train = True
    cfg.data.drop_last_val = False

    # ---- model (same width/heads, slightly less deep) ----
    cfg.model = config_dict.ConfigDict()
    cfg.model.name = "gpt_16L_768d"
    cfg.model.vocab_size = 65536
    cfg.model.max_seq_len = 512

    cfg.model.n_layer = 16     
    cfg.model.n_head  = 16
    cfg.model.d_model = 768
    cfg.model.mlp_mult = 4.0
    cfg.model.dropout = 0.1         # <-- strongly recommended for WikiText
    cfg.model.rope_base = 10000.0
    cfg.model.use_flash_attn = True

    # ---- optim ----
    cfg.optim = config_dict.ConfigDict()
    cfg.optim.optimizer = "adamw"
    cfg.optim.lr = 2e-4             # stable with effective batch 256
    cfg.optim.beta1 = 0.9
    cfg.optim.beta2 = 0.95
    cfg.optim.eps = 1e-8
    cfg.optim.weight_decay = 0.1
    cfg.optim.grad_clip = 1.0
    cfg.optim.scheduler = "cosine"
    cfg.optim.warmup = 4000
    cfg.optim.fused = True

    # ---- train ----
    cfg.train = config_dict.ConfigDict()
    cfg.train.seed = 0
    cfg.train.deterministic = False

    cfg.train.batch_size = 64
    cfg.train.grad_accum_steps = 4  # effective batch = 256

    cfg.train.epochs = 50           # adjust to your step budget

    cfg.train.use_fp16 = True       # will use bf16 on A100
    cfg.train.use_compile = False
    cfg.train.compile_mode = "default"

    cfg.train.ema_decay = 0.999
    cfg.train.eval_with_ema = True

    cfg.train.save_last = True
    cfg.train.save_top_k = 3
    cfg.train.checkpoint_mode = "min"

    # ---- logging ----
    cfg.logging = config_dict.ConfigDict()
    cfg.logging.log_freq = 50

    # ---- generation callback ----
    cfg.train.generation = config_dict.ConfigDict()
    cfg.train.generation.enabled = True
    cfg.train.generation.every_epochs = 1
    cfg.train.generation.num_samples = 2
    cfg.train.generation.max_new_tokens = 200
    cfg.train.generation.temperature = 1.0
    cfg.train.generation.top_k = 50
    cfg.train.generation.prompt = "The meaning of life is"
    cfg.train.generation.start_token_id = 0
    cfg.train.generation.use_ema = True

    return cfg
