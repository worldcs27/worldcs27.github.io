import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 5))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT
class Model2Config(object):
    """
    HALO + prior correction (logit adjustment) config.
    Data is loaded from DATA_MIMICIII by default.
    """

    def __init__(
        self,
        total_vocab_size=0,
        code_vocab_size=0,
        label_vocab_size=25,
        special_vocab_size=3,  # start_record, end_record, pad_visit
        n_positions=56,
        n_ctx=48,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        batch_size=48,
        sample_batch_size=256,
        epoch=50,
        lr=1e-4,
        pos_loss_weight=None,
        # Logit adjustment (prior correction)
        logit_adjust_tau=0.2,
        logit_adjust_eps=1e-8,
        logit_adjust_clip=15.0,
        apply_logit_adjust_in_sampling=True,
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.label_vocab_size = label_vocab_size
        self.special_vocab_size = special_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.epoch = epoch
        self.lr = lr
        self.pos_loss_weight = pos_loss_weight

        self.logit_adjust_tau = float(logit_adjust_tau)
        self.logit_adjust_eps = float(logit_adjust_eps)
        self.logit_adjust_clip = float(logit_adjust_clip) if logit_adjust_clip is not None else None
        self.apply_logit_adjust_in_sampling = bool(apply_logit_adjust_in_sampling)

    @property
    def start_record_token(self):
        return self.code_vocab_size + self.label_vocab_size

    @property
    def end_record_token(self):
        return self.start_record_token + 1

    @property
    def pad_visit_token(self):
        return self.start_record_token + 2

