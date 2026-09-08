from argparse import Namespace


class cfg_model_sprc_rd(Namespace):
    def __init__(self):
        super().__init__()
        self.model_t = Namespace(
            name="timm_wide_resnet50_2",
            kwargs=dict(
                pretrained=False,
                checkpoint_path="",
                strict=False,
                features_only=True,
                out_indices=[1, 2, 3],
            ),
        )
        self.model_s = Namespace(
            name="de_wide_resnet50_2",
            kwargs=dict(pretrained=False, checkpoint_path="", strict=False),
        )
        self.model = Namespace(
            name="sprc_rd",
            kwargs=dict(
                pretrained=False,
                checkpoint_path="",
                strict=True,
                model_t=self.model_t,
                model_s=self.model_s,
                backbone="wide_resnet50_2",
                teacher_init="reference_exact",
                teacher_pretrained=True,
                teacher_checkpoint="",
                model_checkpoint_path="",
                embed_dims=(128, 160, 256),
                num_prototypes=(4, 4, 4),
                min_topk=(4, 2, 1),
                relation_weights=(0.15, 0.30, 0.45),
                temperature=0.2,
                fusion_dim=256,
                fusion_blocks=2,
                norm="bn",
                relation_warmup_epochs=10,
                relation_ramp_epochs=10,
                relation_momentum=0.5,
                relation_smoothing=1.0,
                quant_scale=1e6,
                cf_posterior_temperature=0.35,
                cf_relation_floor=1e-4,
                cf_scale_weights=(0.15, 0.25, 0.60),
                cf_gain_margin=0.08,
                cf_gain_temperature=0.12,
                cf_max_intervention=1.0,
                cf_gate_grad=False,
                decoder_lr=0.005,
                bottleneck_lr=0.001,
            ),
        )
