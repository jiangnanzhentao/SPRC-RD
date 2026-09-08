from argparse import Namespace


class cfg_common(Namespace):
    def __init__(self):
        super().__init__()
        self.seed = 111
        self.size = 256
        self.epoch_full = 200
        self.fvcore_is = False
        self.fvcore_b = 1
        self.fvcore_c = 3
        self.vis = False
        self.vis_dir = None
        self.evaluator = Namespace()
        self.optim = Namespace(lr=0.005, kwargs=dict(name="adam", betas=(0.5, 0.999)))
        self.trainer = Namespace()
        self.trainer.name = "SPRCRDTrainer"
        self.trainer.checkpoint = "runs"
        self.trainer.logdir_sub = ""
        self.trainer.resume_dir = ""
        self.trainer.cuda_deterministic = True
        self.trainer.epoch_full = self.epoch_full
        self.trainer.scheduler_kwargs = dict(
            name="multistep", use_iters=True, milestones=[100, 120], gamma=0.2
        )
        self.trainer.find_unused_parameters = False
        self.trainer.sync_BN = "none"
        self.trainer.dist_BN = ""
        self.trainer.scaler = "none"
        self.trainer.data = Namespace(
            batch_size=32,
            batch_size_per_gpu=None,
            batch_size_test=None,
            batch_size_per_gpu_test=1,
            num_workers_per_gpu=4,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )
        self.loss = Namespace(
            loss_terms=[dict(type="CosLoss", name="cos", avg=False, lam=1.0)],
            clip_grad=None,
            create_graph=False,
            retain_graph=False,
        )
        self.logging = Namespace(
            log_terms_train=[
                dict(name="batch_t", fmt=":>5.3f", add_name="avg"),
                dict(name="data_t", fmt=":>5.3f"),
                dict(name="optim_t", fmt=":>5.3f"),
                dict(name="lr", fmt=":>7.6f"),
                dict(name="total", fmt=":>6.4f", add_name="avg"),
                dict(name="rd", fmt=":>6.4f", add_name="avg"),
                dict(name="cf", fmt=":>6.4f", add_name="avg"),
                dict(name="gain", fmt=":>6.4f", add_name="avg"),
                dict(name="img", fmt=":>6.4f", add_name="avg"),
            ],
            log_terms_test=[
                dict(name="batch_t", fmt=":>5.3f", add_name="avg"),
                dict(name="cos", fmt=":>5.3f", add_name="avg"),
            ],
            train_reset_log_per=50,
            train_log_per=50,
            test_log_per=50,
        )
