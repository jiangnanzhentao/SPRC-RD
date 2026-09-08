import argparse

from configs import get_cfg
from trainer import get_trainer
from util.net import init_training
from util.util import init_checkpoint


def main():
    parser = argparse.ArgumentParser(description="SPRC-RD training and evaluation")
    parser.add_argument(
        "-c",
        "--cfg_path",
        default="configs/benchmark/sprc_rd/sprc_rd_256_200e.py",
    )
    parser.add_argument("-m", "--mode", default="train", choices=["train", "test"])
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--logger_rank", default=0, type=int)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    cfg = get_cfg(parser.parse_args())
    init_training(cfg)
    init_checkpoint(cfg)
    get_trainer(cfg).run()


if __name__ == "__main__":
    main()
