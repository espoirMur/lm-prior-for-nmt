import os
import torch
import argparse
from torch.utils.data import DataLoader
from helpers.opts import exp_options
from models.sent_lm_callbacks import SentLMSamplesCallback
from models.sent_lm_trainer import SentLMTrainer
from modules.data.collates import LMCollate
from modules.data.datasets import SequenceDataset
from modules.data.samplers import BucketTokensSampler, \
    TokenBatchSampler
from modules.initializations import model_init
from modules.models import RNNLM, TransformerLM
from modules.callbacks import LossCallback, CheckpointCallback, \
    GradientCallback
from sys_config import MODEL_CNF_DIR


def run(config):
    # ------------------------------------------------------------------
    # Data Loading and Preprocessing
    # ------------------------------------------------------------------
    print("Building training dataset...")
    train_set = SequenceDataset(config["data"]["train_path"], **config["data"])
    print(train_set)
    print("Building validation dataset...")
    val_set = SequenceDataset(config["data"]["val_path"], vocab=train_set.vocab,
                              **config["data"])
    print(val_set)

    train_sampler = BucketTokensSampler(train_set.lengths,
                                        config["batch_tokens"],
                                        shuffle=True)
    val_sampler = TokenBatchSampler(val_set.lengths, config["batch_tokens"])

    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              num_workers=config["cores"],
                              pin_memory=config["pin_memory"],
                              collate_fn=LMCollate())
    val_loader = DataLoader(val_set, batch_sampler=val_sampler,
                            num_workers=0,
                            pin_memory=False,
                            collate_fn=LMCollate())
    
    embedding, missing = train_set.vocab.read_fasttext(config["data"]["embeddings"])

    embedding = torch.from_numpy(embedding).float()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    ntokens = len(train_set.vocab)
    print("---" * 10)
    print("we are training with the following vocabulary {}".format(ntokens))
    print("---" * 10)
    if config["model"]["type"] == "rnn":
        model = RNNLM(ntokens, **config["model"], embeddings=embedding)
    elif config["model"]["type"] == "transformer":
        model = TransformerLM(ntokens, **config["model"])
    else:
        raise NotImplementedError

    model_init(model, **config.get("init", {}))

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------
    callbacks = [
        LossCallback(config["logging"]["log_interval"]),
        SentLMSamplesCallback(config["logging"]["samples_interval"]),
        CheckpointCallback(config["logging"]["checkpoint_interval"],
                           only_best=True,
                           early_stop=config["optim"].get("early_stop", 100)),
        GradientCallback(config["logging"]["log_interval"])
    ]

    # ------------------------------------------------------------------
    # Training Pipeline
    # ------------------------------------------------------------------
    trainer = SentLMTrainer(model, train_loader, val_loader, config,
                            config["device"],
                            callbacks=callbacks,
                            src_dirs=config["src_dirs"],
                            resume_cp=config["resume_cp"],
                            resume_state_id=config["resume_state_id"])

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------

    for epoch in range(config["epochs"]):
        train_loss = trainer.train_epoch()
        val_loss = trainer.eval_epoch()

        if trainer.early_stop:
            print("Stopping early ...")
            break

    trainer.exp.finalize()

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="configuration file")
    args, extra_args = parser.parse_known_args()
    default_config = os.path.join(MODEL_CNF_DIR, args.config)
    print("the default configurations are ", default_config)
    _config = exp_options(default_config)
    trained_model = run(_config)
