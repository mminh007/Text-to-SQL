import os
from transformers import TrainerCallback

class EpochCheckpointCallback(TrainerCallback):
    """
    Save a LoRA checkpoint every `save_every_n_epochs` epochs.
    Checkpoints are saved to: {output_dir}/checkpoint-epoch-{epoch}/
    
    Useful for resuming training on Google Colab when GPU quota expires.
    """

    def __init__(self, model, tokenizer, output_dir: str, save_every_n_epochs: int):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.save_every_n_epochs = save_every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if epoch > 0 and epoch % self.save_every_n_epochs == 0:
            ckpt_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
            print(f"\n💾 [EpochCheckpoint] Saving checkpoint at epoch {epoch} → {ckpt_dir}")
            self.model.save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)
            print(f"✅ [EpochCheckpoint] Checkpoint saved.\n")
