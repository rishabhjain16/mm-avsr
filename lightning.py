import torch
import torchaudio
import json
import os
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from pytorch_lightning import LightningModule


def compute_word_level_distance(seq1, seq2):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    return torchaudio.functional.edit_distance(seq1, seq2)


class ModelModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.modality = args.modality
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        self.model = E2E(
            odim=len(self.token_list), 
            modality=self.modality, 
            ctc_weight=getattr(args, "ctc_weight", 0.1),
            vision_encoder=getattr(args, "vision_encoder", None),
            audio_encoder=getattr(args, "audio_encoder", None),
            decoder=getattr(args, "decoder", "transformer"),
            use_qlora=getattr(args, "use_qlora", False),
            qlora_r=getattr(args, "qlora_r", 16),
            qlora_alpha=getattr(args, "qlora_alpha", 32),
            vision_model_name=getattr(args, "vision_model_name", None),
            audio_model_name=getattr(args, "audio_model_name", None),
            decoder_model_name=getattr(args, "decoder_model_name", None),
        )

        # -- initialise
        if getattr(args, "pretrained_model_path", None):
            ckpt = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
            if getattr(args, "transfer_frontend", False):
                tmp_ckpt = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("trunk.") or k.startswith("frontend3D.")}
                self.model.frontend.load_state_dict(tmp_ckpt)
                print("Pretrained weights of the frontend component are loaded successfully.")
            elif getattr(args, "transfer_encoder", False):
                # For transfer learning, use state_dict from checkpoint
                state_dict = ckpt.get("state_dict", ckpt)
                tmp_ckpt = {k.replace("frontend.",""):v for k,v in state_dict.items() if k.startswith("frontend.")}
                self.model.frontend.load_state_dict(tmp_ckpt)
                tmp_ckpt = {k.replace("proj_encoder.",""):v for k,v in state_dict.items() if k.startswith("proj_encoder.")}
                self.model.proj_encoder.load_state_dict(tmp_ckpt)
                tmp_ckpt = {k.replace("encoder.",""):v for k,v in state_dict.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp_ckpt)
                print("Pretrained weights of the frontend, proj_encoder and encoder component are loaded successfully.")
            else:
                # Handle PyTorch Lightning checkpoint - extract state_dict
                model_state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                # Remove 'model.' prefix if present
                clean_state = {k.replace("model.", ""): v for k, v in model_state.items()}
                self.model.load_state_dict(clean_state)
                print("Pretrained weights of the full model are loaded successfully. (Using checkpoints instead of pth files)")

    def configure_optimizers(self):
        # Use trainable parameters (important for QLoRA)
        if hasattr(self.model, 'get_trainable_parameters'):
            parameters = self.model.get_trainable_parameters()
        else:
            parameters = self.model.parameters()
            
        optimizer = torch.optim.AdamW(parameters, lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.args.warmup_epochs, self.args.max_epochs, len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes)
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)
        x = self.model.frontend(sample.unsqueeze(0))
        x = self.model.proj_encoder(x)
        enc_feat, _ = self.model.encoder(x, None)
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    #ADDED For saving results in a .json fole
    def test_step(self, sample, sample_idx):
        x = self.model.frontend(sample["input"].unsqueeze(0))
        x = self.model.proj_encoder(x)
        enc_feat, _ = self.model.encoder(x, None)
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        actual_token_id = sample["target"]
        actual = self.text_transform.post_process(actual_token_id)

        # Collect results for JSON output
        self.test_results.append({
            "sample_idx": sample_idx,
            "predicted": predicted,
            "actual": actual,
            "file_path": sample.get("file_path", f"sample_{sample_idx}")  # if available
        })

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())
        return

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        batch_size = batch["inputs"].size(0)
        batch_sizes = self.all_gather(batch_size)
        loss *= batch_sizes.size(0) / batch_sizes.sum()  # world size / batch size

        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def _step(self, batch, batch_idx, step_type):
        # Handle multimodal inputs if present
        audio_x = batch.get("audio_inputs", None)
        audio_lengths = batch.get("audio_input_lengths", None)
        
        loss, loss_ctc, loss_att, acc = self.model(
            batch["inputs"], 
            batch["input_lengths"], 
            batch["targets"],
            audio_x=audio_x,
            audio_lengths=audio_lengths
        )
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        else:
            self.log("loss_val", loss, batch_size=batch_size, sync_dist=True)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size, sync_dist=True)
            self.log("loss_att_val", loss_att, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc_val", acc, batch_size=batch_size, sync_dist=True)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.test_results = []  # Initialize list to collect results
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def on_test_epoch_end(self):
        wer = self.total_edit_distance / self.total_length
        self.log("wer", wer)
        
        # Save results to JSON
        output_file = getattr(self.args, 'output_json', 'test_results.json')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Prepare results dictionary
        results = {
            "wer": float(wer),
            "total_samples": len(self.test_results),
            "total_edit_distance": float(self.total_edit_distance),
            "total_length": float(self.total_length),
            "model_path": getattr(self.args, 'pretrained_model_path', 'unknown'),
            "test_file": getattr(self.args, 'test_file', 'unknown'),
            "modality": getattr(self.args, 'modality', 'unknown'),
            "predictions": self.test_results
        }
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        print(f"WER: {wer:.4f} ({self.total_edit_distance}/{self.total_length})")


def get_beam_search_decoder(
    model,
    token_list,
    rnnlm=None,
    rnnlm_conf=None,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
