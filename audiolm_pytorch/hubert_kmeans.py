from pathlib import Path

import torch
from torch import nn
from einops import rearrange, pack, unpack

import joblib

import fairseq

from torchaudio.functional import resample

from audiolm_pytorch.utils import curtail_to_multiple

import logging
logging.root.setLevel(logging.ERROR)

from transformers import HubertModel

def exists(val):
    return val is not None

class HubertWithKmeans(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(
        self,
        checkpoint_path,
        kmeans_path,
        target_sample_hz = 16000,
        seq_len_multiple_of = None,
        use_mert = False,
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of

        self.use_mert = use_mert
        if not use_mert:
            model_path = Path(checkpoint_path)
            assert model_path.exists(), f'path {checkpoint_path} does not exist'
            checkpoint = torch.load(checkpoint_path)
            load_model_input = {checkpoint_path: checkpoint}
            model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)
            self.model = model[0]
        else:
            self.model = HubertModel.from_pretrained("m-a-p/MERT-v0")
            self.layer = 7 # hardcoded to pull out from this layer in MERT. TODO refactor this later

        kmeans_path = Path(kmeans_path)
        assert kmeans_path.exists(), f'path {kmeans_path} does not exist'

        self.model.eval()

        kmeans = joblib.load(kmeans_path)
        self.kmeans = kmeans

    @property
    def groups(self):
        return 1

    @property
    def codebook_size(self):
        return self.kmeans.n_clusters

    @torch.no_grad()
    def forward(
        self,
        wav_input,
        flatten = True,
        input_sample_hz = None
    ):
        device = wav_input.device

        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        embed = self.model(wav_input, features_only = True)
        # print(f"embed.keys(): {embed.keys()}")
        # padding_mask is also a key but it's None
        print(f"type(wav_input) {type(wav_input)} and shape: {wav_input.shape}")
        print(f"embed['x'] shape: {embed['x'].shape}, embed['features'].shape: {embed['features'].shape}")
        embed, packed_shape = pack([embed['x']], '* d')
        # print(f"wav_input shape: {wav_input.shape}, embed shape: {embed.shape}, packed_shape: {packed_shape}")

        codebook_indices = self.kmeans.predict(embed.cpu().detach().numpy())

        codebook_indices = torch.from_numpy(codebook_indices).to(device).long()
        # print(f"codebook_indices before unpacking: {codebook_indices.shape}")
        if flatten:
            return codebook_indices
        codebook_indices, = unpack(codebook_indices, packed_shape, '*')
        # print(f"codebook_indices after unpacking: {codebook_indices.shape}")
        return codebook_indices
