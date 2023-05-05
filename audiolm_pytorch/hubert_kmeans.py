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

from transformers import HubertModel, Wav2Vec2Processor

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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
        output_layer = 9,
        use_mert=False,
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.output_layer = output_layer

        self.use_mert = use_mert
        if not use_mert:
            model_path = Path(checkpoint_path)
            assert model_path.exists(), f'path {checkpoint_path} does not exist'
            checkpoint = torch.load(checkpoint_path)
            load_model_input = {checkpoint_path: checkpoint}
            model, *_ = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)
            self.model = model[0]
        else:
            self.model = HubertModel.from_pretrained("m-a-p/MERT-v0") # is nn.Module
            # print(f"model is nn module? {isinstance(self.model, nn.Module)}")
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft") # is not nn.Module
            # print(f"processor is nn module? {isinstance(self.processor, nn.Module)}")
            self.layer = output_layer

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
        # on cuda, 1 x length defined in semantic/coarse transformer
        # print(f"wav input shape before processing: {wav_input.shape} and device: {wav_input.device}")

        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        if self.use_mert:
            # wav_input is batch x samples
            # # mert_input is {
            # #   "input_values": processed array of wav_input (it's not copied directly by self.processor),
            # #   "attention_mask": equivalent of torch.ones(mert_input["input_values"].shape)
            # # }
            # sampling_rate = input_sample_hz if exists(input_sample_hz) else self.target_sample_hz
            # mert_input = self.processor(wav_input[0], sampling_rate=sampling_rate, return_tensors="pt") # TODO: is there a problem with batching here? feel like something is wrong here.
            # mert_input["attention_mask"] = mert_input["attention_mask"].cuda()
            # mert_input["input_values"] = mert_input["input_values"].cuda()
            # print(f"wav_input.is_cuda {wav_input.is_cuda}")
            # print(f"mert shape {mert_input['input_values'].shape} and keys {mert_input.keys()}")
            #
            # So transformers processor is totally busted and we just want to have zero mean and unit variance.
            # just normalize, we don't need any of this extra padding stuff
            # the problem is that the transformers library expects you to do all data processing in CPU and forces you
            # to use numpy ndarrays for everything. This defeats the point here because wav_input is already in cuda by this point.
            if wav_input.shape[0] != 1:
                raise AssertionError(f"This normalization code was written under the assumption that for some reason "
                                     f"batch size is only ever going to be 1 in the mert case. If you're seeing this, "
                                     f"that hasn't happened. instead wav_input.shape is {wav_input.shape}")
            mert_input = {
                "input_values": (wav_input - wav_input.mean()) / torch.sqrt(wav_input.var() + 1e-7),
                "attention_mask": torch.ones_like(wav_input)
            }
            outputs = self.model(**mert_input, output_hidden_states=True) # 1 x everything.
            all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze() # 1 x 13 layers x timesteps x 768 feature_dim
            # print(f"all_layer_hidden_states.shape {all_layer_hidden_states.shape} and device {all_layer_hidden_states.device}") # cuda
            embed = all_layer_hidden_states[self.layer] # timesteps x 768 feature_dim
            packed_shape = [torch.Size([1, embed.shape[0]])] # extremely hacky way to replicate einops.pack behavior for packed_shape
        else:
            embed = self.model(
                wav_input,
                features_only=True,
                mask=False,  # thanks to @maitycyrus for noticing that mask is defaulted to True in the fairseq code
                output_layer=self.output_layer
            )
            # print(f"embed.keys(): {embed.keys()}")
            # padding_mask is also a key but it's None
            # print(f"type(wav_input) {type(wav_input)} and shape: {wav_input.shape}") # 1 x 10240 in the example, dependent on max_length or whatever that parameter is
            # print(f"embed['x'] shape: {embed['x'].shape}, embed['features'].shape: {embed['features'].shape}") # 1 x 31 x 768 for both.
            # this is the number of tokens-- derived via 16 KHz sampling to 50 Hz tokens -> 320x reduction, so
            # 10240 / 320 = 32 rounds down to 31.
            embed, packed_shape = pack([embed['x']], '* d')
        # wav_input shape: torch.Size([1, 10240]), embed shape: torch.Size([31, 768]), packed_shape: [torch.Size([1, 31])]
        # print(f"self.use_mert: {self.use_mert}, wav_input shape: {wav_input.shape}, embed shape: {embed.shape}, packed_shape: {packed_shape}")

        codebook_indices = self.kmeans.predict(embed.cpu().detach().numpy())

        codebook_indices = torch.from_numpy(codebook_indices).to(device).long()
        # print(f"codebook_indices before unpacking: {codebook_indices.shape}") # [31]
        if flatten:
            return codebook_indices
        codebook_indices, = unpack(codebook_indices, packed_shape, '*')
        # print(f"codebook_indices after unpacking: {codebook_indices.shape}") # [1, 31]
        return codebook_indices
