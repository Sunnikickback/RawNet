import argparse
import itertools
import os
import sys
from typing import Dict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.RawNet3 import RawNet3
from models.RawNetBasicBlock import Bottle2neck


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DB_dir", type=str, default="./VoxCeleb1/")
    parser.add_argument("--save_path", type=str, default=None)
    return parser


def extract_speaker_embd(
        model, fn: str, n_samples: int, gpu: bool = False
) -> np.ndarray:
    audio, sample_rate = sf.read(fn)
    if len(audio.shape) > 1:
        raise ValueError(
            f"RawNet3 supports mono input only. Input data has a shape of {audio.shape}."
        )

    audio = audio[:n_samples]
    audio = torch.from_numpy(audio.astype(np.float32)).reshape([1, -1])

    if gpu:
        audio = audio.to("cuda")
    with torch.no_grad():
        output = model(audio)
    return output


def compute_embeddings(DB_dir, save_path=None):
    model = RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        encoder_type="ECA",
        nOut=256,
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
    )
    gpu = False

    model.load_state_dict(
        torch.load(
            "RawNet/python/RawNet3/models/weights/model.pt",
            map_location=lambda storage, loc: storage,
        )["model"]
    )
    model.eval()
    print("RawNet3 initialised & weights loaded!")

    if torch.cuda.is_available():
        print("Cuda available, conducting inference on GPU")
        model = model.to("cuda")
        gpu = True

    with open("../../../cleaned_test_list.txt", "r") as f:
        trials = f.readlines()

        files = list(itertools.chain(*[x.strip().split()[-2:] for x in trials]))

        setfiles = list(set(files))
        setfiles.sort()

        embd_dic = {}
        for f in tqdm(setfiles):
            embd_dic[f] = extract_speaker_embd(
                model, os.path.join(DB_dir, f), n_samples=48000, gpu=gpu
            )[0].detach().cpu().numpy()

        ids_dict = {}
        for key in embd_dic.keys():
            id = key.split('/')[0]
            if id in ids_dict:
                ids_dict[id].append(embd_dic[key])
            else:
                ids_dict[id] = [embd_dic[key]]

    if save_path is not None:
        for id in ids_dict.keys():
            for i, emb in enumerate(ids_dict[id]):
                np.save(os.join(save_path, id, i)+'.npy', emb)
    return ids_dict


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    ids_dict = compute_embeddings(DB_dir=args.DB_dir, save_path=args.save_path)


if __name__ == "__main__":
    main()