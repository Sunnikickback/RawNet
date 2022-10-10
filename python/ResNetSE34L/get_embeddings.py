import argparse
import glob
import itertools
import os
import sys
from typing import Dict

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from SpeakerNet import SpeakerNet


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DB_dir", type=str, default="../../../VoxCeleb1/")
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


def loadParameters(model, path):
    self_state = model.state_dict()
    loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
    if len(loaded_state.keys()) == 1 and "model" in loaded_state:
        loaded_state = loaded_state["model"]
        newdict = {}
        delete_list = []
        for name, param in loaded_state.items():
            new_name = "__S__."+name
            newdict[new_name] = param
            delete_list.append(name)
        loaded_state.update(newdict)
        for name in delete_list:
            del loaded_state[name]
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("module.", "")

            if name not in self_state:
                print("{} is not in the model.".format(origname))
                continue

        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
            continue

        self_state[name].copy_(param)


def compute_embeddings(DB_dir, save_path=None):
    model = SpeakerNet("ResNetSE34L")
    gpu = False

    loadParameters(model, "baseline_lite_ap.model")

    model.eval()
    print("ResNet initialised & weights loaded!")

    if torch.cuda.is_available():
        print("Cuda available, conducting inference on GPU")
        model = model.to("cuda")
        gpu = True

    with open("../../trials/list_test_all_cleaned.txt", "r") as f:
        trials = f.readlines()

        files = list(itertools.chain(*[x.strip().split()[-2:] for x in trials]))

        setfiles = list(set(files))
        setfiles.sort()

        embd_dic = {}
        for f in tqdm(setfiles):
            embd_dic[f] = extract_speaker_embd(
                model, os.path.join(DB_dir, f), n_samples=48000, gpu=gpu
            )[0].detach().cpu().numpy()
            if save_path is not None:
                id = f.split('/')[0]
                result_dir = os.path.join(save_path, id)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                np.save(os.path.join(result_dir, get_next_file_name(result_dir)), embd_dic[f])

    return embd_dic


def get_next_file_name(result_dir):
    list = os.listdir(result_dir)
    if len(list) != 0:
        return str(int(list[-1].split('.')[0])+1)+".npy"
    else:
        return "0.npy"


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    embd_dic = compute_embeddings(DB_dir=args.DB_dir, save_path=args.save_path)


if __name__ == "__main__":
    main()