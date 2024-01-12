import random
import SimpleITK as sitk
import json
import numpy as np
import torch
import monai
from copy import deepcopy
from pathlib import Path
from glob import glob
from tqdm import tqdm

import os
import sys
sys.path.append('a')

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from adell_mri.lib.monai_transforms import get_transforms_classification as get_transforms
from adell_mri.lib.modules.config_parsing import parse_config_unet, parse_config_cat
from adell_mri.lib.utils.dataset_filters import filter_dictionary
from adell_mri.lib.utils.network_factories import get_classification_network

class Prostatecancerriskprediction(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # path to image file
        self.image_input_dir = "/input/images/axial-t2-prostate-mri/"
        self.image_input_path = list(Path(self.image_input_dir).glob("*.mha"))[0]

        # load clinical information
        # dictionary with patient_age and psa information
        with open("/input/psa-and-age.json") as fp:
            self.clinical_info = json.load(fp)

        # path to output files
        self.risk_score_output_file = Path("/output/prostate-cancer-risk-score.json")
        self.risk_score_likelihood_output_file = Path("/output/prostate-cancer-risk-score-likelihood.json")
    
    def predict(self):

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        g = torch.Generator()
        g.manual_seed(42)
        rng = np.random.default_rng(42)

        device = "cuda" if torch.cuda.is_available() else "cpu"


        clinical_feature_keys = ['psa']
        

        keys = ['image']
        input_keys = deepcopy(keys)
        
        network_config_vgg = parse_config_cat(os.path.join('adell_mri', 'sample_configs', 'classification-vgg-net.yaml'))
        network_config_res = parse_config_cat(os.path.join('adell_mri', 'sample_configs', 'classification-cat-resnet.yaml'))
        network_config_conv = parse_config_cat(os.path.join('adell_mri', 'sample_configs', 'classification-cat-convnext.yaml'))

        network_config_vgg["batch_size"] = 1
        network_config_res["batch_size"] = 1
        network_config_conv["batch_size"] = 1

        transform_arguments = {
            "keys": keys,
            "mask_key": None,
            "image_masking": False,
            "image_crop_from_mask": False,
            "clinical_feature_keys": clinical_feature_keys,
            "adc_keys": [],
            "target_spacing": [0.3125,0.3125,3.0],
            "crop_size": [128,128,24],
            "pad_size": [128,128,24],
        }

        transforms_prediction = monai.transforms.Compose(
            [
                *get_transforms("pre", **transform_arguments),
                *get_transforms("post", **transform_arguments),
            ]
        )

        global_output = []

        post_proc_fn = torch.nn.Sigmoid()

        prediction_dataset = monai.data.CacheDataset(
            [{"image": self.image_input_path, "psa": self.clinical_info['psa']}],
            transforms_prediction,
            num_workers=8,
            cache_rate=1.0,
        )

        # PL sometimes needs a little hint to detect GPUs.
        torch.ones([1]).to("cuda" if torch.cuda.is_available() else "cpu")

        network_config_vgg["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        network_config_res["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        network_config_conv["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        
        act_fn = "swish"
        batch_preprocessing = None


        vgg = get_classification_network(
            net_type='vgg',
            network_config=network_config_vgg,
            dropout_param=0,
            seed=None,
            n_classes=2,
            keys=keys,
            clinical_feature_keys=clinical_feature_keys,
            train_loader_call=None,
            max_epochs=None,
            warmup_steps=None,
            start_decay=None,
            crop_size=[128,128,24],
            clinical_feature_means=None,
            clinical_feature_stds=None,
            label_smoothing=None,
            mixup_alpha=None,
            partial_mixup=None,
        )

        res = get_classification_network(
            net_type='cat',
            network_config=network_config_res,
            dropout_param=0,
            seed=None,
            n_classes=2,
            keys=keys,
            clinical_feature_keys=clinical_feature_keys,
            train_loader_call=None,
            max_epochs=None,
            warmup_steps=None,
            start_decay=None,
            crop_size=[128,128,24],
            clinical_feature_means=None,
            clinical_feature_stds=None,
            label_smoothing=None,
            mixup_alpha=None,
            partial_mixup=None,
        )

        conv = get_classification_network(
            net_type='cat',
            network_config=network_config_conv,
            dropout_param=0,
            seed=None,
            n_classes=2,
            keys=keys,
            clinical_feature_keys=clinical_feature_keys,
            train_loader_call=None,
            max_epochs=None,
            warmup_steps=None,
            start_decay=None,
            crop_size=[128,128,24],
            clinical_feature_means=None,
            clinical_feature_stds=None,
            label_smoothing=None,
            mixup_alpha=None,
            partial_mixup=None,
        )

        state_dict_vgg = torch.load(os.path.join('adell_mri', 'checkpoints', 'psa_vggnet', 'vggnet-0.00001_fold3_best_epoch=64_V_AUC=0.643.ckpt'))["state_dict"]
        state_dict_res = torch.load(os.path.join('adell_mri', 'checkpoints', 'psa_resnet', 'resnet-0.00001_fold0_best_epoch=201_V_AUC=0.680.ckpt'))["state_dict"]
        state_dict_conv = torch.load(os.path.join('adell_mri', 'checkpoints', 'psa_convnext', 'convnext-0.00001_fold2_best_epoch=2_V_AUC=0.573.ckpt'))["state_dict"]

        state_dict_vgg = {
            k: state_dict_vgg[k]
            for k in state_dict_vgg
            if "loss_fn.weight" not in k
        }
        state_dict_res = {
            k: state_dict_res[k]
            for k in state_dict_res
            if "loss_fn.weight" not in k
        }
        state_dict_conv = {
            k: state_dict_conv[k]
            for k in state_dict_conv
            if "loss_fn.weight" not in k
        }

        vgg.load_state_dict(state_dict_vgg)
        res.load_state_dict(state_dict_res)
        conv.load_state_dict(state_dict_conv)

        vgg = vgg.eval().to(device)
        res = res.eval().to(device)
        conv = conv.eval().to(device)

        # HERE
        #output_dict = {
        #    "iteration": iteration,
        #    "prediction_ids": curr_prediction_ids,
        #    "checkpoint": checkpoint,
        #    "predictions": {},
        #}

        for element in prediction_dataset:
            output_vgg = vgg.forward(
                element["image"].unsqueeze(0).to(device),
                element["tabular"].unsqueeze(0).to(device),
                #**extra_args,
            ).detach()
            output_res = res.forward(
                element["image"].unsqueeze(0).to(device),
                element["tabular"].unsqueeze(0).to(device),
                #**extra_args,
            ).detach()
            output_conv = conv.forward(
                element["image"].unsqueeze(0).to(device),
                element["tabular"].unsqueeze(0).to(device),
                #**extra_args,
            ).detach()
            
            output_vgg = output_vgg.cpu()
            output_res = output_res.cpu()
            output_conv = output_conv.cpu()

            output_vgg = post_proc_fn(output_vgg)
            output_res = post_proc_fn(output_res)
            output_conv = post_proc_fn(output_conv)

            output_vgg = output_vgg.numpy()[0].tolist()[0]
            output_res = output_res.numpy()[0].tolist()[0]
            output_conv = output_conv.numpy()[0].tolist()[0]

            risk_score_likelihood = (output_vgg + output_res + output_conv) / 3

            if risk_score_likelihood > 0.5:
                risk_score = 'High'
            else:
                risk_score = 'Low'
            print('Risk score: ', risk_score)
            print('Risk score likelihood: ', risk_score_likelihood)

            # save case-level class
            with open(str(self.risk_score_output_file), 'w') as f:
                json.dump(risk_score, f)

            # save case-level likelihood
            with open(str(self.risk_score_likelihood_output_file), 'w') as f:
                json.dump(float(risk_score_likelihood), f)

if __name__ == "__main__":
    Prostatecancerriskprediction().predict()
