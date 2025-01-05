
import os
import tempfile
import shutil
import enum
import random
import sys
from typing import Any, Dict, Union, List
from absl import logging
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch, hmmsearch
from alphafold.model import config, data, model
from alphafold.relax import relax
import numpy as np
import pickle
from alphafold.common.confidence import compute_plddt
import hashlib
import jax

logging.set_verbosity(logging.INFO)

alphafold_config = {
    "models_to_relax": "BEST",
    "use_gpu_relax": True,
    "model_preset": "monomer",
    "data_dir": "/data2/zhaoyang/alphafold",
    "jackhmmer_binary_path": "/home/zhaoyang/local/bin/jackhmmer",
    "hhblits_binary_path": "/home/zhaoyang/hhsuite/bin/hhblits",
    "hhsearch_binary_path": "/home/zhaoyang/hhsuite/bin/hhsearch",
    "hmmbuild_binary_path": "/home/zhaoyang/local/bin/hmmbuild",
    "hmmsearch_binary_path": "/home/zhaoyang/local/bin/hmmsearch",
    "kalign_binary_path": "/home/zhaoyang/local/kalign/bin/kalign",
    "template_mmcif_dir": "/data2/zhaoyang/alphafold/pdb_mmcif/mmcif_files",
    "max_template_date": "2021-10-01",
    "obsolete_pdbs_path": "/data2/zhaoyang/alphafold/pdb_mmcif/obsolete.dat",
    "pdb70_database_path": "/data2/zhaoyang/alphafold/pdb70",
    "uniref90_database_path": "/data2/zhaoyang/alphafold/uniref90/uniref90.fasta",
    "mgnify_database_path": "/data2/zhaoyang/alphafold/mgnify/mgy_clusters_2022_05.fa",
    "small_bfd_database_path": "/data2/zhaoyang/alphafold/small_bfd/bfd-first_non_consensus_sequences.fasta",
    "bfd_database_path": "/data2/zhaoyang/alphafold/bfd",  # Add this line
    "uniref30_database_path": "/data2/zhaoyang/alphafold/uniref30",
    "uniprot_database_path": "/data2/zhaoyang/alphafold/uniprot",
    "use_small_bfd": True,
    "use_precomputed_msas": True,
    "random_seed": 42,
    "msa_output_dir": "/data2/zhaoyang/amp"
}


class ModelsToRelax(enum.Enum):
    ALL = 0
    BEST = 1
    NONE = 2


class AlphaFoldPredictor:
    def __init__(self):
        devices = jax.devices()
        assert len(devices) > 0, "No GPU/TPU devices found"
        self.device = devices[0]
        print(f"AlphaFoldPredictor Using device: {self.device}")
        self.config = alphafold_config
        self.local_config = alphafold_config
        self.models_to_relax = ModelsToRelax[self.local_config.get("models_to_relax", "BEST")]
        self.use_gpu_relax = self.local_config.get("use_gpu_relax", False)
        self.num_ensemble = 1 if self.local_config.get("model_preset") != "monomer_casp14" else 8
        self.model_type = "Multimer" if "multimer" in self.local_config.get("model_preset", "") else "Monomer"
        self.random_seed = self.local_config.get("random_seed", random.randrange(sys.maxsize))
        self.msa_output_dir = self.config["msa_output_dir"]
        os.makedirs(self.msa_output_dir, exist_ok=True)
        
        self.data_pipeline = None
        self.model_runners = {}
        self.amber_relaxer = None

        self._initialize_pipeline()
        self._initialize_models()
        self._initialize_relaxer()

    def _initialize_pipeline(self):
        run_multimer_system = "multimer" in self.config.get("model_preset", "")
        template_searcher = (
            hmmsearch.Hmmsearch(
                binary_path=self.config["hmmsearch_binary_path"],
                hmmbuild_binary_path=self.config["hmmbuild_binary_path"],
                database_path=self.config["pdb_seqres_database_path"]
            )
            if run_multimer_system
            else hhsearch.HHSearch(
                binary_path=self.config["hhsearch_binary_path"],
                databases=[os.path.join(self.config["pdb70_database_path"], "pdb70")]
            )
        )
        template_featurizer = (
            templates.HmmsearchHitFeaturizer(
                mmcif_dir=self.config["template_mmcif_dir"],
                max_template_date=self.config["max_template_date"],
                max_hits=20,
                kalign_binary_path=self.config["kalign_binary_path"],
                release_dates_path=None,
                obsolete_pdbs_path=self.config["obsolete_pdbs_path"]
            )
            if run_multimer_system
            else templates.HhsearchHitFeaturizer(
                mmcif_dir=self.config["template_mmcif_dir"],
                max_template_date=self.config["max_template_date"],
                max_hits=20,
                kalign_binary_path=self.config["kalign_binary_path"],
                release_dates_path=None,
                obsolete_pdbs_path=self.config["obsolete_pdbs_path"]
            )
        )
        monomer_pipeline = pipeline.DataPipeline(
            jackhmmer_binary_path=self.config["jackhmmer_binary_path"],
            
            hhblits_binary_path=self.config["hhblits_binary_path"],
            uniref90_database_path=self.config["uniref90_database_path"],
            mgnify_database_path=self.config["mgnify_database_path"],
            bfd_database_path=self.config["bfd_database_path"],
            uniref30_database_path=self.config["uniref30_database_path"],
            small_bfd_database_path=self.config["small_bfd_database_path"],
            template_searcher=template_searcher,
            template_featurizer=template_featurizer,
            use_small_bfd=self.config.get("use_small_bfd", False),
            use_precomputed_msas=self.config.get("use_precomputed_msas", False)
        )
        if run_multimer_system:
            self.data_pipeline = pipeline_multimer.DataPipeline(
                monomer_data_pipeline=monomer_pipeline,
                jackhmmer_binary_path=self.config["jackhmmer_binary_path"],
                uniprot_database_path=self.config["uniprot_database_path"],
                use_precomputed_msas=self.config.get("use_precomputed_msas", False)
            )
        else:
            self.data_pipeline = monomer_pipeline
    
    def _generate_precomputed_msa_path(self, sequence: str):
        """Generate a unique MSA path based on the sequence hash."""
        sequence_hash = hashlib.md5(sequence.encode('utf-8')).hexdigest()
        return os.path.join(self.msa_output_dir, f"precomputed_msa_features_{sequence_hash}.pkl")


    def _initialize_models(self):
        visible_devices = jax.devices("gpu")
        breakpoint()
        print(f"Visible devices: {visible_devices}")
        device = visible_devices[0]
        logging.info(f"Initializing models on device: {device}")

        # Use only the first model from the preset list
        model_names = config.MODEL_PRESETS[self.config.get("model_preset", "monomer")][:1]  # Only use the first model
        for model_name in model_names:
            model_config = config.model_config(model_name)
            if "multimer" in self.config.get("model_preset", ""):
                model_config.model.num_ensemble_eval = self.num_ensemble
            else:
                model_config.data.eval.num_ensemble = self.num_ensemble
            model_config.model.num_recycle = 1
            model_params = data.get_model_haiku_params(
                model_name=model_name, data_dir=self.config["data_dir"]
            )
            model_runner = model.RunModel(model_config, model_params)

            # Explicitly move parameters to the selected device
            model_runner.params = jax.device_put(model_runner.params, device)
            model_runner.config.model_device = device
            self.model_runners[model_name] = model_runner

    def _initialize_relaxer(self):
        self.amber_relaxer = relax.AmberRelaxation(
            max_iterations=0,
            tolerance=2.39,
            stiffness=10.0,
            exclude_residues=[],
            max_outer_iterations=3,
            use_gpu=self.use_gpu_relax
        )

    def predict_structure(self, fasta_contents: List[str]):
        results = []
        for fasta_content in fasta_contents:
            coords, confidence, padding_mask = self._process_single_sequence(fasta_content)
            results.append((coords, confidence, padding_mask))
        return results

    def _process_single_sequence(self, fasta_content: str):
        sequence = "".join(fasta_content.strip().split("\n")[1:])
        logging.info(f"Predicting structure for sequence of length {len(sequence)}")

        tmp_fasta_path = None
        try:
            tmp_fasta_path = tempfile.NamedTemporaryFile(mode='w', delete=False).name
            with open(tmp_fasta_path, "w") as tmp_fasta_file:
                tmp_fasta_file.write(fasta_content)

            precomputed_msa_path = self._generate_precomputed_msa_path(sequence)

            if self.config.get("use_precomputed_msas", False) and os.path.exists(precomputed_msa_path):
                logging.info(f"Using precomputed MSA from {precomputed_msa_path}")
                with open(precomputed_msa_path, "rb") as msa_file:
                    feature_dict = pickle.load(msa_file)
            else:
                feature_dict = self.data_pipeline.process(
                    input_fasta_path=tmp_fasta_path,
                    msa_output_dir=self.msa_output_dir
                )
                with open(precomputed_msa_path, "wb") as msa_file:
                    pickle.dump(feature_dict, msa_file)

            coords_lst, confidence_lst, padding_mask_lst = [], [], []
            max_seq_len = len(sequence)

            for _, model_runner in self.model_runners.items():
                processed_features = model_runner.process_features(feature_dict, random_seed=self.random_seed)
                prediction_result = model_runner.predict(processed_features, random_seed=self.random_seed)

                coords = prediction_result['structure_module']['final_atom_positions']
                confidence_scores = compute_plddt(prediction_result['predicted_lddt']['logits'])
                padding_mask = processed_features['residue_index'] < max_seq_len
                
                if padding_mask.shape[0] != coords.shape[0]:
                    padding_mask = padding_mask[0]
                    
                coords_padded = np.zeros((max_seq_len, coords.shape[1], coords.shape[2]))
                coords_padded[:coords.shape[0], :, :] = coords
                
                confidence_padded = np.zeros((max_seq_len,))
                confidence_padded[:len(confidence_scores)] = confidence_scores
                
                padding_mask_padded = np.zeros((max_seq_len,), dtype=bool)
                padding_mask_padded[:len(padding_mask)] = padding_mask

                coords_lst.append(coords_padded)
                confidence_lst.append(confidence_padded)
                padding_mask_lst.append(padding_mask_padded)

            coords = np.stack(coords_lst, axis=0)
            confidence = np.stack(confidence_lst, axis=0)
            padding_mask = np.stack(padding_mask_lst, axis=0)

            return coords, confidence, padding_mask

        finally:
            if tmp_fasta_path:
                os.remove(tmp_fasta_path)







       

