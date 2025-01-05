# alphafold_utils.py

import logging
import math
import numpy as np
import os
import pickle
import random
import time
import json

import torch

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.data.tools import hhsearch, hmmsearch
from openfold.np import protein
from openfold.utils.script_utils import (
    load_models_from_command_line,
    parse_fasta,
    run_model,
    prep_output,
    relax_protein,
)
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)

from scripts.precompute_embeddings import EmbeddingGenerator
from scripts.utils import add_data_args

class AlphaFoldPredictor:
    TRACING_INTERVAL = 50

    def __init__(self, config_preset="model_1_ptm", output_dir=os.getcwd(), model_device="cuda:0", 
                 use_precomputed_alignments=None, experiment_config_json=None, 
                 long_sequence_inference=False, use_deepspeed_evoformer_attention=False):
        """
        Initialize the AlphaFoldPredictor.

        Args:
            config_preset (str): Model configuration preset.
            output_dir (str): Directory to store outputs.
            model_device (str): Device to run the model on.
            use_precomputed_alignments (str, optional): Path to precomputed alignments.
            experiment_config_json (str, optional): Path to a JSON config file.
            long_sequence_inference (bool): Whether to enable long sequence inference.
            use_deepspeed_evoformer_attention (bool): Whether to use DeepSpeed evoformer attention.
        """
        breakpoint()
        self.output_dir = output_dir
        self.model_device = model_device
        self.use_precomputed_alignments = use_precomputed_alignments
        self.experiment_config_json = experiment_config_json

        self.logger = self._setup_logger()
        self._setup_torch()
        self.config = self._load_config(config_preset)
        self.data_processor = self._initialize_data_processor()
        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        self.alignment_dir = self._setup_alignment_dir()
        self.feature_dicts = {}
        self.model_generator = self._load_models()

    def _setup_logger(self):
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def _setup_torch(self):
        torch_versions = torch.__version__.split(".")
        torch_major_version = int(torch_versions[0])
        torch_minor_version = int(torch_versions[1])
        if (
            torch_major_version > 1
            or (torch_major_version == 1 and torch_minor_version >= 12)
        ):
            # Gives a large speedup on Ampere-class GPUs
            torch.set_float32_matmul_precision("high")

        torch.set_grad_enabled(False)

    def _load_config(self, config_preset):
        config = model_config(
            config_preset,
            long_sequence_inference=self.config_preset if hasattr(self, 'config_preset') else False,
            use_deepspeed_evoformer_attention=self.config_preset if hasattr(self, 'config_preset') else False,
        )

        if self.experiment_config_json:
            with open(self.experiment_config_json, 'r') as f:
                custom_config_dict = json.load(f)
            config.update_from_flattened_dict(custom_config_dict)

        if hasattr(self, 'trace_model') and self.trace_model:
            if not config.data.predict.fixed_size:
                raise ValueError(
                    "Tracing requires that fixed_size mode be enabled in the config"
                )
        return config

    def _initialize_data_processor(self):
        is_multimer = "multimer" in self.config.data.model_preset  # Adjust as per actual config
        is_custom_template = getattr(self, "use_custom_template", False)

        if is_custom_template:
            template_featurizer = templates.CustomHitFeaturizer(
                mmcif_dir=self.config.template_mmcif_dir,
                max_template_date="9999-12-31",  # dummy value
                max_hits=-1,  # dummy value
                kalign_binary_path=self.config.kalign_binary_path,
            )
        elif is_multimer:
            template_featurizer = templates.HmmsearchHitFeaturizer(
                mmcif_dir=self.config.template_mmcif_dir,
                max_template_date=self.config.max_template_date,
                max_hits=self.config.data.predict.max_templates,
                kalign_binary_path=self.config.kalign_binary_path,
                release_dates_path=self.config.release_dates_path,
                obsolete_pdbs_path=self.config.obsolete_pdbs_path,
            )
        else:
            template_featurizer = templates.HhsearchHitFeaturizer(
                mmcif_dir=self.config.template_mmcif_dir,
                max_template_date=self.config.max_template_date,
                max_hits=self.config.data.predict.max_templates,
                kalign_binary_path=self.config.kalign_binary_path,
                release_dates_path=self.config.release_dates_path,
                obsolete_pdbs_path=self.config.obsolete_pdbs_path,
            )

        data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )

        if is_multimer:
            data_processor = data_pipeline.DataPipelineMultimer(
                monomer_data_pipeline=data_processor,
            )

        return data_processor

    def _setup_alignment_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.use_precomputed_alignments is None:
            alignment_dir = os.path.join(self.output_dir, "alignments")
        else:
            alignment_dir = self.use_precomputed_alignments
        return alignment_dir

    def _list_files_with_extensions(self, dir, extensions):
        return [f for f in os.listdir(dir) if f.endswith(extensions)]

    def _gather_sequences(self, fasta_contents):
        tag_list = []
        seq_list = []
        for fasta_content in fasta_contents:
            data = fasta_content  # Assuming fasta_contents are strings
            tags, seqs = parse_fasta(data)

            is_multimer = "multimer" in self.config.data.model_preset  # Adjust as per actual config
            if not is_multimer and len(tags) != 1:
                self.logger.info(
                    f"Input contains more than one sequence but "
                    f"multimer mode is not enabled. Skipping..."
                )
                continue

            tag = '-'.join(tags)
            tag_list.append((tag, tags))
            seq_list.append(seqs)

        # Sort targets by sequence length to optimize processing
        seq_sort_fn = lambda target: sum([len(s) for s in target[1]])
        sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)
        return sorted_targets

    def _load_models(self):
        if (
            "multimer" in self.config.data.model_preset
            and self.config.openfold_checkpoint_path
        ):
            raise ValueError(
                '`openfold_checkpoint_path` was specified, but no OpenFold checkpoints are available for multimer mode'
            )

        model_generator = load_models_from_command_line(
            self.config,
            self.model_device,
            self.config.openfold_checkpoint_path,
            self.config.jax_param_path,
            self.output_dir,
        )
        return model_generator

    def precompute_alignments(self, tags, seqs):
        for tag, seq in zip(tags, seqs):
            tmp_fasta_path = os.path.join(self.output_dir, f"tmp_{os.getpid()}.fasta")
            with open(tmp_fasta_path, "w") as fp:
                fp.write(f">{tag}\n{seq}")

            local_alignment_dir = os.path.join(self.alignment_dir, tag)

            if self.use_precomputed_alignments is None:
                self.logger.info(f"Generating alignments for {tag}...")
                os.makedirs(local_alignment_dir, exist_ok=True)

                is_multimer = "multimer" in self.config.data.model_preset
                if is_multimer:
                    template_searcher = hmmsearch.Hmmsearch(
                        binary_path=self.config.hmmsearch_binary_path,
                        hmmbuild_binary_path=self.config.hmmbuild_binary_path,
                        database_path=self.config.pdb_seqres_database_path,
                    )
                else:
                    template_searcher = hhsearch.HHSearch(
                        binary_path=self.config.hhsearch_binary_path,
                        databases=[self.config.pdb70_database_path],
                    )

                if self.config.use_single_seq_mode:
                    alignment_runner = data_pipeline.AlignmentRunner(
                        jackhmmer_binary_path=self.config.jackhmmer_binary_path,
                        uniref90_database_path=self.config.uniref90_database_path,
                        template_searcher=template_searcher,
                        no_cpus=self.config.cpus,
                    )
                    embedding_generator = EmbeddingGenerator()
                    embedding_generator.run(tmp_fasta_path, self.alignment_dir)
                else:
                    alignment_runner = data_pipeline.AlignmentRunner(
                        jackhmmer_binary_path=self.config.jackhmmer_binary_path,
                        hhblits_binary_path=self.config.hhblits_binary_path,
                        uniref90_database_path=self.config.uniref90_database_path,
                        mgnify_database_path=self.config.mgnify_database_path,
                        bfd_database_path=self.config.bfd_database_path,
                        uniref30_database_path=self.config.uniref30_database_path,
                        uniclust30_database_path=self.config.uniclust30_database_path,
                        uniprot_database_path=self.config.uniprot_database_path,
                        template_searcher=template_searcher,
                        use_small_bfd=self.config.bfd_database_path is None,
                        no_cpus=self.config.cpus,
                    )

                    alignment_runner.run(
                        tmp_fasta_path, local_alignment_dir
                    )
            else:
                self.logger.info(
                    f"Using precomputed alignments for {tag} at {self.alignment_dir}..."
                )

            os.remove(tmp_fasta_path)

    def generate_feature_dict(self, tags, seqs):
        tmp_fasta_path = os.path.join(self.output_dir, f"tmp_{os.getpid()}.fasta")

        is_multimer = "multimer" in self.config.data.model_preset

        if is_multimer:
            with open(tmp_fasta_path, "w") as fp:
                fp.write('\n'.join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
            feature_dict = self.data_processor.process_fasta(
                fasta_path=tmp_fasta_path, alignment_dir=self.alignment_dir,
            )
        elif len(seqs) == 1:
            tag = tags[0]
            seq = seqs[0]
            with open(tmp_fasta_path, "w") as fp:
                fp.write(f">{tag}\n{seq}")

            local_alignment_dir = os.path.join(self.alignment_dir, tag)
            feature_dict = self.data_processor.process_fasta(
                fasta_path=tmp_fasta_path,
                alignment_dir=local_alignment_dir,
                seqemb_mode=self.config.use_single_seq_mode,
            )
        else:
            with open(tmp_fasta_path, "w") as fp:
                fp.write('\n'.join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
            feature_dict = self.data_processor.process_multiseq_fasta(
                fasta_path=tmp_fasta_path, super_alignment_dir=self.alignment_dir,
            )

        os.remove(tmp_fasta_path)
        return feature_dict

    def round_up_seqlen(self, seqlen):
        return int(math.ceil(seqlen / self.TRACING_INTERVAL)) * self.TRACING_INTERVAL

    def predict_structure(self, fasta_contents):
        """
        Predict protein structures from FASTA contents.

        Args:
            fasta_contents (List[str]): List of FASTA formatted strings.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: List of tuples containing coordinates, confidence, and padding masks.
        """
        sorted_targets = self._gather_sequences(fasta_contents)

        structures = []

        for model, output_directory in self.model_generator:
            cur_tracing_interval = 0
            for (tag, tags), seqs in sorted_targets:
                output_name = f'{tag}_{self.config.data.model_preset}'
                if self.config.output_postfix:
                    output_name = f'{output_name}_{self.config.output_postfix}'

                # Precompute or load alignments
                self.precompute_alignments(tags, seqs)

                # Generate or retrieve feature dict
                feature_dict = self.feature_dicts.get(tag, None)
                if feature_dict is None:
                    feature_dict = self.generate_feature_dict(tags, seqs)

                    if self.config.trace_model:
                        n = feature_dict["aatype"].shape[-2]
                        rounded_seqlen = self.round_up_seqlen(n)
                        feature_dict = pad_feature_dict_seq(
                            feature_dict, rounded_seqlen,
                        )

                    self.feature_dicts[tag] = feature_dict

                # Process features
                processed_feature_dict = self.feature_processor.process_features(
                    feature_dict, mode='predict', is_multimer=("multimer" in self.config.data.model_preset)
                )

                processed_feature_dict = {
                    k: torch.as_tensor(v, device=self.model_device)
                    for k, v in processed_feature_dict.items()
                }

                # Trace model if enabled
                if self.config.trace_model:
                    rounded_seqlen = self.round_up_seqlen(processed_feature_dict["aatype"].shape[-2])
                    if rounded_seqlen > cur_tracing_interval:
                        self.logger.info(
                            f"Tracing model at {rounded_seqlen} residues..."
                        )
                        t = time.perf_counter()
                        trace_model_(model, processed_feature_dict)
                        tracing_time = time.perf_counter() - t
                        self.logger.info(f"Tracing time: {tracing_time}")
                        cur_tracing_interval = rounded_seqlen

                # Run the model
                out = run_model(model, processed_feature_dict, tag, self.output_dir)

                # Post-process outputs
                processed_feature_dict = tensor_tree_map(
                    lambda x: np.array(x[..., -1].cpu()),
                    processed_feature_dict
                )
                out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

                unrelaxed_protein = prep_output(
                    out,
                    processed_feature_dict,
                    feature_dict,
                    self.feature_processor,
                    self.config.data.model_preset,
                    self.config.multimer_ri_gap,
                    self.config.subtract_plddt
                )

                unrelaxed_file_suffix = "_unrelaxed.cif" if self.config.cif_output else "_unrelaxed.pdb"
                unrelaxed_output_path = os.path.join(
                    output_directory, f'{output_name}{unrelaxed_file_suffix}'
                )

                with open(unrelaxed_output_path, 'w') as fp:
                    if self.config.cif_output:
                        fp.write(protein.to_modelcif(unrelaxed_protein))
                    else:
                        fp.write(protein.to_pdb(unrelaxed_protein))

                self.logger.info(f"Output written to {unrelaxed_output_path}...")

                # Collect the structure data
                coords, confidence, padding_mask = self._extract_structure(unrelaxed_protein)
                structures.append((coords, confidence, padding_mask))

                # Relax the prediction if not skipped
                if not self.config.skip_relaxation:
                    self.logger.info(f"Running relaxation on {unrelaxed_output_path}...")
                    relax_protein(
                        self.config,
                        self.model_device,
                        unrelaxed_protein,
                        output_directory,
                        output_name,
                        self.config.cif_output
                    )

                # Save model outputs if required
                if self.config.save_outputs:
                    output_dict_path = os.path.join(
                        output_directory, f'{output_name}_output_dict.pkl'
                    )
                    with open(output_dict_path, "wb") as fp:
                        pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

                    self.logger.info(f"Model output written to {output_dict_path}...")

        return structures

    def _extract_structure(self, unrelaxed_protein):
        """
        Extract coordinates, confidence, and padding_mask from unrelaxed_protein.

        Args:
            unrelaxed_protein (Protein): Unrelaxed protein structure.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Coordinates, confidence, padding_mask.
        """
        # Assuming unrelaxed_protein has attributes or methods to get these
        # You need to adjust these based on OpenFold's Protein class
        coords = torch.tensor(unrelaxed_protein.atom_positions)  # Adjust as per actual attribute
        confidence = torch.tensor(unrelaxed_protein.confidence_scores)  # Adjust as per actual attribute
        padding_mask = torch.tensor(unrelaxed_protein.padding_mask)  # Adjust as per actual attribute

        return coords, confidence, padding_mask
