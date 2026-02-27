from __future__ import annotations

def register_csv_dataset(
    dataset_name: str,
    nickname: str,
    require_team_role: bool,
    default_team: str = "left",
    default_role: str = "player",
) -> None:
    """Register CSVReIDDataset to PRTReid if name not already registered."""
    from prtreid.data.datasets import get_image_dataset, register_image_dataset
    from .custom_dataset import ConfiguredCSVReIDDataset, configure_runtime_dataset

    try:
        get_image_dataset(dataset_name)
        return
    except ValueError:
        pass

    configure_runtime_dataset(
        dataset_name=dataset_name,
        require_team_role=require_team_role,
        default_team=default_team,
        default_role=default_role,
    )
    register_image_dataset(dataset_name, ConfiguredCSVReIDDataset, nickname)


def apply_reid_only_patch() -> None:
    """
    Patch ImagePartBasedEngine so `reid_only` mode optimizes only ReID objective.

    Team/role heads remain in the model for compatibility, but their losses are removed.
    """
    from torch import nn
    from prtreid.engine.image.part_based_engine import ImagePartBasedEngine

    if getattr(ImagePartBasedEngine, "_reid_only_patch_applied", False):
        return

    def combine_losses_reid_only(
        self,
        visibility_scores_dict,
        embeddings_dict,
        id_cls_scores_dict,
        team_cls_scores_dict,
        role_cls_scores_dict,
        pids,
        teams,
        roles,
        pixels_cls_scores=None,
        target_masks=None,
        bpa_weight=0,
    ):
        reid_loss, reid_loss_summary = self.GiLt(
            embeddings_dict,
            visibility_scores_dict,
            id_cls_scores_dict,
            pids,
        )
        loss = reid_loss
        loss_summary = reid_loss_summary

        if pixels_cls_scores is not None and target_masks is not None and bpa_weight > 0:
            target_masks = nn.functional.interpolate(
                target_masks,
                pixels_cls_scores.shape[2::],
                mode="bilinear",
                align_corners=True,
            )
            pixels_cls_score_targets = target_masks.argmax(dim=1)
            bpa_loss, bpa_loss_summary = self.body_part_attention_loss(
                pixels_cls_scores,
                pixels_cls_score_targets,
            )
            loss += bpa_weight * bpa_loss
            loss_summary = {**loss_summary, **bpa_loss_summary}

        return loss, loss_summary

    ImagePartBasedEngine.combine_losses = combine_losses_reid_only
    ImagePartBasedEngine._reid_only_patch_applied = True


def apply_generic_random_identity_sampler_patch() -> None:
    """
    Replace TrackLab-customized RandomIdentitySampler iteration with a generic
    ReID sampler that does not assume per-game PID availability constraints.
    """
    import copy
    import random
    from collections import defaultdict

    import numpy as np
    from prtreid.data.sampler import RandomIdentitySampler

    if getattr(RandomIdentitySampler, "_generic_patch_applied", False):
        return

    def _iter_generic(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = [pid for pid in self.pids if len(batch_idxs_dict[pid]) > 0]
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    RandomIdentitySampler.__iter__ = _iter_generic
    RandomIdentitySampler._generic_patch_applied = True


def apply_generic_prtreid_sampler_patch() -> None:
    """
    Relax PrtreidSampler to a generic identity sampler for custom datasets.
    This keeps multitask losses active while removing SoccerNet-specific
    assumptions about fixed left/right/other composition per batch.
    """
    import copy
    import random
    from collections import defaultdict

    import numpy as np
    from prtreid.data.sampler import PrtreidSampler

    if getattr(PrtreidSampler, "_generic_patch_applied", False):
        return

    def _iter_generic(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = [pid for pid in self.pids if len(batch_idxs_dict[pid]) > 0]
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    PrtreidSampler.__iter__ = _iter_generic
    PrtreidSampler._generic_patch_applied = True


def apply_ascii_writer_patch() -> None:
    """
    Avoid UnicodeEncodeError on Windows cp1252 console when PRTReid prints
    final summary table using tabulate(tablefmt='fancy_grid').
    """
    import prtreid.utils.writer as writer_mod

    if getattr(writer_mod, "_ascii_patch_applied", False):
        return

    original_tabulate = writer_mod.tabulate

    def _tabulate_ascii(*args, **kwargs):
        if kwargs.get("tablefmt") == "fancy_grid":
            kwargs["tablefmt"] = "grid"
        return original_tabulate(*args, **kwargs)

    writer_mod.tabulate = _tabulate_ascii
    writer_mod._ascii_patch_applied = True


def apply_triplet_none_guard_patch() -> None:
    """
    Guard GiLt triplet branch when a batch has no valid triplets.
    Some samplers/datasets can produce this case temporarily.
    """
    import torch
    from prtreid.losses.GiLt_loss import GiLtLoss

    if getattr(GiLtLoss, "_triplet_none_guard_applied", False):
        return

    def _safe_compute_triplet_loss(self, embeddings, visibility_scores, pids):
        if self.use_visibility_scores:
            visibility = visibility_scores if len(visibility_scores.shape) == 2 else visibility_scores.unsqueeze(1)
        else:
            visibility = None
        embeddings = embeddings if len(embeddings.shape) == 3 else embeddings.unsqueeze(1)

        result = self.part_triplet_loss(embeddings, pids, parts_visibility=visibility)
        if result is None:
            zero = torch.tensor(0.0, device=embeddings.device)
            return zero, zero, zero

        try:
            triplet_loss, trivial_triplets_ratio, valid_triplets_ratio = result
        except Exception:
            zero = torch.tensor(0.0, device=embeddings.device)
            return zero, zero, zero

        if triplet_loss is None:
            zero = torch.tensor(0.0, device=embeddings.device)
            triplet_loss = zero
        if trivial_triplets_ratio is None:
            trivial_triplets_ratio = torch.tensor(0.0, device=embeddings.device)
        if valid_triplets_ratio is None:
            valid_triplets_ratio = torch.tensor(0.0, device=embeddings.device)

        return triplet_loss, trivial_triplets_ratio, valid_triplets_ratio

    GiLtLoss.compute_triplet_loss = _safe_compute_triplet_loss
    GiLtLoss._triplet_none_guard_applied = True
