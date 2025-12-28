import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .base import BaseSSLModule
from .utils import ProjectionMLP, PredictionMLP


class SwAVModule(BaseSSLModule):
    def __init__(
        self,
        n_prototypes: int = 3000,
        temperature: float = 0.1,
        n_local_crops: int = 6,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters("n_prototypes", "temperature", "n_local_crops")

        self.projection = ProjectionMLP(self.feat_dim, out_dim=128)
        # Prototypes: learnable cluster centers
        self.prototypes = nn.Linear(128, n_prototypes, bias=False)

    @torch.no_grad()
    def _sinkhorn_knopp(self, scores: torch.Tensor) -> torch.Tensor:
        """Sinkhorn-Knopp algorithm to compute optimal transport."""
        Q = torch.exp(scores / self.hparams.sinkhorn_eps).T
        Q /= torch.sum(Q)
        K, B = Q.shape
        u = torch.zeros(K, device=self.device)
        r = torch.ones(K, device=self.device) / K
        c = torch.ones(B, device=self.device) / B
        for _ in range(self.hparams.sinkhorn_iters):
            u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).T

    def _swapped_prediction_loss(
        self, scores: List[torch.Tensor], assignments: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes the swapped prediction loss for SwAV.
        'scores' contains embeddings from all crops.
        'assignments' contains cluster assignments from global crops only.
        """
        loss = 0.0
        n_terms = 0

        # 全てのビュー（グローバル＋ローカル）をループ
        for i, score in enumerate(scores):
            # ターゲットとなるグローバルビューの割り当てをループ
            for j, assignment in enumerate(assignments):
                # グローバルビューが自分自身の割り当てを予測するのをスキップ
                # (最初のlen(assignments)個のscoreがグローバルビューのものと仮定)
                if i == j:
                    continue

                # score i を使って assignment j を予測する際のクロスエントロピー損失
                l = -torch.mean(
                    torch.sum(
                        assignment
                        * F.log_softmax(score / self.hparams.temperature, dim=1),
                        dim=1,
                    )
                )
                loss += l
                n_terms += 1

        # 損失項の数で正規化
        return loss / n_terms if n_terms > 0 else loss

    def training_step(self, batch, batch_idx):
        # Batch is a list of crops from SwAVTransform
        crops, _ = batch

        # Pass all crops through encoder and projection head
        # The first 2 are global crops, the rest are local crops
        projections = [self.projection(self.encoder(crop)) for crop in crops]

        # Compute prototype scores for each view
        # Normalize projections and prototypes
        projections = [F.normalize(p, dim=1) for p in projections]
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        scores = [self.prototypes(p) for p in projections]

        # Compute assignments (codes) using Sinkhorn-Knopp
        # Only use global views (first two) for computing assignments to avoid instability from small crops
        with torch.no_grad():
            assignments = [self._sinkhorn_knopp(s) for s in scores[:2]]

        # Compute swapped prediction loss
        loss = self._swapped_prediction_loss(scores, assignments)

        self.log_dict(
            {
                "train_loss": loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "temperature": self.hparams.temperature,
            }
        )
        return loss
