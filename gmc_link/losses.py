"""
Loss functions for the GMC-Link alignment network.
"""
from infonce import SupervisedInfoNCE

class AlignmentLoss(SupervisedInfoNCE):
    """
    Supervised InfoNCE loss for motion-language alignment.

    This contrastive loss pulls together embeddings of matching motion-language
    pairs (same expression ID) and pushes apart non-matching pairs in the shared
    feature space. Negatives are formed in-batch automatically.

    Args passed to forward():
        features: (2N, D) concatenated motion and text embeddings.
        target:   (2N,) integer class labels — matching pairs share the same ID.
    """

    def __init__(self):
        super().__init__(temperature=0.07)

    def forward(self, features, target):
        """
        Args:
            features: (N, D) normalized embedding vectors for all
                      motion and text samples in the batch.
            target:   (N,) integer labels — matching pairs share the
                      same label.
        Returns:
            Scalar supervised InfoNCE loss.
        """
        return super().forward(features, target)
