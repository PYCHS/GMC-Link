"""
Loss functions for the GMC-Link alignment network.
"""
from infonce import SupervisedInfoNCE

class AlignmentLoss(SupervisedInfoNCE):
    """
    InfoNCE loss for motion-language alignment.

    This loss encourages the model to assign high similarity scores to correct motion-language pairs and low scores to incorrect pairs. It is a binary classification loss where:

    - Positive pairs (correct matches) should have scores close to 1.0.
    - Negative pairs (incorrect matches) should have scores close to 0.0.   

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
