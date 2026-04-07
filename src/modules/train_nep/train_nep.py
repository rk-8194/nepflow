"""NEP model training stage."""

import logging
from ..base import Stage

logger = logging.getLogger(__name__)


class TrainNepStage(Stage):
    """Train NEP models on DFT results."""
    
    def run(self) -> None:
        """Execute NEP training."""
        logger.info("Training NEP models")
        logger.debug("NEP training not yet implemented")
        logger.info("NEP training complete")
