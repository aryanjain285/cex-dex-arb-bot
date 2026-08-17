from loguru import logger

from ..core.types import Opportunity

class SimpleRouter:
    def __init__(self):
        logger.info("Simple router initialised.")

    def plan(self, opp: Opportunity) -> Opportunity:
        """
        The router's role is intentionally minimal in the current architecture.
        It passes the Opportunity straight through, and exists as a seam for
        additional filtering or enrichment.
        """
        # TODO: add richer routing logic here, e.g. selecting between DEXes or CEXes.
        return opp
