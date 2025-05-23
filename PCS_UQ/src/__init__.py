# Make the PCS class directly available at the top level
from .PCS.regression.pcs_uq import PCS_UQ as PCS

__all__ = ['PCS']
