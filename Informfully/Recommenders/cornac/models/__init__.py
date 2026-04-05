# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from .recommender import Recommender
from .recommender import NextBasketRecommender
from .recommender import NextItemRecommender

from .amr import AMR
from .ann import AnnoyANN
from .ann import FaissANN
from .ann import HNSWLibANN
from .ann import ScaNNANN
try:
    from .baseline_only import BaselineOnly
except (ImportError, ModuleNotFoundError):
    BaselineOnly = None
from .beacon import Beacon
from .bivaecf import BiVAECF
try:
    from .bpr import BPR, WBPR
except (ImportError, ModuleNotFoundError):
    BPR = WBPR = None
from .causalrec import CausalRec
try:
    from .c2pf import C2PF
except (ImportError, ModuleNotFoundError, AttributeError):
    C2PF = None
try:
    from .cdl import CDL
except (ImportError, ModuleNotFoundError):
    CDL = None
try:
    from .cdr import CDR
except (ImportError, ModuleNotFoundError):
    CDR = None
from .coe import COE
try:
    from .companion import Companion
except (ImportError, ModuleNotFoundError):
    Companion = None
try:
    from .comparer import ComparERObj, ComparERSub
except (ImportError, ModuleNotFoundError):
    ComparERObj = ComparERSub = None
try:
    from .conv_mf import ConvMF
except (ImportError, ModuleNotFoundError):
    ConvMF = None
from .ctr import CTR
try:
    from .cvae import CVAE
except (ImportError, ModuleNotFoundError):
    CVAE = None
try:
    from .cvaecf import CVAECF
except (ImportError, ModuleNotFoundError):
    CVAECF = None
from .dmrl import DMRL
from .dnntsp import DNNTSP
from .ease import EASE
try:
    from .efm import EFM
except (ImportError, ModuleNotFoundError):
    EFM = None
from .fm import FM
from .gcmc import GCMC
from .global_avg import GlobalAvg
from .gp_top import GPTop
from .gru4rec import GRU4Rec
from .hft import HFT
try:
    from .hpf import HPF
except (ImportError, ModuleNotFoundError):
    HPF = None
try:
    from .hrdr import HRDR
except (ImportError, ModuleNotFoundError):
    HRDR = None
from .hypar import HypAR
from .ibpr import IBPR
try:
    from .knn import ItemKNN, UserKNN
except (ImportError, ModuleNotFoundError):
    ItemKNN = UserKNN = None
from .lightgcn import LightGCN
try:
    from .lrppm import LRPPM
except (ImportError, ModuleNotFoundError):
    LRPPM = None
try:
    from .mcf import MCF
except (ImportError, ModuleNotFoundError):
    MCF = None
try:
    from .mf import MF
except (ImportError, ModuleNotFoundError):
    MF = None
try:
    from .mmmf import MMMF
except (ImportError, ModuleNotFoundError):
    MMMF = None
from .most_pop import MostPop
try:
    from .mter import MTER
except (ImportError, ModuleNotFoundError):
    MTER = None
try:
    from .narre import NARRE
except (ImportError, ModuleNotFoundError):
    NARRE = None
try:
    from .ncf import GMF, MLP, NeuMF
except (ImportError, ModuleNotFoundError):
    GMF = MLP = NeuMF = None
from .ngcf import NGCF
try:
    from .nmf import NMF
except (ImportError, ModuleNotFoundError):
    NMF = None
from .online_ibpr import OnlineIBPR
try:
    from .pcrl import PCRL
except (ImportError, ModuleNotFoundError):
    PCRL = None
try:
    from .pmf import PMF
except (ImportError, ModuleNotFoundError):
    PMF = None
from .recvae import RecVAE
from .sansa import SANSA
try:
    from .sbpr import SBPR
except (ImportError, ModuleNotFoundError):
    SBPR = None
from .skm import SKMeans
try:
    from .sorec import SoRec
except (ImportError, ModuleNotFoundError):
    SoRec = None
from .spop import SPop
from .svd import SVD
from .tifuknn import TIFUKNN
from .trirank import TriRank
from .upcf import UPCF
from .vaecf import VAECF
from .vbpr import VBPR
from .vmf import VMF
try:
    from .wmf import WMF
except (ImportError, ModuleNotFoundError):
    WMF = None
from .dae import DAE
try:
    from .enmf import ENMF
except (ImportError, ModuleNotFoundError):
    ENMF = None
from .drdw import D_RDW
try:
    from .nrms import NRMS
except (ImportError, ModuleNotFoundError):
    NRMS = None
try:
    from .lstur import LSTUR
except (ImportError, ModuleNotFoundError):
    LSTUR = None
try:
    from .npa import NPA
except (ImportError, ModuleNotFoundError):
    NPA = None
try:
    from .rp3_beta import RP3_Beta
except (ImportError, ModuleNotFoundError):
    RP3_Beta = None
from .rwe_d import RWE_D
from .random import RandomModel
from .pld import PLD
from .epd import EPD
try:
    from .beacon import Beacon
except (ImportError, ModuleNotFoundError):
    Beacon = None
