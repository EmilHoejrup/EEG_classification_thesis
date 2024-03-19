from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

DATA_DIR = ROOT / 'data'

CONFIG_FILE = ROOT / 'scripts/' / 'support' / 'configs.yml'

BNCI_CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz',]
BNCI_SFREQ = 250.0
