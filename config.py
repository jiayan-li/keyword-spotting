TEST_PHN_PATH = 'timit/data/TRAIN/DR2/FMMH0/SI907.PHN'
TEST_WAV_PATH = 'timit/data/TRAIN/DR2/FMMH0/SI907.WAV'
TEST_WRD_PATH = 'timit/data/TRAIN/DR2/FMMH0/SI907.WRD'
# 'Somehow we old-timers never figured we would ever retire.'

PHONEME_LIST = [
    'b-n', 'm-n', 'e-n',
    'b-eh', 'm-eh', 'e-eh',
    'b-v', 'm-v', 'e-v',
    'b-axr', 'm-axr', 'e-axr',
    'h#', '#b' 
]

PHONEME_DICT = {
    phoneme: idx for idx, phoneme in enumerate(PHONEME_LIST)}