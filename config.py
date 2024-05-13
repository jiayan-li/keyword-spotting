TEST_PHN_PATH = 'timit/data/TRAIN/DR4/MGAG0/SX61.PHN'
TEST_WAV_PATH = 'timit/data/TRAIN/DR4/MGAG0/SX61.WAV'

PHONEME_LIST = [
    'b-n', 'm-n', 'e-n',
    'b-eh', 'm-eh', 'e-eh',
    'b-v', 'm-v', 'e-v',
    'b-axr', 'm-axr', 'e-axr',
    'h#', '#b' 
]

PHONEME_DICT = {
    phoneme: idx for idx, phoneme in enumerate(PHONEME_LIST)}