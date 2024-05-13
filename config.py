TEST_PHN_PATH = 'timit/data/TRAIN/DR4/MGAG0/SX61.PHN'
TEST_WAV_PATH = 'timit/data/TRAIN/DR4/MGAG0/SX61.WAV'
PHONEME_LIST = [
 ('#b', 'n'),
 ('h#', 'n'),
 ('n',),
 ('n', 'eh'),
 ('eh',),
 ('eh', 'v'),
 ('v',),
 ('v', 'axr'),
 ('axr',),
 ('axr', 'h#'),
 ('axr', '#b')]

# Position of phonemes in the label vector. 
phoneme_to_idx = {'#bn':0, 'h#n', 'n':1, 'neh':2, 'eh':3, 'ehv':4, 
                  'v':5, 'vaxr':6, 'axr':7, 'axr#b':8, 'axrh#', '#b':9, 'h#': 10}


