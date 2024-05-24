# Keyword Spotting with Deep Neural Network and Hidden Markov Model

**Authors**: Jiayan Li and Emin Ozyoruk

## Abstract
This repository contains the implementation and details of a keyword spotting (KWS) system designed to detect the keyword "never" in audio files. The system aims to operate efficiently on low-power processors while maintaining high precision and a small memory footprint. It combines a Deep Neural Network (DNN) for feature extraction and a Hidden Markov Model (HMM) for keyword detection.

## Introduction
Keyword Spotting (KWS) is a critical technology in speech recognition, enabling devices to respond to specific voice commands without manual activation. This system is particularly useful for hands-free interaction with devices, which is essential in scenarios like driving, cooking, or emergency situations. The challenge is to achieve high accuracy and reliability with low computational and power overheads.

## Data and Methods
### Data
The system uses the TIMIT Acoustic-Phonetic Continuous Speech Corpus, consisting of recordings from 630 speakers. The keyword "never" appears in 57 out of 6300 audio files. We used 28 audio files for training and 29 for testing. Features were extracted using the `librosa` library, focusing on Mel-frequency cepstral coefficients (MFCCs).

### Methods
The system uses a DNN as an encoder to process MFCC vectors, outputting probabilities for different phoneme states. An HMM then uses these probabilities to detect the keyword using the Viterbi algorithm. Two types of DNNs were implemented:
- **Feed-forward Neural Network (FF-NN)**: Consisting of 4 linear layers.
- **Long Short-Term Memory Network (LSTM)**: Consisting of 2 LSTM layers followed by 3 linear layers with ReLU activation.

## Experimental Results and Discussion
We evaluated the performance of the DNN encoders and the overall KWS system. The LSTM achieved an accuracy of 61.78% on the test set, while the FF-NN achieved 74.28%. The entire KWS system's performance, evaluated with different sets of emission probabilities, showed promising results, especially in terms of recall.

## Conclusion
Our KWS system effectively detects the keyword "never" with a small training size, moderate accuracy, and high efficiency. The combination of DNN for feature extraction and HMM for keyword detection proves successful, demonstrating potential for deployment on low-power devices with limited training data.

## References
1. Gales, M., & Young, S. (2008). The application of hidden Markov models in speech recognition. Foundations and Trends® in Signal Processing, 1(3), 195-304.
2. Rose, R. C., & Paul, D. B. (1990). A hidden Markov model based keyword recognition system. In International conference on acoustics, speech, and signal processing (pp. 129-132). IEEE.
3. Wilpon, J. G., et al. (1990). Automatic recognition of keywords in unconstrained speech using hidden Markov models. IEEE Transactions on Acoustics, Speech, and Signal Processing, 38(11), 1870-1878.
4. Graves, A., et al. (2013). Speech recognition with deep recurrent neural networks. In 2013 IEEE international conference on acoustics, speech and signal processing (pp. 6645-6649). IEEE.
5. Chen, G., et al. (2014). Small-footprint keyword spotting using deep neural networks. In 2014 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 4087-4091). IEEE.
6. Sigtia, S., et al. (2018). Efficient voice trigger detection for low resource hardware. In Interspeech (pp. 2092-2096).
7. Shrivastava, A., et al. (2021). Optimize what matters: Training DNN-HMM keyword spotting model using end metric. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 4000-4004). IEEE.
