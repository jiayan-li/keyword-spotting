import numpy as np
import pandas as pd
import keras_tuner as kt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adam
from kerastuner import HyperParameters
from sklearn.model_selection import train_test_split
import datetime

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from mfcc_label import prepare_data
from config import *
from joblib import dump, load

#from DNN_utils
def column_str_to_numpy(df_pre, colname:str):
    # Given pd.DataFrame df_pre, convert the column colname from string to numpy array.
    if isinstance(df_pre.iloc[0][colname], str):
        df_pre[colname]=df_pre[colname].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

def upsample_multiple_minority(df_pre: pd.DataFrame):

    # Assuming df_pre is your dataframe
    # Identify majority and minority classes
    majority_classes = [13, 14]
    minority_classes = list(range(13))

    # Find the size of the smallest majority class
    majority_class_size = df_pre['single_class_label'].value_counts().nlargest(2).iloc[1]
    
    # List to hold the upsampled dataframes
    list_df_pre = []

    # Loop through each minority class and upsample
    for class_value in minority_classes:
        df_pre_minority_class = df_pre[df_pre['single_class_label'] == class_value]
        df_pre_minority_upsampled = resample(df_pre_minority_class, 
                                        replace=True,     # sample with replacement
                                        n_samples=majority_class_size,    # to match majority class size
                                        random_state=123) # reproducible results
        list_df_pre.append(df_pre_minority_upsampled)

    # Append majority classes without change
    df_pre_majority = df_pre[df_pre['single_class_label'].isin(majority_classes)]
    list_df_pre.append(df_pre_majority)

    # Concatenate all dataframes
    df_pre_upsampled = pd.concat(list_df_pre)
    return df_pre_upsampled


def reshape_lstm(input: np.ndarray):
    """
    Reshape the input data for LSTM.
    """

    # If the shape is not (num_samples, timesteps, num_features), reshape it
    if len(input.shape) == 2:  # Assuming shape is (num_samples, num_features)
        input = input.reshape((input.shape[0], 1, input.shape[1]))

    return input

def build_model(hp):
    """
    builds a model with hyperparameters.
    """

    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units_lstm1', min_value=64, max_value=128, step=32),
        input_shape=(1, 20),
        return_sequences=True
    ))
    model.add(BatchNormalization())
    model.add(LSTM(
        units=hp.Int('units_lstm2', min_value=64, max_value=128, step=32)
    ))

    # Add optional linear layers
    for i in range(hp.Int('num_linear_layers', 0, 3)):
        model.add(Dense(
            units=hp.Int(f'units_linear_{i}', min_value=64, max_value=128, step=32),
            activation='relu'
        ))

    model.add(Dense(14, activation='softmax'))

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define a function to get the batch size
def get_batch_size(hp):
    return hp.Int('batch_size', min_value=32, max_value=256, step=32)


def preprocess_data(train: bool = True):
    """
    Preprocess the training data.
    """

    if train:
        file_path: int = 'processed_data/dnn_never_train.csv'
    else:
        file_path: int = 'processed_data/dnn_never_test.csv'

    df_pre = pd.read_csv(file_path)
    column_str_to_numpy(df_pre, 'mfcc')
    column_str_to_numpy(df_pre, 'label')
    
    if train:
        df_pre['single_class_label'] = df_pre['label'].apply(lambda x: np.argmax(x))
        df_pre = upsample_multiple_minority(df_pre)
    else: 
        pass

    df_pre.reset_index(drop=True, inplace=True)

    X = np.array(df_pre['mfcc'].tolist())
    y = np.array(df_pre['label'].tolist())

    X = reshape_lstm(X)

    return X, y


def tune_model():

    X, y = preprocess_data(train=True)

    hp = HyperParameters()
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='models',
        project_name='lstm_tuning',
        overwrite=True
    )

    # Get the batch size from the hyperparameters
    batch_size = get_batch_size(hp)
    
    # Perform the search
    tuner.search(X, y, epochs=10, validation_split=0.2, batch_size=batch_size)

    #tuner.search(X, y, epochs=10, validation_split=0.2)

    # Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Get the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hyperparameters.values}")

    return best_model, best_hyperparameters


def final_model(X, y):
    """
    final model with the best hyperparameters.
    """

    model = Sequential()
    model.add(LSTM(
        units=128,
        input_shape=(1, 20),
        return_sequences=True
    ))
    model.add(BatchNormalization())
    model.add(LSTM(
        units=64
    ))

    # Add optional linear layers
    for i in range(3):
        model.add(Dense(
            units=64,
            activation='relu'
        ))

    model.add(Dense(14, activation='softmax'))

    # learning_rate = 0.0089397
    # optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def path_to_emission(file_path_phn: str, file_path_wav: str, model) -> np.ndarray:
    '''
    Given the path of a file, get the emission probabilities.
    Args:
        file_path: Path of the audio file as a string.
    Returns:
        emit: pd.dataframe
            Emission probabilities for each frame in the audio file.
    '''
    df_test = prepare_data(file_path_phn,file_path_wav)
    column_str_to_numpy(df_test, 'mfcc')
    column_str_to_numpy(df_test, 'label')

    X_test = np.array(df_test['mfcc'].tolist())
    # y_test = np.array(df_test['label'].tolist())
    X_test = reshape_lstm(X_test)
 
    emission_data = model.predict(X_test)

    return emission_data


def get_emission_all_paths(model, path_type: str = 'test'):
    paths = load('processed_data/train_test_dataset_never.joblib')[path_type]
    data = {}
    for i in range(len(paths)):
        file_path_wav, file_path_phn, file_path_word = paths[i]
        emission_data = path_to_emission(file_path_phn, file_path_wav, model)
        emission_df = pd.DataFrame({'label': list(emission_data)})
        data[(file_path_wav, file_path_phn, file_path_word)] = emission_df

    return data

def main() -> None:
    """
    Main function to train the model after finding the best parameters.
    """

    # Tune the model using training data
    X, y = preprocess_data(train = True)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = final_model(X_train, y_train)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback, early_stopping_callback],
        verbose=1
    )

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_accuracy}')

    # evaluate the model on the test set
    X_test, y_test = preprocess_data(train = False)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

    # Save the model
    model.save('models/lstm_model.keras')

    # Get the emission probabilities for all paths
    data = get_emission_all_paths(model) 
    dump(data, "processed_data/test_data_for_hmm_keras.joblib")

