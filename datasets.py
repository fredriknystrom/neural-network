import numpy as np

train_data = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])

target_xor = np.array(
    [
        [0],
        [1],
        [1],
        [0]])

target_nand = np.array(
    [
        [1],
        [1],
        [1],
        [0]])

target_or = np.array(
    [
        [0],
        [1],
        [1],
        [1]])

target_and = np.array(
    [
        [0],
        [0],
        [0],
        [1]])

def or_dataset():
    return train_data, target_or

def xor_dataset():
    return train_data, target_xor

def and_dataset():
    return train_data, target_and

def nand_dataset():
    return train_data, target_nand


