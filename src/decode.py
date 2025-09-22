import sys
import os
import warnings
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import struct
from os.path import exists


def word_is_feature_one(word) -> bool:
    return (word[0] & 0b11000000) == 0x40


def word_is_feature_two(word) -> bool:
    return (word[0] & 0b11000000) == 0x80


def parse_feature_one(word) -> dict:
    features = {}
    features["ADC"] = (word[0] & 0b00110000) >> 4
    features["TSTMP"] = word[5] | word[4] << 8 | word[3] << 16 | word[2] << 24
    INT = (
        word[13]
        | word[12] << 8
        | word[11] << 16
        | word[10] << 24
        | word[9] << 32
        | word[8] << 40
        | word[7] << 48
        | word[6] << 56
    )
    sign_INT = INT & 0x8000000000000000
    value_INT = INT & 0x7FFFFFFFFFFFFFFF
    if sign_INT:
        value_INT = INT - 0xFFFFFFFFFFFFFFFF
    # print(f"{value_INT} {value_INT:016x} {~np.uint64(value_INT):016x} {INT-0xFFFFFFFFFFFFFFFF}")

    features["INT"] = np.int64(value_INT)
    features["PEAK"] = word[14] << 8 | word[15]
    return features


def parse_feature_two(word) -> dict:
    features = {}
    features["ADC"] = (word[0] & 0b00110000) >> 4
    features["TOT"] = word[15] | word[14] << 8 | word[13] << 16 | word[12] << 24
    return features


def parse_waveform(word):
    WAVE = []
    for i in [6, 9, 12, 15]:
        lower_four = (word[i - 1] & 0xF0) >> 4
        upper_eight = word[i - 2] << 4
        signed_bit = upper_eight & 0x800
        upper_eight_wo_signed = upper_eight & 0x7FF
        combined = upper_eight_wo_signed + lower_four
        if signed_bit:
            combined = -1 * ((~combined) & 0x7FF)
        WAVE.append(combined)
        upper_four = (word[i - 1] & 0x0F) << 8
        lower_eight = word[i]
        signed_bit = upper_four & 0x800
        upper_four_wo_signed = signed_bit & 0x7FF
        combined = upper_four + lower_eight
        if signed_bit:
            combined = -1 * ((~combined) & 0x7FF)
        WAVE.append(combined)
    return WAVE


def binar_to_words(bin_dat, len_words):
    words = []
    for i in range(len_words):
        # print(bin_dat[16*i:16*(i+1)])
        words.append(bytes(bin_dat[16 * i : 16 * (i + 1)]))
    return words


def decode_words(words):
    for i, w in enumerate(words):
        new_word = [byte for byte in w]
        new_word.reverse()
        # new_word = [new_word[i+1] if i % 2 == 0 else new_word[i-1] for i in range(len(new_word))]
        words[i] = new_word
    return words


def create_empty_packets(words):
    event_counter = 0
    for word in words:
        if word_is_feature_one(word):
            event_counter += 1
    # print(f"Found a total of {event_counter} Events in the data")
    if event_counter == 0:
        warnings.warn("No Feature Words received!")

    packets = {}
    for i in range(event_counter + 1):
        packets[f"Event_{i}"] = {}
    return packets


def packetize_events_feature_first():
    packets = create_empty_packets(words)
    current_event = -1
    start = 0
    # print(f"Starting with processing events. Current Event : {current_event}")
    for i, word in enumerate(words):
        if word_is_feature_one(word):
            current_event += 1
            start = 1
            # print(f"Current Event : {current_event}")
            if f"Event_{current_event}" in packets.keys():
                packets[f"Event_{current_event}"]["Data"] = []
            packets[f"Event_{current_event}"]["Features1"] = parse_feature_one(word)
        elif word_is_feature_two(word):
            packets[f"Event_{current_event}"]["Features2"] = parse_feature_two(word)
        elif start:
            packets[f"Event_{current_event}"]["Data"] += parse_waveform(word)
        if start:
            print(
                [f"0x{w:02x}" for w in word],
                word_is_feature_one(word),
                current_event,
                packets[f"Event_{current_event}"]["Features1"]["INT"],
                packets[f"Event_{current_event}"]["Features1"]["PEAK"],
            )
        else:
            print(
                [f"0x{w:02x}" for w in word], word_is_feature_one(word), current_event
            )
    return packets


def packetize_events(words, debug=False):  # waveform first
    packets = {}  # create_empty_packets(words)
    next = False
    current_event = 0
    current_event_string = f"Event_{current_event}"
    packets[f"Event_{current_event}"] = {}
    packets[f"Event_{current_event}"]["Data"] = []
    packets[f"Event_{current_event}"]["ADC"] = np.nan
    packets[f"Event_{current_event}"]["TSTMP"] = np.nan
    packets[f"Event_{current_event}"]["INT"] = np.nan
    packets[f"Event_{current_event}"]["PEAK"] = np.nan
    packets[f"Event_{current_event}"]["TOT"] = np.nan
    i = 0
    curr_ADC = 0
    while i < len(words):
        word = words[i]
        # if not word_is_feature_one(word):
        #    counter = word[1]<<16+word[2]<<8+word[3]
        #    print(f"CNT:{counter}")
        # else:
        #    print(" ")
        if debug:
            print(
                next,
                f"Event_{current_event}",
                [f"{byte:02x}" for byte in word],
                word_is_feature_one(word),
                word_is_feature_two(word),
                parse_waveform(word),
            )
        if word_is_feature_one(word):
            packets[f"Event_{current_event}"].update( parse_feature_one(word) )
        elif word_is_feature_two(word):
            packets[f"Event_{current_event}"].update( parse_feature_two(word) )
            if debug:
                print(packets[f"Event_{current_event}"])
            next = True
        else:
            packets[f"Event_{current_event}"]["Data"] += parse_waveform(word)

        if (next) & (i < len(words) - 1):
            # print("========== NEXT =============", next, next & i<len(words)-1 )
            current_event += 1
            current_event_string = f"Event_{current_event}"
            packets[f"Event_{current_event}"] = {}
            packets[f"Event_{current_event}"]["Data"] = []
            packets[f"Event_{current_event}"]["ADC"] = np.nan
            packets[f"Event_{current_event}"]["TSTMP"] = np.nan
            packets[f"Event_{current_event}"]["INT"] = np.nan
            packets[f"Event_{current_event}"]["PEAK"] = np.nan
            packets[f"Event_{current_event}"]["TOT"] = np.nan

            next = False

        i += 1
    return packets


def clean_up_packets(packets):
    for event in packets.keys():
        if ("Features1" in packets[event].keys()) & (
            "Features2" in packets[event].keys()
        ):
            if "Features2" in packets[event].keys():
                packets[event]["Features1"]["ToT"] = packets[event]["Features2"]["TOT"]
                packets[event].pop("Features2", None)
            else:
                print("No Features2")
            if "Features1" in packets[event].keys():
                for feat in packets[event]["Features1"].keys():
                    packets[event][feat] = packets[event]["Features1"][feat]
                packets[event].pop("Features1", None)
            else:
                print("No Features2")
        # else:
        # print(packets[event].keys())
        # print('Features1' in packets[event].keys())
        # print('Features2' in packets[event].keys())
        # print('Features1' in packets[event].keys() & 'Features2' in packets[event].keys())
        # print((('Features1' in packets[event].keys()) & ('Features2' in packets[event].keys())))
        # print("Neither Features1, nor Features2")

    return packets


def cast_to_dataframe(packets):
    frames = []
    for i, event in enumerate(packets.keys()):
        frames.append(pd.DataFrame([packets[event]]))
    df = pd.concat(frames)

    df.reset_index(inplace=True)
    return df


def run_on_new(target="LabZynq"):
    pull_new_files(target=target)
    new_dat_to_png()


def run_on_latest(target="LabZynq"):
    pull_latest(target=target)
    new_dat_to_png()


def pull_new_files(target="LabZynq"):
    existing = list_dat_files()
    on_server = list_dat_files_on_server(target=target)
    print(f"Localy {len(existing)} files are available.")
    print(f"{len(on_server)} files are available on server.")

    if on_server == []:
        return
    missing = [file for file in on_server if not (file in existing)]
    # assert len(missing) == len(on_server)-len(existing)
    print(f"Getting {len(missing)} files from {target}:")
    if len(missing) > 0:
        os.system(
            f'rsync --ignore-existing --include="*/" --include="*.dat" --exclude="*" {target}:/sd/dma_data/* ../'
        )
        os.system(
            f'rsync --ignore-existing --include="*/" --include="*.meta" --exclude="*" {target}:/sd/dma_data/* ../'
        )


def pull_latest(target="LabZynq"):
    existing = list_dat_files()
    on_server = list_dat_files_on_server(target=target)
    latest = on_server[0]
    if latest in existing:
        print("No new dat files to pull.")
        return
    if on_server == []:
        return
    print(latest)
    os.system("scp " + target + ":" + "/sd/dma_data/" + latest + " ../")


def list_dat_files_on_server(target="LabZynq") -> list:
    files = os.popen("ssh " + target + ' "ls -t /sd/dma_data/*.dat"').read().split("\n")
    files = [f for f in files if f != ""]
    files = [os.path.basename(f) for f in files]
    print(files)
    return files


def list_dat_files() -> list:
    list = glob.glob("../*.dat")
    list = [os.path.basename(file) for file in list]
    return list


def dat_to_df(file, debug=False):
    assert os.path.splitext(file)[1] == ".dat", print(
        f"Provided file {file} is not a dat file, got a {os.path.splitext(file)[1]} file"
    )
    with open(file, "rb") as f:
        bin_dat = f.read()
    len_words = int(len(bin_dat) / 16)
    words = binar_to_words(bin_dat, len_words)
    words = decode_words(words)

    if debug:
        for word in words:
            print(
                [f"{byte:02x}" for byte in word],
                word_is_feature_one(word),
                word_is_feature_two(word),
            )

    packets = packetize_events(words, debug=debug)
    # packets = clean_up_packets(packets)

    df = cast_to_dataframe(packets)
    return df


def parse_hv_get_voltage(line):
    drop_empty = lambda x: [y for y in x if y != ""]
    split = line.split(" ")
    clean = drop_empty(split)
    if len(clean) < 2:
        return np.nan
    try:
        retval = float(clean[1].strip().replace("0x", "").lstrip("0"))
    except:
        print(f"{line} caused the parser to crash")
        retval = np.nan
    return retval


def meta_to_df(file):
    # print(file)
    defaults_dict = {
        "ADC": [np.nan],
        "BASELINE_offset": [np.nan],
        "TRG_thr": [np.nan],
        "TRG_hyst": [np.nan],
        "TRG_dir": [np.nan],
        "WRITE_LENGHT_set": [np.nan],
        "WRITE_MODE_SEL": [np.nan],
        "WRITE_WAVEFORM_SOFT": [np.nan],
        "hv_get_voltage": [np.nan],
        "rate": [np.nan],
    }
    if not exists(file):

        warnings.warn(f"No meta data found for {file}")
        return pd.DataFrame(defaults_dict)
    with open(file, "r") as f:
        lines = f.readlines()
    drop_empty = lambda x: [y for y in x if y != ""]
    default_fcn = lambda x: (
        drop_empty(x.split(" "))[1].strip().replace("0x", "").lstrip("0")
    )
    converters = {
        "WRITE_MODE_SEL": lambda x: default_fcn(x),
        "WRITE_WAVEFORM_SOFT": lambda x: default_fcn(x),
        "TRG_thr": lambda x: default_fcn(x),
        "TRG_hyst": lambda x: default_fcn(x),
        "WRITE_LENGHT_set": lambda x: default_fcn(x),
        "rate": lambda x: float(x.replace("  ", " ").split(" ")[1]),
        "TRG_dir": lambda x: default_fcn(x),
        "BASELINE_offset": lambda x: default_fcn(x),
        "hv_get_voltage": lambda x: parse_hv_get_voltage(x),
    }

    dict = {}
    for line in lines:
        line = line.strip("\n")
        parname = line.replace("digitalcore_", "").replace("CTRL_", "").split(" ")[0]

        if line.startswith("set_digitalcore"):
            dict["ADC"] = int(line.split(" ")[1])
        elif parname in converters.keys():
            # print(line,line.split(' ')[1])
            # print(line, line.replace('digitalcore_',"").replace('CTRL_',"").split(' ')[0], line.split(' ')[1],line.replace('digitalcore_',"").replace('CTRL_',"").split(' ')[0] in converters.keys())
            dict[parname] = converters[parname](line)
    # print(dict)
    dict = {k: [v] for k, v in dict.items()}
    for key in defaults_dict.keys():
        if key not in dict.keys():
            dict[key] = defaults_dict[key]
    return pd.DataFrame(dict)


def data_to_df(file):
    dat = dat_to_df(file)
    meta = meta_to_df(file + ".meta")
    meta = pd.concat([meta] * len(dat.index), ignore_index=True)
    df = pd.concat([meta, dat], axis=1)
    return df


def dat_to_pkl(file):
    df = dat_to_df(file)
    print(f"Saving as {os.path.splitext(file)[0]+'.pkl'}")
    df.to_pickle(os.path.splitext(file)[0] + ".pkl")


def pkl_all():
    list = list_dat_files()
    for file in list:
        dat_to_pkl("../" + file)


def df_to_png(df, file, stacked=True):
    assert os.path.splitext(file)[1] == ".png"
    df = df.dropna()
    plt.figure()
    if stacked:
        for i, row in df.iterrows():
            plt.stairs(
                row["Data"],
                label=f"INT: {row['INT'] if ('INT' in row) else 'nan'}; PEAK: {row['PEAK'] if ('PEAK' in row) else 'nan'}",
            )
    else:
        offset = 0
        for i, row in df.iterrows():
            plt.stairs(
                row["Data"],
                np.arange(offset, offset + len(row["Data"]) + 1, 1),
                label=f"INT: {row['INT'] if ('INT' in row) else 'nan'}; PEAK: {row['PEAK'] if ('PEAK' in row) else 'nan'}",
            )
            offset += len(row["Data"])
    plt.legend(
        loc="center left",
        ncol=1,
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        mode="expand",
        borderaxespad=0.0,
    )
    plt.savefig(file)
    plt.close()


def pkl_to_png(file, stacked=True):
    assert os.path.splitext(file)[1] == ".pkl"
    df = pd.read_pickle(file)
    df_to_png(df, os.path.splitext(file)[0] + ".png", stacked=stacked)


def plot_pkls(stacked=True):
    files = glob.glob("../*.pkl")
    for file in files:
        pkl_to_png(file, stacked=stacked)


def all_dat_to_png(stacked=True):
    dats = glob.glob("../*.dat")
    for dat in dats:
        df = dat_to_df(dat)
        df_to_png(df, os.path.splitext(dat)[0] + ".png", stacked=stacked)


def dat_to_png(dat, stacked=True):
    df = dat_to_df(dat)
    df_to_png(df, os.path.splitext(dat)[0] + ".png", stacked=stacked)


def new_dat_to_png():
    dats = glob.glob("../*.dat")
    dats_stripped = [os.path.splitext(dat)[0] for dat in dats]
    pngs = glob.glob("../*.png")
    pngs_stripped = [os.path.splitext(png)[0] for png in pngs]
    missing_pngs = [dat for dat in dats_stripped if not (dat in pngs_stripped)]
    assert len(missing_pngs) == len(dats_stripped) - len(pngs_stripped)
    for dat in missing_pngs:
        print(f"Converting {dat} to png")
        df = dat_to_df(dat + ".dat")
        df_to_png(df, os.path.splitext(dat)[0] + ".png")


if __name__ == "__main__":
    if "-t" in sys.argv:
        idx = sys.argv.inedx("-t")
        target = sys.argv[idx + 1]
    else:
        target = "LabZynq"
    if "-l" in sys.argv:
        run_on_latest(target=target)
    if "-a" in sys.argv:
        run_on_new(target=target)
