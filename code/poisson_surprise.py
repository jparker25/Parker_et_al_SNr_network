import numpy as np
from matplotlib import pyplot as plt
import math
import sys


def calc_surprise(rate, burst_candidate):
    interval = burst_candidate[-1] - burst_candidate[0]
    p = np.exp(-rate * interval) * np.sum(
        [
            (rate * interval) ** i / math.factorial(i)
            for i in range(len(burst_candidate), np.max([len(burst_candidate) * 4, 20]))
        ]
    )
    return -np.log(p)


def run_poisson_surprise(
    spike_times, min_spikes=3, max_spikes=10, surprise_threshold=5.0, window=-1
):
    """
    Detect bursts in spike trains using the Poisson surprise method.

    Parameters:
        spike_times (array_like): Spike times in seconds.
        min_spikes (int): Minimum number of spikes in a burst.
        surprise_threshold (float): Threshold for the surprise value to consider a burst.

    Returns:
        bursts (list of tuples): List of tuples representing the start and end times of detected bursts.
    """
    if len(spike_times) < min_spikes:
        return []

    # Calculate inter-spike intervals and the background firing rate
    isi = np.diff(spike_times)
    rate = 1 / np.mean(isi)  # spikes per second

    bursts = []

    spike_i = 0
    while spike_i < len(spike_times) - min_spikes - 1:
        burst_candidate = spike_times[spike_i : spike_i + min_spikes]
        if window != -1:
            window = 2 * np.max(isi)
            window_spikes = spike_times[
                ((spike_times) >= spike_times[spike_i] - window)
                & (((spike_times) < spike_times[spike_i]))
            ]
            rate = window_spikes.shape[0] / window if spike_i > min_spikes else rate
        if np.max(np.diff(burst_candidate)) <= 0.5 * np.mean(isi):
            surprise = calc_surprise(rate, burst_candidate)
            add_forward = True
            while add_forward and len(burst_candidate) < max_spikes:
                new_burst_candidate = spike_times[
                    spike_i : spike_i + burst_candidate.shape[0] + 1
                ]
                new_surprise = calc_surprise(rate, new_burst_candidate)
                if (
                    new_surprise > surprise
                    and new_burst_candidate.shape[0] >= min_spikes
                ):
                    surprise = new_surprise
                    burst_candidate = new_burst_candidate
                else:
                    add_forward = False
            remove_front = True
            while remove_front:
                new_burst_candidate = burst_candidate[1:]
                new_surprise = calc_surprise(rate, new_burst_candidate)
                if (
                    new_surprise > surprise
                    and new_burst_candidate.shape[0] >= min_spikes
                ):
                    surprise = new_surprise
                    burst_candidate = new_burst_candidate
                else:
                    remove_front = False
            bursts.append([surprise, burst_candidate])
            spike_i = np.where(spike_times == burst_candidate[-1])[0][0] + 1

        else:
            spike_i += 1
    return [burst for burst in bursts if burst[0] >= surprise_threshold]


def burst_statistics(spike_times, bursts, recording_length):
    n_bursts = len(bursts)
    burst_firing_rate = np.zeros(n_bursts)  #
    burst_start_times = np.zeros(n_bursts)
    burst_durations = np.zeros(n_bursts)
    burst_spikes = np.zeros(n_bursts)
    burst_surprise = np.zeros(n_bursts)
    inter_burst_intervals = np.zeros(n_bursts - 1)
    burst_count = 0
    for burst in bursts:
        burst_firing_rate[burst_count] = len(burst[1]) / (burst[1][-1] - burst[1][0])
        burst_start_times[burst_count] = burst[1][0]
        burst_durations[burst_count] = burst[1][-1] - burst[1][0]
        burst_spikes[burst_count] = len(burst[1])
        burst_surprise[burst_count] = burst[0]
        burst_count += 1

    for i in range(n_bursts - 1):
        inter_burst_intervals[i] = bursts[i + 1][1][0] - bursts[i][1][-1]

    avg_burst_firing_rate = np.mean(burst_firing_rate)  # average firing rate in a burst
    percent_time_bursting = (
        np.sum(burst_durations) / recording_length
    )  # percent of recording time spent bursting
    percent_spike_bursting = (
        np.sum(burst_spikes) / spike_times.shape[0]
    )  # percent of spikes in baseline part of burst
    avg_burst_duration = np.mean(burst_durations)  # average length of burst

    avg_inter_burst_interval = np.mean(inter_burst_intervals) if len(bursts) > 1 else 0
    cv_inter_burst_interval = (
        np.std(np.diff(inter_burst_intervals)) / np.mean(np.diff(inter_burst_intervals))
        if len(bursts) > 2
        else 0
    )
    avg_surprise = np.mean(burst_surprise)
    non_bursting_firing_rate = (spike_times.shape[0] - np.sum(burst_spikes)) / (
        recording_length - np.sum(burst_durations)
    )
    burst_firing_rate_increase = (
        avg_burst_firing_rate / non_bursting_firing_rate
        if non_bursting_firing_rate > 0
        else 0
    )

    if math.isinf(avg_surprise):
        print(burst_surprise)

    return [
        n_bursts,
        avg_burst_firing_rate,
        percent_time_bursting,
        percent_spike_bursting,
        avg_burst_duration,
        avg_inter_burst_interval,
        cv_inter_burst_interval,
        avg_surprise,
        non_bursting_firing_rate,
        burst_firing_rate_increase,
    ]
