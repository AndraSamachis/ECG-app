import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys



def sinc(x):
    if x == 0:
        return 1
    else:
        return np.sin(np.pi * x) / (np.pi * x)

Fe = 360

def lowpass_filter_coeffs(fc, Fe, N):
    Ft = fc / Fe
    h = np.zeros(N)
    M = (N - 1) // 2
    for n in range(-M, M + 1):
        if n == 0:
            h[n + M] = 2 * Ft
        else:
            h[n + M] = 2 * Ft * sinc(2 * np.pi * Ft * n)
    return h

def highpass_filter_coeffs(fc, Fe, N):
    Ft = fc / Fe
    h = np.zeros(N)
    M = (N - 1) // 2
    for n in range(-M, M + 1):
        if n == 0:
            h[n + M] = 1 - 2 * Ft
        else:
            h[n + M] = sinc(n) - 2 * Ft * sinc(n * 2 * Ft)
    return h


#O(M*N)
def apply_filter(x, h):
    M = len(x)
    N = len(h)
    y = np.zeros(M + N - 1)
    for n in range(len(y)):
        for k in range(N):
            if n - k >= 0 and n - k < M:
                y[n] += h[k] * x[n - k]
    return y

# diferentiere
def differentiate(x):
    return np.diff(x, n=1)

def square(x):
    return np.power(x, 2)

def moving_window_integration(x, window_size):
    window = np.ones(window_size) / window_size
    return apply_filter(x, window)


def detect_peaks(ecg_signal, integrated_result, fs):
    possible_peaks = []
    signal_peaks = []
    r_peaks = []
    SPKI = 0  # estimare curenta a amplitudinii varfurilor semnalului in semnalul integrat
    SPKF = 0
    NPKI = 0   # estimarea curenta a varfului de zgomot
    NPKF = 0
    rr_avg_one = []
    THRESHOLDI1 = 0
    THRESHOLDF1 = 0
    THRESHOLDF2 = 0
    window = round(0.15 * fs)

    FM_peaks = []  # fiducial mask
    # diff calculeaza diferenta dintre elementele consecutive ale unui array
    localDiff = np.diff(integrated_result)

    #dectez varfurile bazate pe trecrea de la o dif + la una -
    for i in range(1, len(localDiff)):
        if i - window > 1 and localDiff[i - 1] > 0 and localDiff[i] < 0:
            FM_peaks.append(i - 1)  # adaug indexul varfului in lista FM_peaks

    #definesc fereastra in jurul varfului
    for index in range(len(FM_peaks)):
        current_peak = FM_peaks[index]  # pozitia actuala a varfului
        left_limit = max(current_peak - window, 0) #lim > 0
        right_limit = min(current_peak + window + 1, len(ecg_signal)) #lim < len(signal)
        max_index = -1
        max_value = -sys.maxsize
        for i in range(left_limit, right_limit):
            if ecg_signal[i] > max_value:
                max_value = ecg_signal[i]
                max_index = i
        if max_index != -1:
            possible_peaks.append(max_index)  # potentialele varfuri - stocam indexul lor

        PEAKF = ecg_signal[max_index]
        PEAKI = integrated_result[current_peak]

        # daca e primul varf sau sunt varfuri ulterioare nedetectate pana in acel moment
        if index == 0:
            if PEAKI >= THRESHOLDI1:
                SPKI = 0.125 * PEAKI + 0.875 * SPKI
                if PEAKF > THRESHOLDF1:
                    SPKF = 0.125 * PEAKF + 0.875 * SPKF
                    signal_peaks.append(max_index)
                else:
                    NPKF = 0.125 * PEAKF + 0.875 * NPKF
            else:
                NPKI = 0.125 * PEAKI + 0.875 * NPKI
                NPKF = 0.125 * PEAKF + 0.875 * NPKF
        else:
            # selectam de la index-8 pana la index
            recent8_FM_peaks = FM_peaks[max(0, index - 7):index + 1]
            # diferentele intre RR-intervale exprimate in secunde pentru 8 cele mai recente vfuri FM
            RRAVERAGE1 = np.diff(recent8_FM_peaks) / fs
            rr_one_mean = np.mean(RRAVERAGE1)  # media dintre cele mai recente 8 batai
            rr_avg_one.append(rr_one_mean)
            # media dintre cele mai recente 8 batai avand intervalele RR care se incadreaza in limitele resp


            if index >= 8:
                rr_avg_two = [RR for RR in RRAVERAGE1 if 0.92 * rr_one_mean < RR < 1.16 * rr_one_mean]
                if len(rr_avg_two) == 8:
                    RRAVERAGE2 = np.mean(rr_avg_two)
                else:
                    RRAVERAGE2 = rr_one_mean
            else:
                RRAVERAGE2 = rr_one_mean


            RR_LOW_LIMIT = 0.92 * RRAVERAGE2
            RR_MISSED_LIMIT = 1.66 * RRAVERAGE2
            # ajusteaza pragurile in functie de activitatea recenta a inimii
            if rr_avg_one[-1] < RR_LOW_LIMIT or rr_avg_one[-1] > RR_MISSED_LIMIT:
                THRESHOLDI1 /= 2
                THRESHOLDF1 /= 2

            # CAUTARE INAPOI
            current_rr_interval = RRAVERAGE1[-1] # utlimul element (ultima diferenta) din RRAVERAGE1
            #stabilim o fereastra in functie de lungimea ultimului interval rr
            search_back_window = round(current_rr_interval * fs) #convertire in nr de esantioane
            #if conditie => presupunem ca o bataie ar putea fi pierduta
            if current_rr_interval > RR_MISSED_LIMIT:
                left_limit = current_peak - search_back_window + 1
                right_limit = current_peak + 1
                search_back_max_index = -1
                max_value = -sys.maxsize
                #cautam val max in semnalul integrat
                for i in range(left_limit, right_limit):
                    if integrated_result[i] > THRESHOLDI1 and integrated_result[i] > max_value:
                        max_value = integrated_result[i]
                        search_back_max_index = i
                PEAKI = integrated_result[search_back_max_index]
                #daca am gasit un maxim
                if search_back_max_index != -1:
                    SPKI = 0.25 * PEAKI + 0.75 * SPKI

                    #stabilim noi limite
                    left_limit = search_back_max_index - round(0.15 * fs) # start cautare inapoi
                    right_limit = min(len(ecg_signal), search_back_max_index) # finish cautare inapoi

                    search_back_max_index2 = -1
                    max_value = -sys.maxsize
                    #cautam si in semnalul filtrat
                    for i in range(left_limit, right_limit):
                        if ecg_signal[i] > THRESHOLDF1 and ecg_signal[i] > max_value:
                            max_value = ecg_signal[i]
                            search_back_max_index2 = i

                    if search_back_max_index2 != -1:
                        PEAKF = ecg_signal[search_back_max_index2]
                        if PEAKF > THRESHOLDF2:
                            SPKF = 0.25 * PEAKF + 0.75 * SPKF

                            signal_peaks.append(search_back_max_index2)



        THRESHOLDI1 = NPKI + 0.25 * (SPKI - NPKI)
        THRESHOLDF1 = NPKF + 0.25 * (SPKF - NPKF)
        THRESHOLDF2 = 0.5 * THRESHOLDF1


    for peak_index in np.unique(signal_peaks):
        peak_index = int(peak_index)
        window = round(0.15 * fs) #fereastra este de 0.15 secunde, 54 esantioane
        #index vf - dim ferestrei
        left_limit = max(peak_index - window, 0)
        # de la index vf adunam dim ferestrei + 1 ca sa includem si indexul vfului
        right_limit = min(peak_index + window + 1, len(ecg_signal))
        max_value = -sys.maxsize
        max_index = -1
        # parcurgem fiecare punct din interiorul ferestrei
        # cautam val maxima
        for j in range(left_limit, right_limit):
            if ecg_signal[j] > max_value:
                max_value = ecg_signal[j]
                max_index = j
        r_peaks.append(max_index)

    return r_peaks



def detect_q_peaks(ecg_signal, r_peaks, fs):
    q_peaks = []
    samples_before_r = range(-5, -21, -1)  # Interval de la -5 la -20 esantioane

    for r_peak in r_peaks:
        min_value = sys.maxsize
        min_index = r_peak

        for offset in samples_before_r:
            sample_index = r_peak + offset
            if sample_index >= 0 and ecg_signal[sample_index] < min_value:
                min_value = ecg_signal[sample_index]
                min_index = sample_index

        q_peaks.append(min_index)

    return q_peaks

def detect_s_peaks(ecg_signal, r_peaks, fs):
    s_peaks = []
    samples_after_r = range(1, 26)  # Interval de la +1 la +25 esantioane

    for r_peak in r_peaks:
        min_value = sys.maxsize
        min_index = r_peak

        for offset in samples_after_r:
            sample_index = r_peak + offset
            if sample_index < len(ecg_signal) and ecg_signal[sample_index] < min_value:
                min_value = ecg_signal[sample_index]
                min_index = sample_index

        s_peaks.append(min_index)

    return s_peaks


def load_ecg_data(file_path):
    ecg_data = pd.read_csv(file_path, header=None)
    ecg_signal = ecg_data[0].values
    return ecg_signal

def save_plot(data, time_sec, title, ylabel, output_file, peaks=None):
    plt.figure(figsize=(15, 8), facecolor = '#FAF8EF')
    plt.plot(time_sec, data, color='blue', label='Semnal ECG')
    #if peaks is not None:
     #   plt.scatter(np.array(peaks) / Fe, [data[i] for i in peaks], color='red', label='Peaks', marker='o')
    plt.title(title)
    plt.xlabel('Timp (s)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def process_and_save_ecg(file_path):
    Fe = 360
    ecg_signal = load_ecg_data(file_path)

    fc_low = 27
    N_low = 21
    lowpass_h = lowpass_filter_coeffs(fc_low, Fe, N_low)

    fc_high = 9
    N_high = 21
    highpass_h = highpass_filter_coeffs(fc_high, Fe, N_high)
    print(len(ecg_signal))
    ecg_segment = ecg_signal[:1800]
    y_low = apply_filter(ecg_segment, lowpass_h)
    y_bandpass = apply_filter(y_low, highpass_h)[:len(ecg_segment)]

    time_sec = np.arange(len(ecg_segment)) / Fe

    image_files = []
    save_plot(ecg_segment, time_sec, 'Semnal ECG initial', 'Amplitudine (mV)', 'original.png')
    image_files.append('original.png')


    save_plot(y_bandpass, time_sec, 'Semnal cu filtru trece sus', 'Amplitude (mV)', 'ecg_highpass_filtered.png')
    image_files.append('ecg_highpass_filtered.png')

    y_diff = differentiate(y_bandpass)
    time_sec_diff = time_sec[:-1]


    save_plot(y_diff, time_sec_diff, 'Semnal derivat', 'Amplitude (mV)', 'ecg_differentiated.png')
    image_files.append('ecg_differentiated.png')
    y_square = square(y_diff)


    save_plot(y_square, time_sec_diff, 'Semnal ridicat la patrat', 'Amplitude (mV)', 'ecg_squared.png')
    image_files.append('ecg_squared.png')
    window_size = int(0.15 * Fe) #54 esantioane
    y_mwi = moving_window_integration(y_square, window_size)[:len(ecg_segment)]


    save_plot(y_mwi, time_sec, 'Semnal preprocesat', 'Amplitudine (mV)', 'ecg_integrated.png')
    image_files.append('ecg_integrated.png')
    r_peaks = detect_peaks(y_bandpass, y_mwi, Fe)
    q_peaks = detect_q_peaks(y_bandpass, r_peaks, Fe)
    s_peaks = detect_s_peaks(y_bandpass, r_peaks, Fe)

    #RR_intervals = np.diff(r_peaks) / Fe
    #pulse = 60 / np.mean(RR_intervals)


    plt.figure(figsize=(15, 8), facecolor = '#FAF8EF')
    plt.plot(time_sec, y_bandpass, color='blue', label='Semnal ECG')
    plt.scatter(np.array(r_peaks) / Fe, [y_bandpass[i] for i in r_peaks], color='red', label='Vârfuri R', marker='o')
    plt.scatter(np.array(q_peaks) / Fe, [y_bandpass[i] for i in q_peaks], color='green', label='Vârfuri Q', marker='o')
    plt.scatter(np.array(s_peaks) / Fe, [y_bandpass[i] for i in s_peaks], color='purple', label='Vârfuri S', marker='o')

    plt.title(f'Semnalul ECG cu vârfurile Q, R, S')
    plt.xlabel('Timp (s)')
    plt.ylabel('Amplitudine (mV)')
    plt.legend()
    plt.grid(True)
    plt.savefig('ecg_final_with_peaks.png')
    image_files.append('ecg_final_with_peaks.png')
    plt.close()

    return image_files
#
# if __name__ == "__main__":
#     process_and_save_ecg(file_path)