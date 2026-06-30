import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

try:
    from scipy.datasets import electrocardiogram
except ImportError:
    try:
        from scipy.misc import electrocardiogram
    except ImportError:
        def electrocardiogram():
            t = np.linspace(0, 10, 3600)
            return signal.gausspulse(t - 3, fc=5)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 14, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3
})

OUT = "Graphics"
os.makedirs(OUT, exist_ok=True)

def load_data():
    ecg = electrocardiogram()
    fs = 360
    start = 17 * fs
    end = 27 * fs
    x = ecg[start:end]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = x - np.mean(x)
    t = np.arange(len(x)) / fs
    return t, x, fs

def inject_noise(t, x):
    np.random.seed(42)
    n_50hz = 0.10 * np.sin(2 * np.pi * 50 * t)
    n_resp = 0.20 * np.sin(2 * np.pi * 0.5 * t)
    n_rand = 0.05 * np.random.normal(0, 1, len(t))
    return x + n_50hz + n_resp + n_rand, n_50hz, n_resp, n_rand

def apply_filters(y, fs):
    b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs)
    y_notch = signal.filtfilt(b_notch, a_notch, y)
    nyq = 0.5 * fs
    b_band, a_band = signal.butter(4, [0.5/nyq, 40.0/nyq], btype='band')
    y_final = signal.filtfilt(b_band, a_band, y_notch)
    return y_notch, y_final

def detect_peaks(sig, height=0.4, distance=150):
    peaks, _ = signal.find_peaks(sig, height=height, distance=distance)
    return peaks

def calculate_snr(clean, target):
    c = clean - np.mean(clean)
    tt = target - np.mean(target)
    noise = tt - c
    p_s, p_n = np.sum(c**2), np.sum(noise**2)
    return 100.0 if p_n == 0 else 10*np.log10(p_s/p_n)

def peaks_to_binary_labels(ref_peaks, test_peaks, n_samples, tolerance=18):
    """Builds beat-by-beat True/False labels by matching detected peaks to reference peaks
    within a physiological tolerance window (~50ms at 360Hz)."""
    y_true = np.zeros(n_samples, dtype=int)
    y_pred = np.zeros(n_samples, dtype=int)
    y_true[ref_peaks] = 1
    matched_test = set()
    for rp in ref_peaks:
        window = np.where(np.abs(test_peaks - rp) <= tolerance)[0]
        if len(window) > 0:
            y_pred[rp] = 1
            matched_test.add(test_peaks[window[0]])
    # unmatched detected peaks = false positives, mark at their own index
    for tp in test_peaks:
        if tp not in matched_test:
            y_pred[tp] = 1
    return y_true, y_pred

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close()
    print(f"-> {name} oluşturuldu.")

def main():
    t, x_clean, fs = load_data()
    y_noisy, n_50hz, n_resp, n_rand = inject_noise(t, x_clean)
    y_notch, y_final = apply_filters(y_noisy, fs)

    snr_in = calculate_snr(x_clean, y_noisy)
    snr_out = calculate_snr(x_clean, y_final)
    print(f"Giriş SNR: {snr_in:.2f} dB | Çıkış SNR: {snr_out:.2f} dB")

    # --- GERÇEK peak detection ---
    ref_peaks = detect_peaks(x_clean, height=0.3, distance=150)
    noisy_peaks = detect_peaks(y_noisy, height=0.4, distance=150)
    clean_peaks = detect_peaks(y_final, height=0.3, distance=150)

    yt_n, yp_n = peaks_to_binary_labels(ref_peaks, noisy_peaks, len(x_clean))
    yt_c, yp_c = peaks_to_binary_labels(ref_peaks, clean_peaks, len(x_clean))

    cm_noisy = confusion_matrix(yt_n, yp_n, labels=[0, 1])
    cm_clean = confusion_matrix(yt_c, yp_c, labels=[0, 1])
    print("Gürültülü CM:\n", cm_noisy)
    print("Filtreli CM:\n", cm_clean)

    # Fig 1: Noise components
    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1); plt.plot(t, n_50hz, 'r'); plt.title('Bileşen 1: 50Hz Şebeke Gürültüsü')
    plt.subplot(3,1,2); plt.plot(t, n_resp, 'g'); plt.title('Bileşen 2: Solunum Kaynaklı Taban Kayması')
    plt.subplot(3,1,3); plt.plot(t, n_rand, 'orange'); plt.title('Bileşen 3: Rastgele Sensör Gürültüsü')
    plt.subplots_adjust(hspace=0.6)
    save_fig('Sekil_1_Gurultu_Bilesenleri.png')

    # Fig 2: Input comparison
    plt.figure(figsize=(12, 5))
    plt.plot(t, y_noisy, color='#e74c3c', alpha=0.6, label='Gürültülü Giriş')
    plt.plot(t, x_clean, color='black', linewidth=2, linestyle='--', label='Orijinal Temiz Sinyal')
    plt.title('Giriş Sinyali ve Referans Verinin Karşılaştırılması')
    plt.xlabel('Zaman (s)'); plt.ylabel('Genlik (mV)'); plt.legend(loc='upper right')
    save_fig('Sekil_2_Giris_Karsilastirma.png')

    # Fig 3: FFT
    freqs = np.fft.fftfreq(len(y_noisy), 1/fs)
    fft_vals = np.abs(np.fft.fft(y_noisy))
    mask = (freqs >= 0) & (freqs <= 120)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs[mask], fft_vals[mask], color='#8e44ad')
    plt.title('Gürültülü Sinyalin Frekans Spektrumu (FFT)')
    plt.xlabel('Frekans (Hz)'); plt.ylabel('Genlik')
    peak_y = np.max(fft_vals[mask])
    plt.annotate('50Hz Şebeke Piki', xy=(50, peak_y*0.8), xytext=(60, peak_y),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    save_fig('Sekil_3_FFT_Analizi.png')

    # Fig 4: Spectrogram
    plt.figure(figsize=(10, 6))
    f, t_spec, Sxx = signal.spectrogram(y_noisy, fs)
    plt.pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-10), shading='gouraud', cmap='inferno')
    plt.title('Zaman-Frekans Analizi (Spektrogram)')
    plt.ylabel('Frekans [Hz]'); plt.xlabel('Zaman [s]')
    plt.colorbar(label='Güç (dB)'); plt.ylim(0, 100)
    save_fig('Sekil_4_Spektrogram.png')

    # Fig 5: Notch output
    plt.figure(figsize=(12, 4))
    plt.plot(t, y_noisy, color='lightgray', label='Giriş', alpha=0.8)
    plt.plot(t, y_notch, color='#2980b9', label='Notch Çıkışı', linewidth=1.2)
    plt.title('Ara Aşama: Notch Filtre Sonrası (50Hz Bastırıldı)'); plt.legend()
    save_fig('Sekil_5_Notch_Cikis.png')

    # Fig 6: Final
    plt.figure(figsize=(12, 5))
    plt.plot(t, x_clean, color='gray', alpha=0.6, linewidth=3, label='Referans (Orijinal)')
    plt.plot(t, y_final, color='#27ae60', linewidth=1.5, label='Filtrelenmiş Sinyal')
    plt.title('Nihai Sonuç: Bandpass + Notch Filtreleme')
    plt.xlabel('Zaman (s)'); plt.legend(loc='upper right')
    save_fig('Sekil_6_Final_Sonuc.png')

    # Fig 7: Zoom
    plt.figure(figsize=(10, 5))
    zs, ze = int(2.0*fs), int(3.5*fs)
    plt.plot(t[zs:ze], y_noisy[zs:ze], '#e74c3c', alpha=0.3, label='Gürültülü')
    plt.plot(t[zs:ze], x_clean[zs:ze], 'k--', alpha=0.5, label='Orijinal', linewidth=2)
    plt.plot(t[zs:ze], y_final[zs:ze], '#27ae60', linewidth=2, label='Temizlenmiş')
    plt.title('Sinyal Detay Analizi (QRS Kompleksi)'); plt.legend(loc='upper right')
    save_fig('Sekil_7_Zoom_Detay.png')

    # Fig 8: SNR
    plt.figure(figsize=(8, 6))
    gain = snr_out - snr_in
    bars = plt.bar(['Giriş (Kirli)', 'Çıkış (Temiz)'], [snr_in, snr_out], color=['#c0392b', '#27ae60'])
    plt.title(f'SNR İyileşmesi: +{gain:.1f} dB'); plt.ylabel('SNR (dB)')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2., h+0.5, f'{h:.1f} dB', ha='center', va='bottom', fontsize=14, weight='bold')
    plt.ylim(min(0, snr_in)-5, snr_out+5)
    save_fig('Sekil_8_SNR_Metrik.png')

    # Fig 9: Confusion matrices (REAL)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_noisy, annot=True, fmt='d', cmap='Reds', ax=ax[0], cbar=False, annot_kws={"size": 14})
    ax[0].set_title('Gürültülü Sinyal QRS Tespiti'); ax[0].set_xlabel('Tahmin Edilen'); ax[0].set_ylabel('Gerçek Durum')
    ax[0].set_xticklabels(['Atış Yok', 'Atış Var']); ax[0].set_yticklabels(['Atış Yok', 'Atış Var'])
    sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Greens', ax=ax[1], cbar=False, annot_kws={"size": 14})
    ax[1].set_title('Filtreli Sinyal QRS Tespiti'); ax[1].set_xlabel('Tahmin Edilen'); ax[1].set_ylabel('Gerçek Durum')
    ax[1].set_xticklabels(['Atış Yok', 'Atış Var']); ax[1].set_yticklabels(['Atış Yok', 'Atış Var'])
    plt.suptitle('Kalp Atışı Tespiti Doğruluk Matrisi (Gerçek Veriden Hesaplanmış)')
    save_fig('Sekil_9_Confusion_Matrix.png')

    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(f"SNR_IN={snr_in:.2f}\nSNR_OUT={snr_out:.2f}\nGAIN={gain:.2f}\n")
        f.write(f"CM_NOISY={cm_noisy.tolist()}\nCM_CLEAN={cm_clean.tolist()}\n")
        f.write(f"REF_PEAKS={len(ref_peaks)}\n")

if __name__ == "__main__":
    main()
