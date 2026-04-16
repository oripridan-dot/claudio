#include <cmath>
#include <vector>
#include <algorithm>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

extern "C" {

EMSCRIPTEN_KEEPALIVE
float computeRMS(const float* buffer, int length) {
    if (length <= 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        sum += buffer[i] * buffer[i];
    }
    return std::sqrt(sum / length);
}

EMSCRIPTEN_KEEPALIVE
void autocorrelationF0(const float* buffer, int length, float sampleRate, float* outF0, float* outConfidence) {
    int minPeriod = static_cast<int>(sampleRate / 1046.0f); // C6 max
    int maxPeriod = static_cast<int>(sampleRate / 40.0f);   // E1 min
    if (maxPeriod >= length) maxPeriod = length - 1;

    if (maxPeriod <= minPeriod) {
        *outF0 = 0.0f;
        *outConfidence = 0.0f;
        return;
    }

    float energy = 0.0f;
    for (int i = 0; i < length; i++) {
        energy += buffer[i] * buffer[i];
    }
    float rms = std::sqrt(energy / length);
    if (rms < 0.005f) {
        *outF0 = 0.0f;
        *outConfidence = 0.0f;
        return; // Silence gate
    }

    float bestCorr = 0.0f;
    int bestPeriod = 0;

    for (int period = minPeriod; period <= maxPeriod; period++) {
        float corr = 0.0f;
        float e1 = 0.0f;
        float e2 = 0.0f;
        int len = length - period;
        for (int i = 0; i < len; i++) {
            corr += buffer[i] * buffer[i + period];
            e1 += buffer[i] * buffer[i];
            e2 += buffer[i + period] * buffer[i + period];
        }
        float norm = std::sqrt(e1 * e2);
        float normCorr = norm > 0 ? (corr / norm) : 0.0f;

        if (normCorr > bestCorr) {
            bestCorr = normCorr;
            bestPeriod = period;
        }
    }

    if (bestCorr < 0.5f || bestPeriod == 0) {
        *outF0 = 0.0f;
        *outConfidence = 0.0f;
        return;
    }

    *outF0 = sampleRate / bestPeriod;
    *outConfidence = bestCorr;
}

EMSCRIPTEN_KEEPALIVE
float computeSpectralCentroid(const float* fft, int length, float sampleRate) {
    float weightedSum = 0.0f;
    float totalMag = 0.0f;
    float binWidth = sampleRate / (length * 2.0f);
    
    for (int i = 0; i < length; i++) {
        float mag = std::abs(fft[i]);
        weightedSum += mag * (i * binWidth);
        totalMag += mag;
    }
    return totalMag > 0 ? (weightedSum / totalMag) : 0.0f;
}

// MFCC State
static const int N_MFCC = 13;
static const int N_MELS = 26;
static std::vector<std::vector<float>> filterbank;

EMSCRIPTEN_KEEPALIVE
void initFilterbank(int fftSize, float sampleRate) {
    int numBins = fftSize / 2;
    float maxFreq = sampleRate / 2.0f;
    float maxMel = 2595.0f * std::log10(1.0f + maxFreq / 700.0f);

    std::vector<float> melPoints(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; i++) {
        float mel = (i * maxMel) / (N_MELS + 1);
        melPoints[i] = 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }

    std::vector<int> binPoints(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; i++) {
        binPoints[i] = static_cast<int>((numBins * melPoints[i]) / maxFreq);
    }

    filterbank.clear();
    filterbank.resize(N_MELS, std::vector<float>(numBins, 0.0f));

    for (int i = 0; i < N_MELS; i++) {
        for (int j = binPoints[i]; j < binPoints[i + 1]; j++) {
            filterbank[i][j] = (j - binPoints[i]) / static_cast<float>(binPoints[i + 1] - binPoints[i] + 1e-10f);
        }
        for (int j = binPoints[i + 1]; j < binPoints[i + 2]; j++) {
            filterbank[i][j] = (binPoints[i + 2] - j) / static_cast<float>(binPoints[i + 2] - binPoints[i + 1] + 1e-10f);
        }
    }
}

EMSCRIPTEN_KEEPALIVE
void computeMFCC(const float* magSpectrum, int spectrumLength, float* outMfcc) {
    if (filterbank.empty()) return;

    std::vector<float> melEnergies(N_MELS, 0.0f);
    for (int i = 0; i < N_MELS; i++) {
        float energy = 0.0f;
        int filterLen = filterbank[i].size();
        int maxLen = std::min(spectrumLength, filterLen);
        for (int j = 0; j < maxLen; j++) {
            energy += magSpectrum[j] * filterbank[i][j];
        }
        melEnergies[i] = std::log(std::max(energy, 1e-10f));
    }

    for (int k = 0; k < N_MFCC; k++) {
        float sum = 0.0f;
        for (int n = 0; n < N_MELS; n++) {
            sum += melEnergies[n] * std::cos((M_PI * k * (n + 0.5f)) / N_MELS);
        }
        outMfcc[k] = sum;
    }
}

} // extern "C"
