# MATLAB App Designer Integration Guide

This guide outlines exactly how to build the GUI around the exported ONNX model.

## 1. Importing the Network
Use the Deep Learning Toolbox to import the exported Transformer ONNX file:
```matlab
% In your App Designer startup callback
netName = 'rtn_denoiser.onnx';
app.Net = importONNXNetwork(netName, 'OutputLayerType', 'regression');
```

## 2. Generating the Lorentzian PSD
To visualize the Random Telegraph Noise characteristics, calculate the Power Spectral Density (PSD) using `pwelch`:
```matlab
fs = 1e9; % Sampling frequency (1 GHz for 1 ns dt)
[pxx, f] = pwelch(app.NoisySignal, [], [], [], fs);
plot(app.UIAxes_PSD, f, 10*log10(pxx));
set(app.UIAxes_PSD, 'XScale', 'log'); % Lorentzian knee visible in log scale
```

## 3. The Butterworth Comparison
Implement the classical 2nd-order low-pass Butterworth filter:
```matlab
fc = app.CutoffSlider.Value; % Cut-off from user interface
[b, a] = butter(2, fc/(fs/2));
classical_denoised = filtfilt(b, a, app.NoisySignal); % Zero-phase offline, or 'filter' for real-time lag
```

## 4. AI Inference Overlay
```matlab
% Prepare sequence
inputTensor = dlarray(reshape(app.NoisySignal, [1, 1, length(app.NoisySignal)]), 'CBT');

% Predict
[cleanSeqDl, paramsDl] = predict(app.Net, inputTensor);
app.CleanSignalAI = extractdata(cleanSeqDl);
app.Tau_c = extractdata(paramsDl(1));
app.Tau_e = extractdata(paramsDl(2));

% Plot overlay
plot(app.UIAxes_Main, app.Time, app.CleanSignalAI, 'LineWidth', 2, 'DisplayName', 'Transformer (Zero-Lag)');
hold(app.UIAxes_Main, 'on');
plot(app.UIAxes_Main, app.Time, classical_denoised, 'r--', 'DisplayName', 'Butterworth');
legend(app.UIAxes_Main, 'show');
```

## 5. Exporting PDF Report
```matlab
exportapp(app.UIFigure, 'RTN_Analysis_Report.pdf');
```
