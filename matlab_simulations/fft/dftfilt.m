function g = dftfilt(f, H, classout)
%DFTFILT Performs frequency domain filtering.
%   g = DFTFILT(f, H, classout) filters f in the frequency domain using the
%   filter transfer function H. The output, g, is the filtered
%   image, which has the same size as f. 
% Valid values of CLASSOUT are
% 'original'    The output is of the same class as the input.
%               This is the default if CLASSOUT is not included
%               in the call.
% 'fltpoint'    The output is floating point of class single, unless
%               both f ans H are of class double, in which case the
%               output also is of class double.

% DFTFILT automatically pads f to be the same size as H.
%   DFTFILT assumes that f is real and that H is a real, uncentered
%   circularly-symmetric filter function. 

% Convert the input to floating point.
[f, revertClass] = tofloat(f);

% Obtain the FFT of the padded input.
F = fft2(f, size(H, 1), size(H, 2));

% Perform filtering. 
g = real(ifft2(H.*F));

% Crop to original size.
g = g(1:size(f, 1), 1:size(f, 2)); % g is of class single here.

% Convert the output to the same class as the input if so specified.
if nargin == 2 || strcmp(classout, 'original')
    g = revertClass(g);
elseif strcmp(classout, 'fltpoint')
    return
else
    error('Undefined class for the output image.')
end